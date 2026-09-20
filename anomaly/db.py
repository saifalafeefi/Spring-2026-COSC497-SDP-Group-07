"""the master's store: who was measured, on what, when, and what the model said.

everything before this lived in three places that all forget: a numpy file per
BOARD, the browser's localStorage, and the master's RAM. that made three things
impossible -- telling two people apart on one board, keeping a flag history that
outlives a browser tab, and recording an induced-stress session at all.

the model here is deliberately not device-centric:

    subject   a person. owns a BASELINE, because a baseline describes a body,
              not a board. calibration used to be keyed to the MAC, so two
              people sharing a board silently shared one baseline.
    device    a board. owns nothing but its identity and where it was last seen.
    session   a subject on a device for a stretch of time. everything measured
              hangs off a session, so a score always knows whose body, which
              hardware and which model produced it.

sqlite, stdlib, one file, one writer (the master). no server to run and nothing
new in requirements.txt. at 1 Hz a nine-minute protocol session is 540 rows.

RAW WAVEFORM IS NOT STORED HERE. the project's privacy property is that raw
physiological data stays on the device; recording it is an explicit per-session
act, and then only the FILE PATH lands in `session.raw_path`.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time

# subject data, never model artifacts -- kept out of anomaly/saved/ (which is
# partly committed) so it cannot be committed by reflex.
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pulse.db")

SCHEMA_VERSION = 6

KINDS = ("calibration", "monitor", "protocol")
PHASES = ("baseline", "induction", "recovery")
VERDICTS = ("real", "artifact", "unsure")

_SCHEMA = """
CREATE TABLE subject (
  id            INTEGER PRIMARY KEY,
  code          TEXT    NOT NULL UNIQUE,      -- 'S01', the label used in results
  display_name  TEXT,
  created       REAL    NOT NULL,
  notes         TEXT
);

CREATE TABLE device (
  id            TEXT    PRIMARY KEY,          -- 'pulse-a4f2c1', MAC-derived
  first_seen    REAL    NOT NULL,
  last_seen     REAL    NOT NULL,
  last_ip       TEXT,
  fw_note       TEXT,
  name          TEXT,                          -- what a person calls this board
  -- who is wearing this board. sessions open and close on their own as contact
  -- comes and goes, so this is what has to persist between them: without it a
  -- lifted finger would lose the identity along with the session.
  subject_id    INTEGER REFERENCES subject(id)
);

-- which model produced a score. a threshold is meaningless without it: swap the
-- model and every stored level changes scale.
CREATE TABLE model (
  id            INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL,
  sha           TEXT,                          -- digest of the artifact
  created       REAL    NOT NULL,
  notes         TEXT,
  UNIQUE(name, sha)
);

CREATE TABLE session (
  id            INTEGER PRIMARY KEY,
  subject_id    INTEGER NOT NULL REFERENCES subject(id),
  device_id     TEXT    NOT NULL REFERENCES device(id),
  model_id      INTEGER REFERENCES model(id),
  kind          TEXT    NOT NULL CHECK (kind IN ('calibration','monitor','protocol')),
  started       REAL    NOT NULL,
  ended         REAL,                          -- NULL = still open
  sens          REAL,                          -- operating point in force
  raw_path      TEXT,                          -- set only if raw was recorded
  notes         TEXT
);

-- a subject's calm reference. `active` rather than delete: a result published
-- against an old baseline must stay reproducible.
CREATE TABLE baseline (
  id            INTEGER PRIMARY KEY,
  subject_id    INTEGER NOT NULL REFERENCES subject(id),
  session_id    INTEGER REFERENCES session(id),
  model_id      INTEGER REFERENCES model(id),
  threshold     REAL    NOT NULL,
  ref_lo        REAL    NOT NULL,
  ref_hi        REAL    NOT NULL,
  n_windows     INTEGER,
  fs            INTEGER,
  win_len       INTEGER,
  source        TEXT,                          -- 'device' | 'wesad' | 'import'
  -- The one number that survives a re-wear: how many robust sigmas above this
  -- person's OWN live calm their flag sits. ref_lo/ref_hi record where calm sat
  -- on the calibration day, and that moves the moment the sensor is re-placed
  -- (median 1.60 sigmas, p90 10.60, measured on 27 subjects at two placements).
  -- k_sigma does not move, so one calibration lasts.
  k_sigma       REAL,
  created       REAL    NOT NULL,
  active        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE reading (
  session_id    INTEGER NOT NULL REFERENCES session(id),
  t             REAL    NOT NULL,              -- epoch seconds, UTC
  score         REAL,                          -- raw reconstruction error
  level         REAL,                          -- 0-1, against this subject's baseline
  flag          INTEGER,
  bpm           REAL,
  spo2          REAL,
  contact       INTEGER,
  quality       REAL,                          -- 0-1 from anomaly.quality
  PRIMARY KEY (session_id, t)
);

CREATE TABLE event (
  id            INTEGER PRIMARY KEY,
  session_id    INTEGER NOT NULL REFERENCES session(id),
  t_start       REAL    NOT NULL,
  t_end         REAL,                          -- NULL = still flagged
  peak_level    REAL,
  acknowledged  INTEGER NOT NULL DEFAULT 0,
  ack_at        REAL,
  note          TEXT,
  -- what a human concluded this was. the whole point of a sensitivity-first
  -- detector is that somebody reviews every flag; recording only THAT they
  -- looked, and not what they found, throws the answer away.
  verdict       TEXT CHECK (verdict IS NULL OR
                            verdict IN ('real','artifact','unsure')),
  window_path   TEXT                           -- the waveform that fired it
);

-- the ground truth. the induction timestamp IS the label, so a protocol session
-- without these rows is unscoreable no matter how good the readings are.
CREATE TABLE protocol_mark (
  id            INTEGER PRIMARY KEY,
  session_id    INTEGER NOT NULL REFERENCES session(id),
  t             REAL    NOT NULL,
  phase         TEXT    NOT NULL CHECK (phase IN ('baseline','induction','recovery')),
  label         TEXT
);

-- small facts about the store itself: what has already been imported, and so
-- on. without this, deleting a subject and restarting would silently resurrect
-- them from the legacy scorer file they came from.
CREATE TABLE meta (
  key           TEXT    PRIMARY KEY,
  value         TEXT,
  created       REAL    NOT NULL
);

CREATE INDEX idx_session_subject ON session(subject_id, started DESC);
CREATE INDEX idx_session_device  ON session(device_id, started DESC);
CREATE INDEX idx_session_open    ON session(device_id) WHERE ended IS NULL;
CREATE INDEX idx_baseline_active ON baseline(subject_id) WHERE active = 1;
CREATE INDEX idx_event_session   ON event(session_id, t_start DESC);
CREATE INDEX idx_mark_session    ON protocol_mark(session_id, t);
"""


def sha256_of(path: str):
    """digest a model artifact so a stored score can be traced to what made it."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Db:
    """one connection, one lock. writes are ~1/s/device and a few hundred bytes,
    so there is nothing to gain from a pool and a lot to lose from two writers."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False: fleet.py writes from the event loop and reads
        # from the NET executor. the lock, not the thread, is what serialises.
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.con.execute("PRAGMA journal_mode=WAL")   # roster reads while master writes
            self.con.execute("PRAGMA foreign_keys=ON")
            self.con.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # ---------------------------------------------------------------- plumbing

    def _migrate(self):
        with self.lock:
            v = self.con.execute("PRAGMA user_version").fetchone()[0]
            if v > SCHEMA_VERSION:
                raise RuntimeError(
                    "%s is schema v%d, this code speaks v%d -- newer code wrote it; "
                    "do not downgrade in place" % (self.path, v, SCHEMA_VERSION))
            if v == 0:
                self.con.executescript(_SCHEMA)
                v = SCHEMA_VERSION
            if v == 1:
                # v2 moved the wearer onto the board, so it survives between the
                # sessions that now open and close by themselves.
                self.con.execute("ALTER TABLE device ADD COLUMN subject_id "
                                 "INTEGER REFERENCES subject(id)")
                v = 2
            if v == 2:
                # v3: a MAC-derived id identifies a board, it does not name one.
                self.con.execute("ALTER TABLE device ADD COLUMN name TEXT")
                v = 3
            if v == 3:
                self.con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, "
                                 "value TEXT, created REAL NOT NULL)")
                v = 4
            if v == 4:
                # v5: signal quality per reading, and what a human made of a flag
                self.con.execute("ALTER TABLE reading ADD COLUMN quality REAL")
                self.con.execute("ALTER TABLE event ADD COLUMN verdict TEXT")
                self.con.execute("ALTER TABLE event ADD COLUMN window_path TEXT")
                v = 5
            if v == 5:
                self.con.execute("ALTER TABLE baseline ADD COLUMN k_sigma REAL")
                v = 6
            self.con.execute("PRAGMA user_version=%d" % v)
            self.con.commit()

    def _w(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.con.execute(sql, args)
            self.con.commit()
            return cur.lastrowid

    def q(self, sql: str, args=()) -> list:
        with self.lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    def q1(self, sql: str, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    def backup(self, path: str) -> str:
        """a consistent single-file snapshot, safe to take while the master runs.

        copying pulse.db on its own is NOT a backup: in WAL mode the newest
        commits sit in pulse.db-wal until a checkpoint, so the copy silently
        comes back missing the most recent work.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        dst = sqlite3.connect(path)
        try:
            with self.lock:
                self.con.backup(dst)
        finally:
            dst.close()
        return path

    def close(self):
        with self.lock:
            self.con.close()

    # ---------------------------------------------------------------- devices

    def seen_device(self, dev_id: str, ip=None):
        now = time.time()
        self._w("""INSERT INTO device (id, first_seen, last_seen, last_ip)
                   VALUES (?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen,
                                                 last_ip=COALESCE(excluded.last_ip, last_ip)""",
                (dev_id, now, now, ip))

    def devices(self) -> list:
        return self.q("SELECT * FROM device ORDER BY id")

    def assign_subject(self, device_id: str, subject_id):
        """put a person on a board, or take them off with subject_id=None.

        the open session ends either way: a session is one subject on one board,
        so changing the wearer necessarily ends it rather than silently
        re-attributing the readings already in it.
        """
        self.close_open_sessions(device_id)
        if subject_id is not None:
            # a person wears one board. without this they stayed assigned to the
            # old one too, and two boards would open two concurrent sessions for
            # the same body.
            for d in self.q("SELECT id FROM device WHERE subject_id=? AND id<>?",
                            (subject_id, device_id)):
                self.close_open_sessions(d["id"])
                self._w("UPDATE device SET subject_id=NULL WHERE id=?", (d["id"],))
        self._w("UPDATE device SET subject_id=? WHERE id=?", (subject_id, device_id))

    def meta_get(self, key: str):
        r = self.q1("SELECT value FROM meta WHERE key=?", (key,))
        return r["value"] if r else None

    def meta_set(self, key: str, value: str = ""):
        self._w("INSERT INTO meta (key, value, created) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value, time.time()))

    def rename_device(self, device_id: str, name: str):
        """a label for humans. the id stays the identity -- everything keys on it."""
        self._w("UPDATE device SET name=? WHERE id=?", (name.strip() or None, device_id))

    def device(self, device_id: str):
        return self.q1("SELECT * FROM device WHERE id=?", (device_id,))

    def assigned_subject(self, device_id: str):
        return self.q1("""SELECT sub.* FROM device d JOIN subject sub ON sub.id = d.subject_id
                          WHERE d.id=?""", (device_id,))

    # ------------------------------------------------------------ housekeeping

    def close_stale_sessions(self) -> int:
        """a session left open by a killed master ended when the readings did,
        not when the next master happens to start. stamp it there instead."""
        rows = self.q("SELECT id, started FROM session WHERE ended IS NULL")
        for r in rows:
            last = self.q1("SELECT MAX(t) AS t FROM reading WHERE session_id=?", (r["id"],))
            self._w("UPDATE session SET ended=? WHERE id=?",
                    ((last and last["t"]) or r["started"], r["id"]))
            self._w("UPDATE event SET t_end=COALESCE(t_end, ?) WHERE session_id=? AND t_end IS NULL",
                    ((last and last["t"]) or r["started"], r["id"]))
        return len(rows)

    # ---------------------------------------------------------------- subjects

    def create_subject(self, code: str, display_name: str = "", notes: str = "") -> int:
        return self._w("INSERT INTO subject (code, display_name, created, notes) VALUES (?,?,?,?)",
                       (code, display_name or code, time.time(), notes))

    def subject_by_code(self, code: str):
        return self.q1("SELECT * FROM subject WHERE code=?", (code,))

    def rename_subject(self, subject_id: int, name=None, code=None):
        """the display name is free text; the code is the label results are
        reported under, so it has to stay unique and is checked by the caller."""
        if name is not None:
            self._w("UPDATE subject SET display_name=? WHERE id=?",
                    (name.strip() or None, subject_id))
        if code is not None and code.strip():
            self._w("UPDATE subject SET code=? WHERE id=?", (code.strip(), subject_id))

    def subject(self, subject_id: int):
        return self.q1("SELECT * FROM subject WHERE id=?", (subject_id,))

    def subjects(self) -> list:
        return self.q("SELECT * FROM subject ORDER BY code")

    def next_subject_code(self) -> str:
        """S01, S02, ... skipping whatever is taken."""
        taken = set(s["code"] for s in self.subjects())
        i = 1
        while "S%02d" % i in taken:
            i += 1
        return "S%02d" % i

    # ----------------------------------------------------------------- models

    def register_model(self, name: str, path=None, notes: str = "") -> int:
        sha = sha256_of(path) if path else None
        row = self.q1("SELECT id FROM model WHERE name=? AND sha IS ?", (name, sha))
        if row:
            return row["id"]
        return self._w("INSERT INTO model (name, sha, created, notes) VALUES (?,?,?,?)",
                       (name, sha, time.time(), notes))

    # --------------------------------------------------------------- sessions

    def open_session(self, subject_id: int, device_id: str, kind: str,
                     model_id=None, sens=None, raw_path=None, notes: str = "") -> int:
        if kind not in KINDS:
            raise ValueError("kind must be one of %s" % (KINDS,))
        self.close_open_sessions(device_id)     # a board measures one subject at a time
        return self._w("""INSERT INTO session
                            (subject_id, device_id, model_id, kind, started, sens, raw_path, notes)
                          VALUES (?,?,?,?,?,?,?,?)""",
                       (subject_id, device_id, model_id, kind, time.time(), sens, raw_path, notes))

    def record_session(self, subject_id: int, device_id: str, kind: str,
                       started: float, ended: float, model_id=None,
                       notes: str = "") -> int:
        """a session that is already over -- a calibration collection, an import.
        does not disturb whatever is open, because it never was open."""
        if kind not in KINDS:
            raise ValueError("kind must be one of %s" % (KINDS,))
        return self._w("""INSERT INTO session
                            (subject_id, device_id, model_id, kind, started, ended, notes)
                          VALUES (?,?,?,?,?,?,?)""",
                       (subject_id, device_id, model_id, kind, started, ended, notes))

    def close_session(self, session_id: int):
        self._w("UPDATE session SET ended=? WHERE id=? AND ended IS NULL",
                (time.time(), session_id))

    def close_open_sessions(self, device_id: str):
        self._w("UPDATE session SET ended=? WHERE device_id=? AND ended IS NULL",
                (time.time(), device_id))

    def current_session(self, device_id: str):
        """who is on this board right now, with their subject joined in."""
        return self.q1("""SELECT s.*, sub.code AS subject_code, sub.display_name
                          FROM session s JOIN subject sub ON sub.id = s.subject_id
                          WHERE s.device_id=? AND s.ended IS NULL
                          ORDER BY s.started DESC LIMIT 1""", (device_id,))

    def sessions_for_subject(self, subject_id: int, limit: int = 50) -> list:
        return self.q("""SELECT * FROM session WHERE subject_id=?
                         ORDER BY started DESC LIMIT ?""", (subject_id, limit))

    # -------------------------------------------------------------- baselines

    def save_baseline(self, subject_id: int, threshold: float, ref_lo: float, ref_hi: float,
                      n_windows: int = 0, fs: int = 64, win_len: int = 3840,
                      session_id=None, model_id=None, source: str = "device",
                      created=None, k_sigma=None) -> int:
        """a new baseline retires the old one; the old row stays for reproducibility."""
        with self.lock:
            self.con.execute("UPDATE baseline SET active=0 WHERE subject_id=? AND active=1",
                             (subject_id,))
            cur = self.con.execute(
                """INSERT INTO baseline (subject_id, session_id, model_id, threshold, ref_lo,
                                         ref_hi, n_windows, fs, win_len, source, created,
                                         active, k_sigma)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?)""",
                (subject_id, session_id, model_id, threshold, ref_lo, ref_hi,
                 n_windows, fs, win_len, source, created or time.time(), k_sigma))
            self.con.commit()
            return cur.lastrowid

    def active_baseline(self, subject_id: int):
        return self.q1("SELECT * FROM baseline WHERE subject_id=? AND active=1", (subject_id,))

    # --------------------------------------------------------------- readings

    def add_reading(self, session_id: int, score=None, level=None, flag=None,
                    bpm=None, spo2=None, contact=None, quality=None, t=None):
        self._w("""INSERT OR REPLACE INTO reading
                     (session_id, t, score, level, flag, bpm, spo2, contact, quality)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (session_id, t or time.time(), score, level,
                 None if flag is None else int(flag), bpm, spo2,
                 None if contact is None else int(contact), quality))

    def readings(self, session_id: int, limit: int = 5000) -> list:
        return self.q("SELECT * FROM reading WHERE session_id=? ORDER BY t LIMIT ?",
                      (session_id, limit))

    # ----------------------------------------------------------------- events

    def open_event(self, session_id: int, level=None, t=None) -> int:
        return self._w("INSERT INTO event (session_id, t_start, peak_level) VALUES (?,?,?)",
                       (session_id, t or time.time(), level))

    def bump_event(self, event_id: int, level: float):
        self._w("UPDATE event SET peak_level=MAX(COALESCE(peak_level,0),?) WHERE id=?",
                (level, event_id))

    def close_event(self, event_id: int, t=None):
        self._w("UPDATE event SET t_end=? WHERE id=? AND t_end IS NULL",
                (t or time.time(), event_id))

    def ack_event(self, event_id: int, verdict=None, note: str = ""):
        """a human looked, and this is what they concluded."""
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError("verdict must be one of %s" % (VERDICTS,))
        self._w("UPDATE event SET acknowledged=1, ack_at=?, verdict=?, note=? WHERE id=?",
                (time.time(), verdict, note, event_id))

    def set_event_window(self, event_id: int, path: str):
        self._w("UPDATE event SET window_path=? WHERE id=?", (path, event_id))

    def clear_events(self, device_id: str = None, subject_id: int = None) -> int:
        """forget the flag history for one board or one person; returns the count.

        The saved waveforms under flags/ are deliberately left on disk. Clearing
        a list in the roster should not quietly destroy the only record of what
        those flags actually looked like -- that is the evidence a verdict was
        given on, and it is what makes a bad flag diagnosable weeks later.
        """
        if (device_id is None) == (subject_id is None):
            raise ValueError("clear_events takes exactly one of device_id, subject_id")
        col, val = (("device_id", device_id) if device_id is not None
                    else ("subject_id", subject_id))
        with self.lock:
            cur = self.con.execute(
                "DELETE FROM event WHERE session_id IN "
                "(SELECT id FROM session WHERE %s=?)" % col, (val,))
            self.con.commit()
            return cur.rowcount

    def verdict_tally(self, subject_id=None) -> dict:
        """how the reviewed flags came out -- the number that says whether this
        detector is finding deviations or finding knocks on the sensor."""
        where, args = "", ()
        if subject_id is not None:
            where, args = "WHERE s.subject_id=?", (subject_id,)
        rows = self.q("""SELECT e.verdict, COUNT(*) c FROM event e
                         JOIN session s ON s.id = e.session_id %s
                         GROUP BY e.verdict""" % where, args)
        out = {v: 0 for v in VERDICTS}
        out["unreviewed"] = 0
        for r in rows:
            out[r["verdict"] or "unreviewed"] = r["c"]
        return out

    def events_for_subject(self, subject_id: int, limit: int = 50) -> list:
        return self.q("""SELECT e.*, s.device_id, s.kind FROM event e
                         JOIN session s ON s.id = e.session_id
                         WHERE s.subject_id=? ORDER BY e.t_start DESC LIMIT ?""",
                      (subject_id, limit))

    def events_for_device(self, device_id: str, limit: int = 50) -> list:
        return self.q("""SELECT e.*, s.subject_id, sub.code AS subject_code
                         FROM event e JOIN session s ON s.id = e.session_id
                         JOIN subject sub ON sub.id = s.subject_id
                         WHERE s.device_id=? ORDER BY e.t_start DESC LIMIT ?""",
                      (device_id, limit))

    # ----------------------------------------------------------- roster views

    def subject_stats(self, subject_id: int) -> dict:
        """what deleting this person would destroy, so the confirmation can say so."""
        one = lambda sql: self.q1(sql, (subject_id,))["c"]
        return {
            "n_sessions": one("SELECT COUNT(*) c FROM session WHERE subject_id=?"),
            "n_baselines": one("SELECT COUNT(*) c FROM baseline WHERE subject_id=?"),
            "n_readings": one("SELECT COUNT(*) c FROM reading WHERE session_id IN "
                              "(SELECT id FROM session WHERE subject_id=?)"),
            "n_events": one("SELECT COUNT(*) c FROM event WHERE session_id IN "
                            "(SELECT id FROM session WHERE subject_id=?)"),
            "devices": [r["id"] for r in
                        self.q("SELECT id FROM device WHERE subject_id=?", (subject_id,))],
        }

    def delete_subject(self, subject_id: int):
        """the person and everything measured about them, in one transaction.

        readings, events and marks hang off sessions rather than off the subject,
        so they have to go first -- the foreign keys would refuse otherwise, and
        leaving them would be worse: rows about a person who no longer exists.
        """
        sub = "(SELECT id FROM session WHERE subject_id=?)"
        with self.lock:
            c = self.con
            c.execute("UPDATE device SET subject_id=NULL WHERE subject_id=?", (subject_id,))
            c.execute("DELETE FROM reading WHERE session_id IN " + sub, (subject_id,))
            c.execute("DELETE FROM event WHERE session_id IN " + sub, (subject_id,))
            c.execute("DELETE FROM protocol_mark WHERE session_id IN " + sub, (subject_id,))
            c.execute("DELETE FROM baseline WHERE subject_id=?", (subject_id,))
            c.execute("DELETE FROM session WHERE subject_id=?", (subject_id,))
            c.execute("DELETE FROM subject WHERE id=?", (subject_id,))
            c.commit()

    def subjects_status(self) -> list:
        """one row per subject with everything the roster shows, in one query
        rather than N+1 round trips as the subject list grows."""
        return self.q("""
            SELECT sub.id, sub.code, sub.display_name, sub.created, sub.notes,
                   b.threshold, b.n_windows, b.created AS baseline_created,
                   (SELECT d.id FROM device d WHERE d.subject_id = sub.id LIMIT 1) AS device_id,
                   (SELECT COUNT(*) FROM session s WHERE s.subject_id = sub.id) AS n_sessions,
                   (SELECT MAX(s.started) FROM session s WHERE s.subject_id = sub.id) AS last_session,
                   (SELECT COUNT(*) FROM event e JOIN session s ON s.id = e.session_id
                     WHERE s.subject_id = sub.id) AS n_events,
                   (SELECT COUNT(*) FROM reading r JOIN session s ON s.id = r.session_id
                     WHERE s.subject_id = sub.id) AS n_readings,
                   (SELECT COUNT(*) FROM event e JOIN session s ON s.id = e.session_id
                     WHERE s.subject_id = sub.id AND e.acknowledged = 0) AS n_unacked
            FROM subject sub
            LEFT JOIN baseline b ON b.subject_id = sub.id AND b.active = 1
            ORDER BY sub.code""")

    # ------------------------------------------------------------ ground truth

    def add_mark(self, session_id: int, phase: str, label: str = "", t=None) -> int:
        if phase not in PHASES:
            raise ValueError("phase must be one of %s" % (PHASES,))
        return self._w("INSERT INTO protocol_mark (session_id, t, phase, label) VALUES (?,?,?,?)",
                       (session_id, t or time.time(), phase, label))

    def marks(self, session_id: int) -> list:
        return self.q("SELECT * FROM protocol_mark WHERE session_id=? ORDER BY t", (session_id,))


# ------------------------------------------------------- migrating what exists

def import_legacy_scorers(db: Db, saved_dir=None, verbose: bool = True) -> list:
    """fold the per-BOARD scorer_*.npz files into per-SUBJECT baselines.

    each file becomes a device, a subject nobody has named yet, a zero-length
    calibration session stamped with the file's own `created`, and one baseline.
    scorer.npz is deliberately skipped: it is the model's shipped WESAD-wrist
    default, not anybody's baseline.

    idempotent -- a device that already has an imported baseline is left alone,
    so this is safe to run on every start.
    """
    import glob
    import numpy as np

    saved_dir = saved_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
    done = []
    for path in sorted(glob.glob(os.path.join(saved_dir, "scorer_*.npz"))):
        name = os.path.basename(path)
        z = np.load(path, allow_pickle=False)
        keys = set(z.files)
        if not {"threshold", "ref_lo", "ref_hi"} <= keys:
            continue

        # scorer_device.npz predates device ids -- it came off the USB rig, which
        # never reported one. give it a stable stand-in rather than dropping it:
        # it holds the calm session the transfer-delta number was measured on.
        dev_id = str(z["device_id"].item()) if "device_id" in keys else "usb-rig"

        # keyed on the FILE, not on whether a baseline exists: keying on the
        # baseline meant deleting the subject let the next start import them
        # right back again.
        if db.meta_get("imported:" + name):
            if verbose:
                print("  %-28s already imported, skipped" % name)
            continue

        # a store written before the meta table: adopt whatever it already
        # imported rather than importing it a second time under a new code.
        legacy = db.q1("""SELECT b.id FROM baseline b JOIN session s ON s.id = b.session_id
                          WHERE s.device_id=? AND b.source='import' LIMIT 1""", (dev_id,))
        if legacy:
            db.meta_set("imported:" + name, "adopted")
            if verbose:
                print("  %-28s already imported, skipped" % name)
            continue

        created = time.time()
        if "created" in keys:
            try:
                import datetime as dt
                created = dt.datetime.fromisoformat(str(z["created"].item())).timestamp()
            except Exception:
                pass

        db.seen_device(dev_id)
        code = db.next_subject_code()
        sub = db.create_subject(code, "unnamed (%s)" % dev_id,
                                notes="imported from %s -- rename to the real subject" % name)
        sess = db.record_session(sub, dev_id, "calibration", created, created,
                                 notes="imported from %s" % name)
        db.save_baseline(
            sub, float(z["threshold"]), float(z["ref_lo"]), float(z["ref_hi"]),
            n_windows=int(z["n_windows"].item()) if "n_windows" in keys else 0,
            fs=int(z["fs"].item()) if "fs" in keys else 64,
            win_len=int(z["win_len"].item()) if "win_len" in keys else 3840,
            session_id=sess, source="import", created=created)
        db.meta_set("imported:" + name, code)
        # put them on the board they were calibrated on, unless somebody is
        # already there. the old scorer file WAS an assignment, implicitly -- not
        # carrying it over would silently stop a working board from scoring.
        if not db.assigned_subject(dev_id):
            db.assign_subject(dev_id, sub)
        done.append((name, code, dev_id))
        if verbose:
            print("  %-28s -> subject %s on device %s" % (name, code, dev_id))
    return done


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="inspect or initialise the master's store")
    ap.add_argument("--db", default=DEFAULT_PATH)
    ap.add_argument("--import-legacy", action="store_true",
                    help="fold anomaly/saved/scorer_*.npz into per-subject baselines")
    ap.add_argument("--backup", metavar="FILE",
                    help="write a consistent snapshot (safe while the master runs)")
    a = ap.parse_args(argv)

    db = Db(a.db)
    print("  %s (schema v%d)\n" % (db.path, SCHEMA_VERSION))
    if a.import_legacy:
        n = import_legacy_scorers(db)
        print("\n  imported %d\n" % len(n))

    if a.backup:
        print("  snapshot -> %s" % db.backup(a.backup))

    for t in ("subject", "device", "session", "baseline", "reading", "event", "protocol_mark"):
        print("  %-14s %d" % (t, db.q1("SELECT COUNT(*) c FROM %s" % t)["c"]))
    subs = db.subjects()
    if subs:
        print("\n  subjects:")
        for s in subs:
            b = db.active_baseline(s["id"])
            print("    %-5s %-26s %s" % (
                s["code"], s["display_name"] or "",
                "baseline thr %.5f (%d windows)" % (b["threshold"], b["n_windows"] or 0)
                if b else "no baseline"))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
