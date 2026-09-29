// The on-device stress detector: movement gate -> 10 features -> Mahalanobis
// -> the live self-referencing scale. Plain C++, no Arduino, no allocation:
// the sketch hands it one arena (PSRAM) and a 1 Hz tick.
//
// Every routine here is a line-for-line translation of the Python mirror in
// anomaly/board_export.py, which `python -m anomaly.board_export --check`
// verifies against scipy, numpy and anomaly/quality.py on real WESAD windows.
// bd_selftest() then checks THIS translation against numbers the Python
// reference computed, on the board, at boot. The numbers live in
// board_model.h, which that script generates; edit neither by hand without
// re-running both checks.
//
// What it costs against the host: PR-AUC 0.635 vs 0.706 on WESAD LOSO (the
// baseline row in anomaly/RESULTS.md). What it buys: a verdict with no master,
// and a raw waveform that never has to leave the board.
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "board_model.h"

struct bd_cpx {
  float r, i;
};

// ------------------------------------------------------------------ arena
struct BdWork {
  float *hist;      // BD_HIST   the snapshot the gate reads
  float *win;       // BD_WIN    the window it assembles
  double *ext;      // BD_WIN + 2*BD_PADLEN  zero-phase filter buffer
  double *xd;       // BD_WIN    the z-scored window
  double *dd;       // BD_WIN    sample-to-sample steps
  double *dbuf;     // BD_WIN    scratch for medians / percentiles
  uint8_t *mask;    // BD_WIN    the spike mask
  int *ia;          // BD_WIN/2  peaks, or run starts
  int *ib;          // BD_WIN/2  peak order, or run ends
  uint8_t *keep;    // BD_WIN/2
  bd_cpx *fin;      // BD_WIN
  bd_cpx *fout;     // BD_WIN
  bd_cpx *tw;       // BD_WIN    FFT twiddles
  double *amp;      // BD_HIST/BD_FS  per-second amplitude
  int *picks;       // BD_HIST/BD_FS
  int fac[32];      // FFT factorisation: (p, m) pairs
};

#define BD_NSEC (BD_HIST / BD_FS)

static inline size_t bd__al(size_t n) { return (n + 7) & ~(size_t)7; }

static inline size_t bd_arena_bytes() {
  const size_t W = BD_WIN, H = BD_HIST;
  return bd__al(H * sizeof(float)) + bd__al(W * sizeof(float)) +
         bd__al((W + 2 * BD_PADLEN) * sizeof(double)) + 3 * bd__al(W * sizeof(double)) +
         bd__al(W) + 2 * bd__al((W / 2) * sizeof(int)) + bd__al(W / 2) +
         3 * bd__al(W * sizeof(bd_cpx)) + bd__al(BD_NSEC * sizeof(double)) +
         bd__al(BD_NSEC * sizeof(int));
}

// kiss_fft's factorisation: 4s first, then 2, then odd primes
static inline void bd__factor(int n, int *fac) {
  int p = 4, k = 0;
  while (n > 1) {
    while (n % p) p = (p == 4) ? 2 : (p == 2 ? 3 : p + 2);
    n /= p;
    fac[k++] = p;
    fac[k++] = n;
  }
}

static inline bool bd_init(BdWork *w, void *arena, size_t bytes) {
  if (arena == NULL || bytes < bd_arena_bytes()) return false;
  uint8_t *a = (uint8_t *)arena;
  const size_t W = BD_WIN, H = BD_HIST;
#define BD__TAKE(ptr, T, n) \
  do { ptr = (T *)a; a += bd__al((n) * sizeof(T)); } while (0)
  BD__TAKE(w->hist, float, H);
  BD__TAKE(w->win, float, W);
  BD__TAKE(w->ext, double, W + 2 * BD_PADLEN);
  BD__TAKE(w->xd, double, W);
  BD__TAKE(w->dd, double, W);
  BD__TAKE(w->dbuf, double, W);
  BD__TAKE(w->mask, uint8_t, W);
  BD__TAKE(w->ia, int, W / 2);
  BD__TAKE(w->ib, int, W / 2);
  BD__TAKE(w->keep, uint8_t, W / 2);
  BD__TAKE(w->fin, bd_cpx, W);
  BD__TAKE(w->fout, bd_cpx, W);
  BD__TAKE(w->tw, bd_cpx, W);
  BD__TAKE(w->amp, double, BD_NSEC);
  BD__TAKE(w->picks, int, BD_NSEC);
#undef BD__TAKE
  for (int k = 0; k < BD_WIN; k++) {          // computed in double, stored as float
    double ph = -2.0 * 3.14159265358979323846 * (double)k / (double)BD_WIN;
    w->tw[k].r = (float)cos(ph);
    w->tw[k].i = (float)sin(ph);
  }
  memset(w->fac, 0, sizeof(w->fac));
  bd__factor(BD_WIN, w->fac);
  return true;
}

// ---------------------------------------------------------------- helpers
static int bd__cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

// numpy's default (linear) percentile. SORTS buf in place.
static inline double bd__pct(double *buf, int n, double q) {
  qsort(buf, (size_t)n, sizeof(double), bd__cmpd);
  double pos = (double)(n - 1) * q / 100.0;
  int lo = (int)floor(pos);
  int hi = (lo + 1 < n) ? lo + 1 : n - 1;
  return buf[lo] + (pos - (double)lo) * (buf[hi] - buf[lo]);
}

static inline double bd__std(const float *x, int n) {
  double m = 0.0, s = 0.0;
  for (int i = 0; i < n; i++) m += x[i];
  m /= n;
  for (int i = 0; i < n; i++) s += ((double)x[i] - m) * ((double)x[i] - m);
  return sqrt(s / n);
}

// --------------------------------------------------------- the zero-phase filter
// scipy.signal.sosfiltfilt with its defaults: odd extension by BD_PADLEN, and
// the sosfilt_zi steady state scaled by the first sample of each pass.
static inline void bd__sos_pass(double *y, int n, double x0, int dir) {
  double z[2][2];
  for (int s = 0; s < 2; s++) {
    z[s][0] = BD_ZI[s][0] * x0;
    z[s][1] = BD_ZI[s][1] * x0;
  }
  for (int t = 0; t < n; t++) {
    int k = dir > 0 ? t : n - 1 - t;
    double v = y[k];
    for (int s = 0; s < 2; s++) {
      const double *c = BD_SOS[s];
      double o = c[0] * v + z[s][0];
      z[s][0] = c[1] * v - c[4] * o + z[s][1];
      z[s][1] = c[2] * v - c[5] * o;
      v = o;
    }
    y[k] = v;
  }
}

// x (BD_WIN) -> w->ext; the band-passed signal is w->ext + BD_PADLEN
static inline void bd__filtfilt(BdWork *w, const double *x) {
  const int n = BD_WIN, p = BD_PADLEN, N = n + 2 * p;
  double *e = w->ext;
  for (int i = 0; i < p; i++) e[i] = 2.0 * x[0] - x[p - i];
  for (int i = 0; i < n; i++) e[p + i] = x[i];
  for (int j = 0; j < p; j++) e[p + n + j] = 2.0 * x[n - 1] - x[n - 2 - j];
  bd__sos_pass(e, N, e[0], +1);
  bd__sos_pass(e, N, e[N - 1], -1);
}

// ------------------------------------------------------------ find_peaks
// scipy.signal.find_peaks(x, distance=, prominence=): local maxima (plateaus to
// their midpoint), then distance by descending height, then prominence.
// Returns the count; the peaks land in w->ia.
static inline int bd__find_peaks(BdWork *w, const double *x, int n, int distance,
                                 bool use_prom, double prom) {
  int *pk = w->ia, *ord = w->ib;
  uint8_t *keep = w->keep;
  int m = 0, i = 1, imax = n - 1;
  while (i < imax) {
    if (x[i - 1] < x[i]) {
      int ahead = i + 1;
      while (ahead < imax && x[ahead] == x[i]) ahead++;
      if (x[ahead] < x[i]) {
        pk[m++] = (i + ahead - 1) / 2;
        i = ahead;
      }
    }
    i++;
  }
  // ascending by height, ties by position; insertion sort -- m is ~100
  for (int j = 0; j < m; j++) {
    int v = j, q = j - 1;
    while (q >= 0 && (x[pk[ord[q]]] > x[pk[v]] ||
                      (x[pk[ord[q]]] == x[pk[v]] && ord[q] > v))) {
      ord[q + 1] = ord[q];
      q--;
    }
    ord[q + 1] = v;
    keep[j] = 1;
  }
  for (int r = m - 1; r >= 0; r--) {
    int j = ord[r];
    if (!keep[j]) continue;
    for (int k = j - 1; k >= 0 && pk[j] - pk[k] < distance; k--) keep[k] = 0;
    for (int k = j + 1; k < m && pk[k] - pk[j] < distance; k++) keep[k] = 0;
  }
  int out = 0;
  for (int j = 0; j < m; j++) {
    if (!keep[j]) continue;
    int p = pk[j];
    if (use_prom) {
      double h = x[p], lmin = h, rmin = h;
      for (int k = p; k >= 0 && x[k] <= h; k--)
        if (x[k] < lmin) lmin = x[k];
      for (int k = p; k <= n - 1 && x[k] <= h; k++)
        if (x[k] < rmin) rmin = x[k];
      if (h - (lmin > rmin ? lmin : rmin) < prom) continue;
    }
    pk[out++] = p;     // out <= j, so this never overwrites an unread peak
  }
  return out;
}

// ------------------------------------------------------------------ FFT
// kiss-style recursive mixed-radix DIT with a generic butterfly
static void bd__fft_work(const BdWork *w, bd_cpx *fo, const bd_cpx *fi, int fstride,
                         const int *fac) {
  const int p = fac[0], m = fac[1];
  if (m == 1) {
    for (int q = 0; q < p; q++) fo[q] = fi[q * fstride];
  } else {
    for (int q = 0; q < p; q++) bd__fft_work(w, fo + q * m, fi + q * fstride, fstride * p, fac + 2);
  }
  bd_cpx scratch[8];                  // p <= 5 for BD_WIN = 3840
  for (int u = 0; u < m; u++) {
    int k = u;
    for (int q1 = 0; q1 < p; q1++) {
      scratch[q1] = fo[k];
      k += m;
    }
    k = u;
    for (int q1 = 0; q1 < p; q1++) {
      bd_cpx acc = scratch[0];
      int tidx = 0;
      for (int q = 1; q < p; q++) {
        tidx += fstride * k;
        if (tidx >= BD_WIN) tidx -= BD_WIN;
        const bd_cpx t = w->tw[tidx];
        acc.r += scratch[q].r * t.r - scratch[q].i * t.i;
        acc.i += scratch[q].r * t.i + scratch[q].i * t.r;
      }
      fo[k] = acc;
      k += m;
    }
  }
}

// -------------------------------------------------------------- features
// anomaly/features.extract_features, in its order:
// hr_bpm, ibi_sdnn, ibi_rmssd, n_peaks_per_s, bp_std, bp_ptp, dom_freq,
// pulse_band_ratio, spec_entropy, mean_abs_diff. NAN where Python has NaN.
static inline void bd_features(BdWork *w, const float *x, float f[BD_NF]) {
  const int n = BD_WIN;
  double mean = 0.0, var = 0.0;
  for (int i = 0; i < n; i++) mean += x[i];
  mean /= n;
  for (int i = 0; i < n; i++) var += ((double)x[i] - mean) * ((double)x[i] - mean);
  double sd = sqrt(var / n);
  double *xd = w->xd;
  for (int i = 0; i < n; i++) xd[i] = ((double)x[i] - mean) / (sd + 1e-8);
  for (int i = 0; i < BD_NF; i++) f[i] = NAN;

  bd__filtfilt(w, xd);
  const double *bp = w->ext + BD_PADLEN;
  double bm = 0.0, bv = 0.0, bmax = bp[0], bmin = bp[0];
  for (int i = 0; i < n; i++) {
    bm += bp[i];
    if (bp[i] > bmax) bmax = bp[i];
    if (bp[i] < bmin) bmin = bp[i];
  }
  bm /= n;
  for (int i = 0; i < n; i++) bv += (bp[i] - bm) * (bp[i] - bm);
  double bsd = sqrt(bv / n);
  f[4] = (float)bsd;
  f[5] = (float)(bmax - bmin);
  double mad = 0.0;
  for (int i = 1; i < n; i++) mad += fabs(xd[i] - xd[i - 1]);
  f[9] = (float)(mad / (n - 1));

  int np = bd__find_peaks(w, bp, n, BD_PEAK_DIST, bsd > 0, bsd * 0.3);
  if (np >= 3) {
    int k = np - 1;
    double *ibi = w->dbuf, *srt = w->dd;
    double im = 0.0;
    for (int j = 0; j < k; j++) {
      ibi[j] = (double)(w->ia[j + 1] - w->ia[j]) / BD_FS;
      srt[j] = ibi[j];
      im += ibi[j];
    }
    qsort(srt, (size_t)k, sizeof(double), bd__cmpd);
    double med = (k % 2) ? srt[k / 2] : 0.5 * (srt[k / 2 - 1] + srt[k / 2]);
    f[0] = (float)(60.0 / med);
    im /= k;
    double s1 = 0.0, s2 = 0.0;
    for (int j = 0; j < k; j++) s1 += (ibi[j] - im) * (ibi[j] - im);
    for (int j = 0; j + 1 < k; j++) s2 += (ibi[j + 1] - ibi[j]) * (ibi[j + 1] - ibi[j]);
    f[1] = (float)sqrt(s1 / k);
    f[2] = (float)sqrt(s2 / (k - 1));
  }
  f[3] = (float)(np / ((double)n / BD_FS));

  for (int i = 0; i < n; i++) {
    w->fin[i].r = (float)xd[i];
    w->fin[i].i = 0.0f;
  }
  bd__fft_work(w, w->fout, w->fin, 1, w->fac);
  double tot = 0.0, bs = 0.0, best = -1.0;
  int arg = 0;
  double *psd = w->dbuf;
  for (int k = 0; k < BD_NBINS; k++) {
    double r = w->fout[k].r, im = w->fout[k].i;
    psd[k] = r * r + im * im;
    tot += psd[k];
    if (k >= BD_BAND_LO && k <= BD_BAND_HI) {
      bs += psd[k];
      if (psd[k] > best) {                 // first maximum, as argmax
        best = psd[k];
        arg = k;
      }
    }
  }
  tot += 1e-12;
  if (bs > 0) f[6] = (float)((double)arg * BD_FS / (double)n);
  f[7] = (float)(bs / tot);
  double ent = 0.0;
  for (int k = 0; k < BD_NBINS; k++) {
    double p = psd[k] / tot;
    ent += p * log(p + 1e-12);
  }
  f[8] = (float)(-ent / log((double)BD_NBINS));
}

// squared Mahalanobis from the WESAD calm Gaussian, then its root: the distance
// reads more linearly for the live scale, and every metric that matters is rank-based
static inline float bd_score(const float f[BD_NF]) {
  float z[BD_NF];
  for (int i = 0; i < BD_NF; i++) {
    float v = isnan(f[i]) ? BD_MEDIAN[i] : f[i];
    z[i] = (v - BD_MU[i]) / BD_SD[i] - BD_CENTER[i];
  }
  double m = 0.0;
  for (int i = 0; i < BD_NF; i++)
    for (int j = 0; j < BD_NF; j++) m += (double)z[i] * BD_INVCOV[i][j] * (double)z[j];
  return (float)sqrt(m > 0.0 ? m : 0.0);
}

// ------------------------------------------------------------ movement gate
// anomaly/quality.clean_window: drop the loud seconds, back-fill from older
// clean history, then interpolate across sample-level knocks.
enum BdGate { BD_GATE_OK = 0, BD_GATE_FILLING, BD_GATE_FLAT, BD_GATE_MOVEMENT };

struct BdClean {
  int status;          // BdGate
  int dropped;         // seconds excised
  int repaired;        // samples interpolated
  int usable;          // seconds usable, when refused for movement
  float quality;       // need / span
  float env;           // loudest second / reference
};

static inline void bd__artifact_mask(BdWork *w, float *win, int n, int *repaired) {
  double *d = w->dd, *b = w->dbuf;
  uint8_t *mask = w->mask;
  d[0] = 0.0;
  for (int i = 1; i < n; i++) d[i] = (double)win[i] - (double)win[i - 1];
  memcpy(b, d, sizeof(double) * n);
  double med = bd__pct(b, n, 50.0);
  for (int i = 0; i < n; i++) b[i] = fabs(d[i] - med);
  double mad = bd__pct(b, n, 50.0) * 1.4826;
  memset(mask, 0, n);
  *repaired = 0;
  if (mad < BD_MIN_MAD) return;
  bool any = false;
  for (int i = 0; i < n; i++) {
    if (fabs(d[i] - med) > (double)BD_SPIKE_Z * mad) {
      int lo = i - BD_SPREAD < 0 ? 0 : i - BD_SPREAD;
      int hi = i + BD_SPREAD + 1 > n ? n : i + BD_SPREAD + 1;
      for (int k = lo; k < hi; k++) mask[k] = 1;
      any = true;
    }
  }
  if (!any) return;

  // bridge the displaced plateau between two transients
  int nr = 0;
  for (int i = 0; i < n;) {
    if (!mask[i]) { i++; continue; }
    int j = i;
    while (j < n && mask[j]) j++;
    w->ia[nr] = i;
    w->ib[nr] = j;
    nr++;
    i = j;
  }
  if (nr >= 2) {
    int g = 0;
    for (int i = 0; i < n; i++)
      if (!mask[i]) b[g++] = win[i];
    if (g >= 2) {
      double gmed = bd__pct(b, g, 50.0);
      g = 0;
      for (int i = 0; i < n; i++)
        if (!mask[i]) b[g++] = fabs((double)win[i] - gmed);
      double gmad = bd__pct(b, g, 50.0) * 1.4826;
      if (gmad >= BD_MIN_MAD) {
        for (int r = 0; r + 1 < nr; r++) {
          int end = w->ib[r], start = w->ia[r + 1];
          if (start - end <= 0 || start - end > BD_BRIDGE_GAP) continue;
          int s = 0;
          for (int k = end; k < start; k++) b[s++] = win[k];
          if (fabs(bd__pct(b, s, 50.0) - gmed) > (double)BD_BRIDGE_Z * gmad)
            for (int k = end; k < start; k++) mask[k] = 1;
        }
      }
    }
  }

  int bad = 0, good = 0;
  for (int i = 0; i < n; i++) bad += mask[i];
  good = n - bad;
  if (bad == 0 || (double)bad / n > (double)BD_MAX_REPAIR || good < 2) return;
  for (int i = 0; i < n;) {
    if (!mask[i]) { i++; continue; }
    int j = i;
    while (j < n && mask[j]) j++;
    int a = i - 1, c = j;
    for (int k = i; k < j; k++) {
      if (a < 0) win[k] = win[c];
      else if (c >= n) win[k] = win[a];
      else win[k] = (float)((double)win[a] + (double)(k - a) *
                            ((double)win[c] - (double)win[a]) / (double)(c - a));
    }
    i = j;
  }
  *repaired = bad;
}

// hist = w->hist[0..nhist); on BD_GATE_OK the window is w->win
static inline int bd_clean(BdWork *w, int nhist, BdClean *c) {
  const int n = BD_FS, need = BD_WIN / BD_FS;
  const int k = nhist / n;
  memset(c, 0, sizeof(*c));
  if (k < need) return c->status = BD_GATE_FILLING;
  for (int i = 0; i < k; i++) w->amp[i] = bd__std(w->hist + i * n, n);
  memcpy(w->dbuf, w->amp, sizeof(double) * k);
  double ref = bd__pct(w->dbuf, k, BD_ENVELOPE_REF_Q);
  if (ref < BD_MIN_MAD) return c->status = BD_GATE_FLAT;
  double env = 0.0;
  for (int i = 0; i < k; i++)
    if (w->amp[i] / ref > env) env = w->amp[i] / ref;
  c->env = (float)env;
  int np = 0;
  for (int i = k - 1; i >= 0 && np < need; i--)
    if (w->amp[i] / ref <= (double)BD_ENVELOPE_RATIO) w->picks[np++] = i;
  if (np < need) {
    c->usable = np;
    return c->status = BD_GATE_MOVEMENT;
  }
  for (int j = 0; j < need; j++) {                 // picks are newest-first
    int s = w->picks[need - 1 - j];
    memcpy(w->win + j * n, w->hist + s * n, sizeof(float) * n);
  }
  int span = w->picks[0] - w->picks[need - 1] + 1;
  bd__artifact_mask(w, w->win, BD_WIN, &c->repaired);
  c->dropped = span - need;
  c->quality = (float)need / (float)span;
  return c->status = BD_GATE_OK;
}

// ------------------------------------------------------------ live scale
// anomaly/fleet.Device.apply: EMA, the wearer's own recent calm as the
// reference, the slider as the line, hysteresis on the flag.
struct BdScale {
  float recent[BD_NORM_N];
  int nrec, head;
  float ema;
  bool has_ema;
  bool has_level;
  float level, centre, spread, z;
  int above, below;
  bool flag;
};

static inline void bd_scale_reset(BdScale *s) { memset(s, 0, sizeof(*s)); }

// the finger came off for good: the window is gone, the reference is not
static inline void bd_scale_lost(BdScale *s) {
  s->has_ema = false;
  s->has_level = false;
  s->flag = false;
}

static inline float bd_thr_level(float sens) {
  if (sens < 0.0f) sens = 0.0f;
  if (sens > 1.0f) sens = 1.0f;
  return BD_THR_MIN + (BD_THR_MAX - BD_THR_MIN) * sens;
}

static inline void bd_scale_apply(BdScale *s, float raw, float sens, double *scratch) {
  s->recent[s->head] = raw;
  s->head = (s->head + 1) % BD_NORM_N;
  if (s->nrec < BD_NORM_N) s->nrec++;
  s->ema = s->has_ema ? (1.0f - BD_EMA_ALPHA) * s->ema + BD_EMA_ALPHA * raw : raw;
  s->has_ema = true;
  if (s->nrec < BD_NORM_MIN) {
    s->has_level = false;
    s->flag = false;
    return;
  }
  const int n = s->nrec;
  for (int i = 0; i < n; i++) scratch[i] = s->recent[i];
  double centre = bd__pct(scratch, n, BD_NORM_Q);
  double med = bd__pct(scratch, n, 50.0);          // still sorted, same values
  for (int i = 0; i < n; i++) scratch[i] = fabs((double)s->recent[i] - med);
  double mad = bd__pct(scratch, n, 50.0) * 1.4826;
  double sp = mad;
  if (BD_SPREAD_FLOOR * fabs(centre) > sp) sp = BD_SPREAD_FLOOR * fabs(centre);
  if (sp < 1e-6) sp = 1e-6;
  s->centre = (float)centre;
  s->spread = (float)sp;
  s->z = (float)(((double)s->ema - centre) / sp);
  float k = bd_thr_level(sens) * BD_Z_FULL;
  float lv = s->z / BD_Z_FULL;
  s->level = lv < 0.0f ? 0.0f : (lv > 1.0f ? 1.0f : lv);
  s->has_level = true;
  if (s->z >= k) {
    s->above++;
    s->below = 0;
  } else {
    s->below++;
    s->above = 0;
  }
  if (!s->flag && s->above >= BD_FLAG_ON_S) s->flag = true;
  else if (s->flag && s->below >= BD_FLAG_OFF_S) s->flag = false;
}

// -------------------------------------------------------------- self-test
// the C++ against numbers the Python reference computed; msg says what failed
static inline bool bd__close(float got, float want, float tol) {
  if (isnan(want) || isnan(got)) return isnan(want) && isnan(got);
  float scale = fabsf(want) > 1.0f ? fabsf(want) : 1.0f;
  return fabsf(got - want) <= tol * scale;
}

static inline bool bd_selftest(BdWork *w, char *msg, size_t len) {
  float f[BD_NF];
  for (int t = 0; t < BD_NTEST; t++) {
    bd_features(w, BD_TEST_X[t], f);
    for (int i = 0; i < BD_NF; i++) {
      if (!bd__close(f[i], BD_TEST_F[t][i], 2e-3f)) {
        snprintf(msg, len, "fail t%d f%d got %.6g want %.6g", t, i, (double)f[i],
                 (double)BD_TEST_F[t][i]);
        return false;
      }
    }
    float sc = bd_score(f);
    if (!bd__close(sc, BD_TEST_SCORE[t], 2e-3f)) {
      snprintf(msg, len, "fail t%d score got %.6g want %.6g", t, (double)sc,
               (double)BD_TEST_SCORE[t]);
      return false;
    }
  }
  const int nh = 2 * BD_WIN;
  for (int i = 0; i < nh; i++) w->hist[i] = BD_TEST_X[0][i % BD_WIN];
  for (int i = 70 * BD_FS; i < 75 * BD_FS; i++) w->hist[i] *= 6.0f;
  w->hist[BD_GATE_KNOCK_AT] += BD_GATE_KNOCK;
  BdClean c;
  if (bd_clean(w, nh, &c) != BD_GATE_OK || c.dropped != BD_GATE_DROPPED || c.repaired == 0) {
    snprintf(msg, len, "fail gate status %d dropped %d repaired %d", c.status, c.dropped,
             c.repaired);
    return false;
  }
  double s1 = 0.0, s2 = 0.0;
  for (int i = 0; i < BD_WIN; i++) {
    s1 += w->win[i];
    s2 += (double)w->win[i] * w->win[i];
  }
  if (fabs(s1 - BD_GATE_SUM) > 1e-5 * fabs(BD_GATE_SUM) + 1e-3 ||
      fabs(s2 - BD_GATE_SUMSQ) > 1e-5 * fabs(BD_GATE_SUMSQ)) {
    snprintf(msg, len, "fail gate sum %.6g/%.6g want %.6g/%.6g", s1, s2, BD_GATE_SUM,
             BD_GATE_SUMSQ);
    return false;
  }
  snprintf(msg, len, "pass %s", BD_MODEL_TAG);
  return true;
}
