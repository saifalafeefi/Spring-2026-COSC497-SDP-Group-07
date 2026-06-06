"""subject-wise cross-validation — the only honest split for this dataset.

subject leakage is the named killer: a person must never sit in both train and
test. these helpers split by SUBJECT, never by window.
"""
from __future__ import annotations

from .wesad import SUBJECTS


def leave_one_subject_out(subjects=None):
    """yield (train_subjects, test_subject) for each subject in turn."""
    subjects = list(subjects or SUBJECTS)
    for s in subjects:
        train = [x for x in subjects if x != s]
        yield train, s


def kfold_subjects(subjects=None, k: int = 5, seed: int = 0):
    """yield (train_subjects, test_subjects) for k subject-disjoint folds."""
    import random
    subjects = list(subjects or SUBJECTS)
    random.Random(seed).shuffle(subjects)
    folds = [subjects[i::k] for i in range(k)]
    for i in range(k):
        test = folds[i]
        train = [s for j, f in enumerate(folds) if j != i for s in f]
        yield train, test
