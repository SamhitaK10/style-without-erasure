"""Shared library for the Style Without Erasure experiments.

CURRENT STATE: only `cues` is populated. The 40 cue phrasings have been
extracted here verbatim, but the experiment scripts still carry their own
inline copies (six in total), and the placebo builder still exists in three
copies inside the experiment scripts. Finishing the extraction --
`placebo`, `divergence`, `scenarios`, `stats` -- is item 7 in
docs/REPRODUCIBILITY.md.

Until then, treat this package as documentation of intent rather than the
single source of truth, and do not edit `cues.py` without editing the
inline copies in `experiments/`.
"""

from .cues import CUES

__all__ = ["CUES"]
