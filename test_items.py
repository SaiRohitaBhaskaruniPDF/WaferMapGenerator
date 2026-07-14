"""
Test items — per-die test results (2026.07.10 spec, "Test items" section).

A wafer-sort program runs a list of TESTS on every die. The spec wants:

  * Test count in orders of magnitude: 100, 1000, ... up to 1M (never 307).
  * A Pass/Fail vs Parametric split, default 50/50, selectable in 10% steps.
      - Pass/Fail tests report 0 (fail) or 1 (pass).
      - Parametric tests report a real value from an RNG (0.0..1.0 default).
  * Data-shape options for parametric values:
      exponential (10^rng), quantized (0.2 steps), signed (-1..+1),
      scientific notation (X.XXe±YY), constant (one repeated value).
  * Test names: simple PARAM_0001 with leading zeros; "nice to have" verbose
    names of length 31/63/127/255 that stress-test downstream UIs, either
    "obnoxious" (differentiation only in the trailing number) or "chunked"
    (gibberish 8-char chunks so the front differs too).

The functions here are deliberately stateless + seeded so a given
(wafer, die) always produces the same test values — reproducible files.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

# Test counts allowed by the spec (orders of magnitude only).
TEST_COUNT_CHOICES = (100, 1_000, 10_000, 100_000, 1_000_000)

# Parametric value shapes and the test limits that go with each shape.
# Limits matter because a FAILING parametric test must report a value that
# actually violates its limits, otherwise the file looks inconsistent.
VALUE_SHAPES = {
    #   shape        (low limit, high limit)
    "uniform":     (0.0, 1.0),      # plain RNG real 0.0..1.0 (spec default)
    "exponential": (1.0, 10.0),     # value = 10^(rng 0..1)  -> 1..10
    "quantized":   (0.0, 1.0),      # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    "signed":      (-1.0, 1.0),     # positive & negative values
    "scientific":  (-9.99e20, 9.99e20),  # X.XXe-YY .. X.XXe+YY
    "constant":    (0.0, 1.0),      # one repeated value (0.5)
}

NAMING_STYLES = ("simple", "obnoxious", "chunked")
VERBOSE_LENGTHS = (31, 63, 127, 255)


@dataclass
class TestPlan:
    """Everything about the synthetic test program.

    parametric_pct: share of tests that are parametric (0..100, 10% steps).
                    The remaining tests are pass/fail (0/1 result).
    """
    count: int = 100
    parametric_pct: int = 50
    value_shape: str = "uniform"
    naming_style: str = "simple"
    name_length: int = 31          # only used by the verbose styles
    names: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.names:
            self.names = make_test_names(
                self.count, self.naming_style, self.name_length)

    @property
    def n_parametric(self) -> int:
        """Number of parametric tests (the first block of test numbers)."""
        return round(self.count * self.parametric_pct / 100)

    def is_parametric(self, test_idx: int) -> bool:
        """Tests 0..n_parametric-1 are parametric, the rest are pass/fail."""
        return test_idx < self.n_parametric

    @property
    def limits(self) -> Tuple[float, float]:
        """(low, high) test limits for the chosen parametric value shape."""
        return VALUE_SHAPES.get(self.value_shape, VALUE_SHAPES["uniform"])


# ---------------------------------------------------------------------------
# Test names
# ---------------------------------------------------------------------------

def _gibberish_chunks(rng: random.Random, total_len: int) -> str:
    """Build a string of 8-char gibberish chunks joined by '_' (spec's
    "less obnoxious verbose" naming), truncated to total_len."""
    chunks = []
    while sum(len(c) for c in chunks) + len(chunks) < total_len:
        chunks.append("".join(rng.choices(string.ascii_uppercase, k=8)))
    return "_".join(chunks)[:total_len]


def make_test_names(count: int, style: str = "simple",
                    length: int = 31) -> List[str]:
    """Generate `count` test names in the requested style.

    simple    : PARAM_0001 — number zero-padded so names sort correctly.
    obnoxious : one fixed long prefix, differentiation ONLY in the trailing
                _number (a UI checker — the interesting part is at the very
                end of a `length`-char string).
    chunked   : gibberish 8-char chunks so the FRONT of each name differs
                too, still ending in _number, total length = `length`.
    """
    digits = max(4, len(str(count)))  # PARAM_0001 style zero-padding

    if style == "simple":
        return [f"PARAM_{i + 1:0{digits}d}" for i in range(count)]

    if style == "obnoxious":
        # Same prefix on every name; only the trailing number differs.
        suffix_len = 1 + digits                       # "_0001"
        prefix = ("VERBOSE_TEST_ITEM_NAME_" * 20)[: length - suffix_len]
        return [f"{prefix}_{i + 1:0{digits}d}"[:length].rjust(length, "X")
                for i in range(count)]

    if style == "chunked":
        # Per-name gibberish prefix (seeded by index -> reproducible).
        suffix_len = 1 + digits
        names = []
        for i in range(count):
            rng = random.Random(i)  # index-seeded so names never change
            prefix = _gibberish_chunks(rng, length - suffix_len)
            names.append(f"{prefix}_{i + 1:0{digits}d}")
        return names

    raise ValueError(f"Unknown naming style: {style!r}")


# ---------------------------------------------------------------------------
# Test values
# ---------------------------------------------------------------------------

def _shaped_value(rng: random.Random, shape: str) -> float:
    """Draw one in-limits parametric value with the requested shape."""
    r = rng.random()
    if shape == "exponential":
        return 10.0 ** r                       # 1 .. 10
    if shape == "quantized":
        return round(r * 5) / 5.0              # 0.0, 0.2, ... 1.0
    if shape == "signed":
        return r * 2.0 - 1.0                   # -1 .. +1
    if shape == "scientific":
        mantissa = 1.0 + r * 8.99              # 1.00 .. 9.99
        exponent = rng.randint(-20, 20)        # e-20 .. e+20
        return mantissa * (10.0 ** exponent)
    if shape == "constant":
        return 0.5                             # one value, always
    return r                                   # uniform 0..1 (default)


def generate_die_results(plan: TestPlan, die_passed: bool,
                         seed: int) -> Iterator[Tuple[int, bool, float, bool]]:
    """Yield (test_number, is_parametric, value, test_passed) for one die.

    Rules from the spec:
      * A PASSING die passes every test.
      * A FAILING die fails at least one test — we pick one seeded-random
        test as the killer; everything after it reports pass too (real
        testers usually stop-on-fail, but we keep emitting for simplicity).
      * Pass/Fail tests report exactly 0.0 (fail) or 1.0 (pass).
      * Parametric tests report an in-limits shaped value when passing and
        an out-of-limits value when failing.

    Seeded per die so re-running the generator reproduces identical data.
    """
    rng = random.Random(seed)
    killer = rng.randrange(plan.count) if not die_passed else -1
    lo, hi = plan.limits

    for t in range(plan.count):
        failed_here = (t == killer)
        if plan.is_parametric(t):
            if failed_here:
                # Report a value 10% past the high limit -> a clear violation.
                span = (hi - lo) or 1.0
                value = hi + 0.1 * span
            else:
                value = _shaped_value(rng, plan.value_shape)
            yield t + 1, True, value, not failed_here
        else:
            # Pass/Fail item: result is literally 0 or 1 per the spec.
            yield t + 1, False, 0.0 if failed_here else 1.0, not failed_here


# ---------------------------------------------------------------------------
# Size guardrail
# ---------------------------------------------------------------------------

def estimate_result_count(test_count: int, dies_per_wafer: int,
                          num_wafers: int) -> int:
    """Total number of per-test records a full export would contain.

    Used by the UI to warn before someone asks for 1M tests x 700 dies x 25
    wafers (which would be a multi-terabyte STDF)."""
    return test_count * dies_per_wafer * num_wafers
