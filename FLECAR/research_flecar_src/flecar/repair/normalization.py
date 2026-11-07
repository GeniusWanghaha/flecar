from __future__ import annotations


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def bounded_ratio(x: float, *, scale: float = 1.0) -> float:
    x_pos = max(0.0, float(x))
    s = max(1e-9, float(scale))
    return float(x_pos / (x_pos + s))


def normalize_nonnegative_surrogate(x: float, *, scale: float = 1.0) -> float:
    return bounded_ratio(float(x), scale=float(scale))


def normalize_margin_deficit(margin: float, *, scale: float = 1.0) -> float:
    return bounded_ratio(max(0.0, -float(margin)), scale=float(scale))


def normalize_fraction(num: float, den: float, *, default: float = 0.0) -> float:
    den_safe = float(den)
    if den_safe <= 1e-9:
        return clip01(float(default))
    return clip01(float(num) / den_safe)

