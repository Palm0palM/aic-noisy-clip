"""Unmodified numpy-only functions from teammate balance.py; strength fixed by caller."""
import numpy as np

def probabilities(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or min(p.shape) < 1 or not np.isfinite(p).all():
        raise ValueError("Expected a nonempty finite [samples, classes] matrix")
    if (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("Invalid probabilities")
    return p / p.sum(1, keepdims=True)

def uniform_alignment(value, strength=0.5, temperature=1.0, smoothing=0.0,
                      max_iter=1000, tolerance=1e-7):
    """Sinkhorn class-bias correction; soft mass does not fix hard counts.

    smoothing is an EXPERIMENTAL inversion of the known training smoothing,
    not an assertion that the network probabilities are perfectly calibrated.
    """
    p = probabilities(value)
    if not 0 <= strength <= 1 or temperature <= 0 or not 0 <= smoothing < 1:
        raise ValueError("Invalid alignment parameters")
    if max_iter <= 0 or tolerance <= 0:
        raise ValueError("Invalid convergence controls")
    p = np.maximum(p - smoothing / p.shape[1], 1e-12)
    p /= p.sum(1, keepdims=True)
    logp = np.log(p) / temperature
    q = np.exp(logp - logp.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    bias = np.zeros(p.shape[1])
    target = len(p) / p.shape[1]
    error = np.inf
    for step in range(max_iter):
        factor = target / np.maximum(q.sum(0), 1e-300)
        q *= factor
        bias += np.log(factor)
        q /= np.maximum(q.sum(1, keepdims=True), 1e-300)
        error = float(np.max(np.abs(q.sum(0) / target - 1)))
        if error < tolerance:
            break
    scores = logp + strength * bias[None, :]
    scores -= scores.max(1, keepdims=True)
    aligned = np.exp(scores)
    aligned /= aligned.sum(1, keepdims=True)
    return aligned, {"iterations": step + 1, "max_relative_mass_error": error,
                     "converged": bool(error < tolerance), "strength": strength,
                     "temperature": temperature, "smoothing": smoothing}
