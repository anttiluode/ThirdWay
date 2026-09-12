"""Gate T1: a computational highway can move through coordinates.

Two mode identities rotate continuously through a 2-D parameter basis. Candidate
vectors are presented unordered, with random sign and small noise. A fixed
prototype matcher becomes stale when the roads move far from their initial
coordinates. A continuation tracker matches each new mode to the previous one
and preserves identity.

The exact transport operator satisfies L_t E_k(t) = E_k(t+1) up to floating
point precision; the observed tracker must recover the same continuation from
noisy unordered vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RunResult:
    fixed_identity_accuracy: float
    covariant_identity_accuracy: float
    fixed_route_reward: float
    covariant_route_reward: float
    max_exact_covariance_residual: float


@dataclass(frozen=True)
class BatteryResult:
    fixed_identity_accuracy: float
    covariant_identity_accuracy: float
    fixed_route_reward: float
    covariant_route_reward: float
    max_exact_covariance_residual: float
    min_covariant_identity_accuracy: float


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _assignment(prototypes: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Map prototype identity -> candidate index by maximum total |cosine|."""
    score = np.abs(prototypes.T @ candidates)
    identity_score = score[0, 0] + score[1, 1]
    swapped_score = score[0, 1] + score[1, 0]
    return np.asarray([0, 1] if identity_score >= swapped_score else [1, 0])


def run(seed: int, *, steps: int = 240, noise: float = 0.015) -> RunResult:
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    dtheta = float(rng.uniform(0.025, 0.045))

    initial = rotation(theta0)
    fixed_prototypes = initial.copy()
    tracked_prototypes = initial.copy()
    previous_exact = initial.copy()

    fixed_identity = []
    covariant_identity = []
    fixed_reward = []
    covariant_reward = []
    residuals = []

    for t in range(1, steps + 1):
        theta = theta0 + dtheta * t
        exact_modes = rotation(theta)  # columns are the persistent identities

        order = np.asarray([0, 1] if rng.random() > 0.5 else [1, 0])
        candidates = exact_modes[:, order].copy()
        candidates *= np.where(rng.random(2) < 0.5, -1.0, 1.0)
        candidates += noise * rng.normal(size=candidates.shape)
        candidates /= np.linalg.norm(candidates, axis=0, keepdims=True)

        fixed_assignment = _assignment(fixed_prototypes, candidates)
        covariant_assignment = _assignment(tracked_prototypes, candidates)

        fixed_identity_guess = order[fixed_assignment]
        covariant_identity_guess = order[covariant_assignment]
        fixed_ok = bool(np.all(fixed_identity_guess == np.asarray([0, 1])))
        covariant_ok = bool(np.all(covariant_identity_guess == np.asarray([0, 1])))
        fixed_identity.append(fixed_ok)
        covariant_identity.append(covariant_ok)

        # A context asks for one computational identity. Selection itself is
        # trivial once identity is known; the test is whether the router's
        # identity map stayed attached to the moving road.
        context = int(rng.integers(2))
        fixed_selected_identity = int(order[fixed_assignment[context]])
        covariant_selected_identity = int(order[covariant_assignment[context]])
        fixed_reward.append(1.0 if fixed_selected_identity == context else -1.0)
        covariant_reward.append(1.0 if covariant_selected_identity == context else -1.0)

        # Update only the continuation tracker, sign-aligned to yesterday's road.
        new_tracked = np.column_stack(
            [
                candidates[:, covariant_assignment[0]],
                candidates[:, covariant_assignment[1]],
            ]
        )
        signs = np.sign(np.sum(tracked_prototypes * new_tracked, axis=0))
        signs[signs == 0.0] = 1.0
        tracked_prototypes = new_tracked * signs

        # Exact cocycle transport for the noiseless moving basis.
        transport = exact_modes @ previous_exact.T
        residuals.append(
            float(np.linalg.norm(transport @ previous_exact - exact_modes, ord="fro"))
        )
        previous_exact = exact_modes

    return RunResult(
        fixed_identity_accuracy=float(np.mean(fixed_identity)),
        covariant_identity_accuracy=float(np.mean(covariant_identity)),
        fixed_route_reward=float(np.mean(fixed_reward)),
        covariant_route_reward=float(np.mean(covariant_reward)),
        max_exact_covariance_residual=float(np.max(residuals)),
    )


def run_battery(seeds: Iterable[int] = range(8)) -> BatteryResult:
    runs = [run(int(seed)) for seed in seeds]
    return BatteryResult(
        fixed_identity_accuracy=float(np.mean([r.fixed_identity_accuracy for r in runs])),
        covariant_identity_accuracy=float(
            np.mean([r.covariant_identity_accuracy for r in runs])
        ),
        fixed_route_reward=float(np.mean([r.fixed_route_reward for r in runs])),
        covariant_route_reward=float(np.mean([r.covariant_route_reward for r in runs])),
        max_exact_covariance_residual=float(
            max(r.max_exact_covariance_residual for r in runs)
        ),
        min_covariant_identity_accuracy=float(
            min(r.covariant_identity_accuracy for r in runs)
        ),
    )


def main() -> None:
    r = run_battery()
    print("Gate T1 — moving road / covariant computational identity")
    print(f"fixed identity accuracy:     {r.fixed_identity_accuracy:.6f}")
    print(f"covariant identity accuracy: {r.covariant_identity_accuracy:.6f}")
    print(f"fixed route reward:          {r.fixed_route_reward:.6f}")
    print(f"covariant route reward:      {r.covariant_route_reward:.6f}")
    print(f"max exact covariance residual: {r.max_exact_covariance_residual:.3e}")

    passed = (
        r.fixed_identity_accuracy < 0.60
        and r.covariant_identity_accuracy > 0.99
        and r.min_covariant_identity_accuracy > 0.99
        and r.covariant_route_reward > 0.99
        and r.max_exact_covariance_residual < 1e-12
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
