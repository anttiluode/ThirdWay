"""Gate T3: credit cannot be finer than identifiable computational identity.

This gate connects three older lines:

- MovingProblem / AlgoSchalgo: individual spectral axes become unreliable near
  degeneracy; the identifiable object can be a subspace rather than a vector.
- GeometricNeuronV24: if the current observation geometry cannot distinguish
  hypotheses, another addressed measurement can.
- ThirdWay T0-T2: delayed scalar consequence can rewrite a router only when the
  causal address remains attached to the continuing computation.

Part A uses repeated noisy eigenvalue crossings. A one-vector-at-a-time overlap
tracker is compared with an adaptive ambiguity-block tracker. The block tracker
carries the previous semantic basis through a temporarily ambiguous subspace by
an orthogonal Procrustes lift.

Part B is the attacker. During an exact degeneracy the hidden semantic axes are
rotated inside the degenerate subspace. The primary operator is *exactly blind*
to that rotation, so no passive primary-operator tracker is entitled to assign
individual identity. An optional second operator, queried only while the
primary spectrum is ambiguous, breaks the gauge and restores fine-grained
credit.

The reward itself remains a delayed scalar. The route address lives in the
tracked substrate, not in the reward packet.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RouteResult:
    route_accuracy: float
    reward: float
    task_accuracy: float
    semantic_fidelity: float
    query_fraction: float
    ambiguity_fraction: float


@dataclass(frozen=True)
class BatteryResult:
    route_accuracy: float
    reward: float
    task_accuracy: float
    semantic_fidelity: float
    query_fraction: float
    ambiguity_fraction: float
    min_route_accuracy: float


def _givens(d: int, i: int, j: int, theta: float) -> np.ndarray:
    r = np.eye(d)
    c, s = np.cos(theta), np.sin(theta)
    r[i, i] = c
    r[j, j] = c
    r[i, j] = -s
    r[j, i] = s
    return r


def _base_frame(t: int) -> np.ndarray:
    """Smooth four-dimensional moving representation."""
    return (
        _givens(4, 0, 2, 0.35 * np.sin(2.0 * np.pi * t / 600.0))
        @ _givens(4, 1, 3, 0.28 * np.sin(2.0 * np.pi * t / 850.0 + 0.4))
        @ _givens(4, 2, 3, 0.22 * np.sin(2.0 * np.pi * t / 1100.0 + 1.1))
    )


def _internal_rotation(phi: float) -> np.ndarray:
    return _givens(4, 0, 1, phi)


def _crossing_values(t: int, period: int = 300) -> np.ndarray:
    u = np.cos(2.0 * np.pi * t / period)
    return np.asarray([0.75 + 0.20 * u, 0.75 - 0.20 * u, 0.25, -0.35])


def _eigh_desc(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _match_vectors(previous: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Brute-force nearest-overlap vector tracker; dimension is only four."""
    d = previous.shape[1]
    best_score = -np.inf
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(d)):
        score = sum(abs(previous[:, i] @ vecs[:, perm[i]]) for i in range(d))
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    out = vecs[:, best_perm].copy()
    for i in range(d):
        if previous[:, i] @ out[:, i] < 0:
            out[:, i] *= -1.0
    return out


def _spectral_clusters(vals: np.ndarray, threshold: float) -> list[list[int]]:
    clusters: list[list[int]] = []
    start = 0
    for j in range(len(vals) - 1):
        if abs(vals[j] - vals[j + 1]) >= threshold:
            clusters.append(list(range(start, j + 1)))
            start = j + 1
    clusters.append(list(range(start, len(vals))))
    return clusters


def _closest_basis_in_subspace(u: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(u.T @ previous_block, full_matrices=False)
    return u @ (left @ right_t)


def _adaptive_block_update(
    previous: np.ndarray,
    vals: np.ndarray,
    vecs: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, bool]:
    """Track only the granularity the current spectrum actually identifies."""
    d = len(vals)
    clusters = _spectral_clusters(vals, threshold)
    remaining = set(range(d))
    out = np.zeros_like(previous)
    ambiguous = False

    for inds in sorted(clusters, key=len, reverse=True):
        u = vecs[:, inds]
        m = len(inds)
        best: tuple[float, tuple[int, ...]] | None = None
        for subset in itertools.combinations(sorted(remaining), m):
            overlap = np.linalg.norm(u.T @ previous[:, subset], "fro") ** 2
            if best is None or overlap > best[0]:
                best = (float(overlap), subset)
        assert best is not None
        labels = sorted(best[1])

        if m == 1:
            v = u[:, 0].copy()
            if previous[:, labels[0]] @ v < 0:
                v *= -1.0
            out[:, labels[0]] = v
        else:
            ambiguous = True
            out[:, labels] = _closest_basis_in_subspace(u, previous[:, labels])

        for label in labels:
            remaining.remove(label)

    return out, ambiguous


def _route_distribution(
    logits: np.ndarray, *, beta: float = 3.0, exploration: float = 0.08
) -> np.ndarray:
    scaled = beta * logits
    p = np.exp(scaled - np.max(scaled))
    p /= np.sum(p)
    return (1.0 - exploration) * p + exploration / p.size


def _sample_world(rng: np.random.Generator, context: int) -> tuple[np.ndarray, float]:
    """Same-answer/different-computation toy inherited from the GAx bridge."""
    target = 1.0 if rng.random() < 0.5 else -1.0
    if context == 0:
        x0 = target
        x1 = 1.0 if rng.random() < 0.5 else -1.0
        x2 = 1.0 if rng.random() < 0.5 else -1.0
    else:
        x1 = 1.0 if rng.random() < 0.5 else -1.0
        x2 = target * x1
        x0 = 1.0 if rng.random() < 0.5 else -1.0
    predictions = np.tanh(4.0 * np.asarray([x0, x1 * x2], dtype=float))
    return predictions, target


def _semantic_route(frame: np.ndarray, estimate: np.ndarray, label: int) -> int:
    overlap = np.abs(frame[:, :2].T @ estimate[:, label])
    return int(np.argmax(overlap))


def _fidelity(frame: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(np.sum(frame[:, :2] * estimate[:, :2], axis=0))))


def _learn_with_tracker(
    seed: int,
    tracker: str,
    *,
    events: int = 5000,
    noise: float = 0.020,
    ambiguity_threshold: float = 0.12,
    hidden_gauge: bool = False,
    plateau: tuple[int, int] = (1500, 2500),
    eligibility_decay: float = 0.85,
    learning_rate: float = 0.015,
    delay_min: int = 3,
    delay_max: int = 10,
    tail: int = 1000,
) -> RouteResult:
    if tracker not in {"overlap", "blocks", "active_family", "oracle"}:
        raise ValueError(f"unknown tracker: {tracker}")

    rng = np.random.default_rng(seed)
    gain_logits = np.zeros((2, 2), dtype=float)
    eligibility = np.zeros((2, 2), dtype=float)
    reward_buckets: list[list[float]] = [[] for _ in range(delay_max + 1)]

    def world(t: int) -> tuple[np.ndarray, np.ndarray, bool]:
        if not hidden_gauge:
            return _base_frame(t), _crossing_values(t), False
        a, b = plateau
        if t < a:
            phi = 0.0
            pair = (0.95, 0.55)
            hidden = False
        elif t <= b:
            phi = ((t - a) / max(1, b - a)) * (np.pi / 2.0)
            pair = (0.75, 0.75)
            hidden = True
        else:
            phi = np.pi / 2.0
            pair = (0.95, 0.55)
            hidden = False
        frame = _base_frame(t) @ _internal_rotation(phi)
        vals = np.asarray([pair[0], pair[1], 0.25, -0.35])
        return frame, vals, hidden

    frame, true_vals, _ = world(0)
    a0 = frame @ np.diag(true_vals) @ frame.T
    e0 = rng.normal(size=(4, 4))
    e0 = 0.5 * (e0 + e0.T)
    vals, vecs = _eigh_desc(a0 + noise * e0)
    estimate = _match_vectors(frame, vecs)  # one initial grounding only

    history: list[tuple[bool, float, bool]] = []
    fidelities: list[float] = []
    queries = 0
    ambiguous_steps = 0

    for t in range(events):
        frame, true_vals, _ = world(t)

        if t > 0:
            primary = frame @ np.diag(true_vals) @ frame.T
            eps = rng.normal(size=(4, 4))
            eps = 0.5 * (eps + eps.T)
            vals_a, vecs_a = _eigh_desc(primary + noise * eps)
            primary_gap = abs(vals_a[0] - vals_a[1])
            ambiguous = primary_gap < ambiguity_threshold
            ambiguous_steps += int(ambiguous)

            if tracker == "oracle":
                estimate = frame.copy()
            elif tracker == "overlap":
                estimate = _match_vectors(estimate, vecs_a)
            elif tracker == "blocks":
                estimate, _ = _adaptive_block_update(
                    estimate, vals_a, vecs_a, ambiguity_threshold
                )
            else:  # active_family
                if ambiguous:
                    # A second operator shares the same computational frame but
                    # has a stable signature that breaks the primary gauge.
                    secondary_vals = np.asarray([1.15, 0.55, 0.15, -0.35])
                    secondary = frame @ np.diag(secondary_vals) @ frame.T
                    eps_b = rng.normal(size=(4, 4))
                    eps_b = 0.5 * (eps_b + eps_b.T)
                    _, vecs_b = _eigh_desc(secondary + noise * eps_b)
                    estimate = _match_vectors(estimate, vecs_b)
                    queries += 1
                else:
                    estimate, _ = _adaptive_block_update(
                        estimate, vals_a, vecs_a, ambiguity_threshold
                    )

        eligibility *= eligibility_decay
        bucket = reward_buckets[t % len(reward_buckets)]
        for reward in bucket:
            gain_logits += learning_rate * reward * eligibility
            gain_logits -= np.mean(gain_logits, axis=1, keepdims=True)
            np.clip(gain_logits, -4.0, 4.0, out=gain_logits)
        bucket.clear()

        context = int(rng.integers(2))
        route_prob = _route_distribution(gain_logits[context])
        label = 0 if rng.random() < route_prob[0] else 1
        semantic_route = _semantic_route(frame, estimate, label)

        predictions, target = _sample_world(rng, context)
        prediction = float(predictions[semantic_route])
        reward = float(target * prediction)

        score = -route_prob.copy()
        score[label] += 1.0
        eligibility[context] += score

        delay = int(rng.integers(delay_min, delay_max + 1))
        reward_buckets[(t + delay) % len(reward_buckets)].append(reward)

        history.append(
            (semantic_route == context, reward, np.sign(prediction) == target)
        )
        fidelities.append(_fidelity(frame, estimate))

    late = history[-tail:]
    return RouteResult(
        route_accuracy=float(np.mean([x[0] for x in late])),
        reward=float(np.mean([x[1] for x in late])),
        task_accuracy=float(np.mean([x[2] for x in late])),
        semantic_fidelity=float(np.mean(fidelities[-tail:])),
        query_fraction=float(queries / max(1, events - 1)),
        ambiguity_fraction=float(ambiguous_steps / max(1, events - 1)),
    )


def _battery(
    tracker: str,
    *,
    hidden_gauge: bool,
    seeds: Iterable[int] = range(8),
) -> BatteryResult:
    runs = [
        _learn_with_tracker(int(seed), tracker, hidden_gauge=hidden_gauge)
        for seed in seeds
    ]
    return BatteryResult(
        route_accuracy=float(np.mean([r.route_accuracy for r in runs])),
        reward=float(np.mean([r.reward for r in runs])),
        task_accuracy=float(np.mean([r.task_accuracy for r in runs])),
        semantic_fidelity=float(np.mean([r.semantic_fidelity for r in runs])),
        query_fraction=float(np.mean([r.query_fraction for r in runs])),
        ambiguity_fraction=float(np.mean([r.ambiguity_fraction for r in runs])),
        min_route_accuracy=float(min(r.route_accuracy for r in runs)),
    )


def run_battery(seeds: Iterable[int] = range(8)) -> dict[str, dict[str, BatteryResult]]:
    seeds = tuple(int(s) for s in seeds)
    crossing = {
        tracker: _battery(tracker, hidden_gauge=False, seeds=seeds)
        for tracker in ("overlap", "blocks", "oracle")
    }
    hidden = {
        tracker: _battery(tracker, hidden_gauge=True, seeds=seeds)
        for tracker in ("overlap", "blocks", "active_family", "oracle")
    }
    return {"crossing": crossing, "hidden_gauge": hidden}


def main() -> None:
    result = run_battery()
    for world_name, rows in result.items():
        print(world_name)
        for name, r in rows.items():
            print(
                f"  {name:13s} route={r.route_accuracy:.6f} "
                f"reward={r.reward:.6f} task={r.task_accuracy:.6f} "
                f"fidelity={r.semantic_fidelity:.6f} "
                f"query={r.query_fraction:.6f} ambiguity={r.ambiguity_fraction:.6f} "
                f"min_route={r.min_route_accuracy:.6f}"
            )

    c = result["crossing"]
    h = result["hidden_gauge"]
    passed = (
        c["blocks"].route_accuracy > 0.90
        and c["blocks"].semantic_fidelity > 0.98
        and c["overlap"].route_accuracy < 0.65
        and abs(c["blocks"].route_accuracy - c["oracle"].route_accuracy) < 0.03
        and h["blocks"].route_accuracy < 0.60
        and h["active_family"].route_accuracy > 0.90
        and h["active_family"].semantic_fidelity > 0.98
        and h["active_family"].query_fraction < 0.30
        and abs(h["active_family"].route_accuracy - h["oracle"].route_accuracy) < 0.05
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
