"""Gate T2: delayed credit must move with the computational highway.

T0 showed that delayed scalar consequence can rewrite a router when the route
address is preserved locally. T1 showed that route identity can continue while
its coordinates move. T2 combines them: eligibility is stored in physical
coordinates while the basis rotates during the reward delay.

Two learners are compared:
  covariant - transport eligibility with the moving basis before reward arrives.
  stale     - let eligibility decay in yesterday's coordinates.

An internal modal shadow is used only as a diagnostic to verify that covariant
transport preserves the intended route-local credit to floating-point error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RunResult:
    late_route_accuracy: float
    late_reward: float
    late_task_accuracy: float
    max_modal_credit_mismatch: float


@dataclass(frozen=True)
class BatteryResult:
    route_accuracy: float
    reward: float
    task_accuracy: float
    max_modal_credit_mismatch: float
    min_route_accuracy: float


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=float)


def route_distribution(
    logits: np.ndarray, *, beta: float, exploration: float
) -> np.ndarray:
    scaled = beta * logits
    p = np.exp(scaled - np.max(scaled))
    p /= np.sum(p)
    return (1.0 - exploration) * p + exploration / p.size


def sample_world(rng: np.random.Generator, context: int) -> tuple[np.ndarray, float]:
    target = 1.0 if rng.random() < 0.5 else -1.0
    if context == 0:
        x0 = target
        x1 = 1.0 if rng.random() < 0.5 else -1.0
        x2 = 1.0 if rng.random() < 0.5 else -1.0
    else:
        x1 = 1.0 if rng.random() < 0.5 else -1.0
        x2 = target * x1
        x0 = 1.0 if rng.random() < 0.5 else -1.0
    return np.tanh(4.0 * np.asarray([x0, x1 * x2], dtype=float)), target


def run(
    seed: int,
    mode: str = "covariant",
    *,
    events: int = 5000,
    eligibility_decay: float = 0.85,
    learning_rate: float = 0.015,
    beta: float = 3.0,
    exploration: float = 0.08,
    delay_min: int = 3,
    delay_max: int = 10,
    dtheta: float = 0.32,
    tail: int = 1000,
) -> RunResult:
    if mode not in {"covariant", "stale"}:
        raise ValueError(f"unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    gain_logits = np.zeros((2, 2), dtype=float)

    # One physical coordinate vector per context. Modal identity is encoded by
    # direction in the current basis, not by a reward label.
    eligibility_physical = np.zeros((2, 2), dtype=float)

    # Diagnostic shadow in ideal modal coordinates. It is never used to learn.
    eligibility_modal_shadow = np.zeros((2, 2), dtype=float)

    reward_buckets: list[list[float]] = [[] for _ in range(delay_max + 1)]
    history: list[tuple[bool, float, bool]] = []

    theta = 0.0
    basis = rotation(theta)
    max_mismatch = 0.0

    for t in range(events):
        next_basis = rotation(theta + dtheta)
        transport = next_basis @ basis.T

        eligibility_physical *= eligibility_decay
        eligibility_modal_shadow *= eligibility_decay
        if mode == "covariant":
            eligibility_physical = (transport @ eligibility_physical.T).T
        # stale mode intentionally leaves the trace in old coordinates

        basis = next_basis
        theta += dtheta

        mismatch = np.max(
            np.abs(eligibility_physical @ basis - eligibility_modal_shadow)
        )
        max_mismatch = max(max_mismatch, float(mismatch))

        bucket = reward_buckets[t % len(reward_buckets)]
        for reward in bucket:
            modal_credit = eligibility_physical @ basis
            gain_logits += learning_rate * reward * modal_credit
            gain_logits -= np.mean(gain_logits, axis=1, keepdims=True)
            np.clip(gain_logits, -4.0, 4.0, out=gain_logits)
        bucket.clear()

        context = int(rng.integers(2))
        predictions, target = sample_world(rng, context)
        route_prob = route_distribution(
            gain_logits[context], beta=beta, exploration=exploration
        )
        route = 0 if rng.random() < route_prob[0] else 1
        prediction = float(predictions[route])
        reward = float(target * prediction)

        score = -route_prob.copy()
        score[route] += 1.0

        # Store the same event in physical and diagnostic modal coordinates.
        eligibility_physical[context] += basis @ score
        eligibility_modal_shadow[context] += score

        mismatch = np.max(
            np.abs(eligibility_physical @ basis - eligibility_modal_shadow)
        )
        max_mismatch = max(max_mismatch, float(mismatch))

        delay = int(rng.integers(delay_min, delay_max + 1))
        reward_buckets[(t + delay) % len(reward_buckets)].append(reward)
        history.append((route == context, reward, np.sign(prediction) == target))

    late = history[-tail:]
    return RunResult(
        late_route_accuracy=float(np.mean([x[0] for x in late])),
        late_reward=float(np.mean([x[1] for x in late])),
        late_task_accuracy=float(np.mean([x[2] for x in late])),
        max_modal_credit_mismatch=max_mismatch,
    )


def run_battery(
    seeds: Iterable[int] = range(8),
    modes: Iterable[str] = ("covariant", "stale"),
) -> dict[str, BatteryResult]:
    out: dict[str, BatteryResult] = {}
    for mode in modes:
        runs = [run(int(seed), mode=mode) for seed in seeds]
        out[mode] = BatteryResult(
            route_accuracy=float(np.mean([r.late_route_accuracy for r in runs])),
            reward=float(np.mean([r.late_reward for r in runs])),
            task_accuracy=float(np.mean([r.late_task_accuracy for r in runs])),
            max_modal_credit_mismatch=float(
                max(r.max_modal_credit_mismatch for r in runs)
            ),
            min_route_accuracy=float(min(r.late_route_accuracy for r in runs)),
        )
    return out


def main() -> None:
    r = run_battery()
    for mode, result in r.items():
        print(
            f"{mode:9s} route={result.route_accuracy:.6f} "
            f"reward={result.reward:.6f} task={result.task_accuracy:.6f} "
            f"max_mismatch={result.max_modal_credit_mismatch:.3e} "
            f"min_route={result.min_route_accuracy:.6f}"
        )

    passed = (
        r["covariant"].route_accuracy > 0.90
        and r["covariant"].min_route_accuracy > 0.90
        and r["covariant"].max_modal_credit_mismatch < 1e-12
        and r["stale"].route_accuracy < 0.30
        and r["covariant"].reward > r["stale"].reward + 0.60
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
