"""Gate T0: route-local delayed credit rewrites context-dependent modal gain.

This is deliberately a minimal fusion gate. The computational highways are
seeded (direct vs relational) so the gate asks only whether delayed scalar
consequence can rewrite the router when the address is stored locally rather
than carried by the reward packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class RunResult:
    route_probabilities: np.ndarray
    late_route_accuracy: float
    late_reward: float
    late_task_accuracy: float


@dataclass(frozen=True)
class BatteryResult:
    route_accuracy: float
    reward: float
    task_accuracy: float
    mean_route_probabilities: np.ndarray
    min_route_accuracy: float


def _sample_world(rng: np.random.Generator, context: int) -> tuple[np.ndarray, float]:
    """Return two specialist predictions and the target.

    Context 0 makes x0 causal and the relation x1*x2 incidental.
    Context 1 makes x1*x2 causal and x0 incidental.
    """
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


def _route_distribution(
    logits: np.ndarray, *, beta: float, exploration: float
) -> np.ndarray:
    """Normalized positive diagonal gain with an exploration floor."""
    scaled = beta * logits
    positive = np.exp(scaled - np.max(scaled))
    positive /= np.sum(positive)
    return (1.0 - exploration) * positive + exploration / positive.size


def run(
    seed: int,
    mode: str = "local",
    *,
    events: int = 5000,
    eligibility_decay: float = 0.85,
    learning_rate: float = 0.015,
    beta: float = 3.0,
    exploration: float = 0.08,
    delay_min: int = 3,
    delay_max: int = 10,
    tail: int = 1000,
) -> RunResult:
    """Run one deterministic seed.

    Modes:
      local    - delayed reward multiplies route/context-local eligibility.
      pooled   - all eligibility addresses are collapsed to one scalar.
      shuffled - route identity is swapped before the update.
      current  - delayed history is discarded.
    """
    if mode not in {"local", "pooled", "shuffled", "current"}:
        raise ValueError(f"unknown mode: {mode}")

    rng = np.random.default_rng(seed)
    gain_logits = np.zeros((2, 2), dtype=float)
    eligibility = np.zeros((2, 2), dtype=float)

    # Ring buffer. Reward packets contain only a scalar; no context/route tag.
    reward_buckets: list[list[float]] = [[] for _ in range(delay_max + 1)]
    history: list[tuple[bool, float, bool]] = []

    for t in range(events):
        eligibility *= eligibility_decay

        bucket = reward_buckets[t % len(reward_buckets)]
        for reward in bucket:
            if mode == "local":
                credit = eligibility
            elif mode == "pooled":
                credit = np.full_like(eligibility, np.mean(eligibility))
            elif mode == "shuffled":
                credit = eligibility[:, ::-1]
            else:  # current-only: no causal trace survived the delay
                credit = np.zeros_like(eligibility)

            gain_logits += learning_rate * reward * credit
            # Only relative gain matters inside each context.
            gain_logits -= np.mean(gain_logits, axis=1, keepdims=True)
            np.clip(gain_logits, -4.0, 4.0, out=gain_logits)
        bucket.clear()

        context = int(rng.integers(2))
        predictions, target = _sample_world(rng, context)
        route_prob = _route_distribution(
            gain_logits[context], beta=beta, exploration=exploration
        )
        route = 0 if rng.random() < route_prob[0] else 1
        prediction = float(predictions[route])
        reward = float(target * prediction)

        # Local policy-score trace. The future reward does not carry this address.
        score = -route_prob.copy()
        score[route] += 1.0
        if mode != "current":
            eligibility[context] += score

        delay = int(rng.integers(delay_min, delay_max + 1))
        reward_buckets[(t + delay) % len(reward_buckets)].append(reward)
        history.append((route == context, reward, np.sign(prediction) == target))

    late = history[-tail:]
    final_probs = np.vstack(
        [
            _route_distribution(
                gain_logits[c], beta=beta, exploration=exploration
            )
            for c in range(2)
        ]
    )
    return RunResult(
        route_probabilities=final_probs,
        late_route_accuracy=float(np.mean([x[0] for x in late])),
        late_reward=float(np.mean([x[1] for x in late])),
        late_task_accuracy=float(np.mean([x[2] for x in late])),
    )


def run_battery(
    seeds: Iterable[int] = range(8),
    modes: Iterable[str] = ("local", "pooled", "current", "shuffled"),
) -> Dict[str, BatteryResult]:
    out: Dict[str, BatteryResult] = {}
    for mode in modes:
        runs = [run(int(seed), mode=mode) for seed in seeds]
        out[mode] = BatteryResult(
            route_accuracy=float(np.mean([r.late_route_accuracy for r in runs])),
            reward=float(np.mean([r.late_reward for r in runs])),
            task_accuracy=float(np.mean([r.late_task_accuracy for r in runs])),
            mean_route_probabilities=np.mean(
                [r.route_probabilities for r in runs], axis=0
            ),
            min_route_accuracy=float(min(r.late_route_accuracy for r in runs)),
        )
    return out


def _fmt(result: BatteryResult) -> str:
    return (
        f"route={result.route_accuracy:.6f}  "
        f"reward={result.reward:.6f}  "
        f"task={result.task_accuracy:.6f}  "
        f"min_route={result.min_route_accuracy:.6f}"
    )


def main() -> None:
    results = run_battery()
    print("Gate T0 — delayed scalar consequence / local route address")
    for mode, result in results.items():
        print(f"{mode:8s} {_fmt(result)}")
        print(result.mean_route_probabilities)

    local = results["local"]
    pooled = results["pooled"]
    shuffled = results["shuffled"]
    passed = (
        local.route_accuracy > 0.90
        and local.min_route_accuracy > 0.90
        and local.reward > pooled.reward + 0.30
        and shuffled.route_accuracy < 0.10
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
