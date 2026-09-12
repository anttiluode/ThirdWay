"""One-off diagnostic for the frozen T6 temporal-order failure.

This script does not change the T6 gate. It asks whether repeated local edit
injections dominate the linearized damage score strongly enough that permuting
the intervening Jacobian order leaves predictive power intact.
"""

from __future__ import annotations

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    COLLISION_THRESHOLD,
    PREDICTOR_EXAMPLES,
    _auroc,
    _score_battery,
    apply_edit,
    batch_hidden_trajectories,
    build_default_pair_battery,
    make_dataset,
    make_seed_model,
)


def _injection_contributions(current, perturbation, dataset, skill):
    subset = dataset.for_skill(skill)
    n = min(PREDICTOR_EXAMPLES, subset.targets.size)
    x = subset.inputs[:n]
    targets = subset.targets[:n]
    states = batch_hidden_trajectories(current, x)
    prediction = states[:, -1, :] @ current.C
    damp = 1.0 - states[:, 1:, :] ** 2
    t_count = x.shape[1]

    magnitudes = []
    last_step_harm = 0.0
    for source_t in range(t_count):
        delta = (states[:, source_t, :] @ perturbation.T) * damp[:, source_t, :]
        for t in range(source_t + 1, t_count):
            delta = (delta @ current.W.T) * damp[:, t, :]
        delta_output = delta @ current.C
        signed_loss = 2.0 * (prediction - targets) * delta_output
        magnitudes.append(float(np.mean(np.abs(signed_loss))))
        if source_t == t_count - 1:
            last_step_harm = float(np.mean(np.maximum(signed_loss, 0.0)))

    magnitudes = np.asarray(magnitudes, dtype=float)
    total = float(np.sum(magnitudes))
    late = float(np.sum(magnitudes[-4:]) / max(total, 1e-15))
    final = float(magnitudes[-1] / max(total, 1e-15))
    return late, final, last_step_harm


def main():
    cases = build_default_pair_battery()
    base = make_seed_model(seed=23)
    data = make_dataset(seed=53, examples_per_skill=128)
    score_map = _score_battery(cases, base, data)
    test = np.asarray([case.split == "test" for case in cases], dtype=bool)
    truth = np.asarray(
        [case.damage.old_skill_damage > COLLISION_THRESHOLD for case in cases],
        dtype=bool,
    )

    ordered = score_map["causal"]
    shuffled = score_map["causal_shuffled"]
    corr = float(np.corrcoef(ordered[test], shuffled[test])[0, 1])
    rel_diff = float(
        np.mean(np.abs(ordered[test] - shuffled[test]))
        / max(float(np.mean(np.abs(ordered[test]))), 1e-15)
    )

    late_fractions = []
    final_fractions = []
    last_scores = []
    for case in cases:
        retained_model = apply_edit(base, case.retained)
        late, final, last_score = _injection_contributions(
            retained_model,
            case.candidate.delta_w,
            data,
            case.retained.skill,
        )
        late_fractions.append(late)
        final_fractions.append(final)
        last_scores.append(last_score)

    late_fractions = np.asarray(late_fractions)
    final_fractions = np.asarray(final_fractions)
    last_scores = np.asarray(last_scores)

    print(f"corr(ordered, shuffled) heldout={corr:.6f}")
    print(f"mean relative ordered-shuffled score difference={rel_diff:.6f}")
    print(f"mean late-4 injection contribution fraction={np.mean(late_fractions[test]):.6f}")
    print(f"mean final-injection contribution fraction={np.mean(final_fractions[test]):.6f}")
    print(f"last-step-only heldout AUROC={_auroc(last_scores[test], truth[test]):.6f}")


if __name__ == "__main__":
    main()
