"""Predeclared T6 failure characterization at 0.5x, 1.0x and 1.5x candidate strength.

The primary 1.0x world and gate are unchanged. Only the incoming candidate edit
is rescaled; the already-retained edit remains at its original strength.
"""

from __future__ import annotations

import numpy as np

from experiments.gate_t6_persistent_skill_compatibility import (
    COLLISION_THRESHOLD,
    CandidateEdit,
    PairCase,
    _auroc,
    _correlation,
    _score_battery,
    apply_edit,
    build_default_pair_battery,
    make_dataset,
    make_seed_model,
    mean_squared_loss,
    measure_persistent_damage,
)


FACTORS = (0.5, 1.0, 1.5)
STATIC = ("parameter", "gradient", "fisher", "static_jacobian")


def _scaled_candidate(base, proposal, edit, factor):
    delta = factor * edit.delta_w
    trial = CandidateEdit(
        skill=edit.skill,
        seed=edit.seed,
        delta_w=delta,
        base_loss=mean_squared_loss(base, proposal, edit.skill),
        edited_loss=0.0,
    )
    edited_model = apply_edit(base, trial)
    return CandidateEdit(
        skill=edit.skill,
        seed=edit.seed,
        delta_w=delta,
        base_loss=trial.base_loss,
        edited_loss=mean_squared_loss(edited_model, proposal, edit.skill),
    )


def main():
    original = build_default_pair_battery()
    base = make_seed_model(seed=23)
    proposal = make_dataset(seed=41, examples_per_skill=128)
    evaluation = make_dataset(seed=53, examples_per_skill=128)

    for factor in FACTORS:
        cache = {}
        cases = []
        for case in original:
            identity = case.candidate.identity
            if identity not in cache:
                cache[identity] = _scaled_candidate(base, proposal, case.candidate, factor)
            candidate = cache[identity]
            damage = measure_persistent_damage(base, case.retained, candidate, evaluation)
            cases.append(
                PairCase(
                    retained=case.retained,
                    candidate=candidate,
                    split=case.split,
                    damage=damage,
                )
            )

        score_map = _score_battery(cases, base, evaluation)
        test = np.asarray([case.split == "test" for case in cases], dtype=bool)
        damage = np.asarray([case.damage.old_skill_damage for case in cases])
        truth = damage > COLLISION_THRESHOLD
        useful = np.asarray([edit.improvement > 1e-4 for edit in cache.values()])

        causal_r = _correlation(score_map["causal"][test], damage[test])
        causal_auc = _auroc(score_map["causal"][test], truth[test])
        shuffled_auc = _auroc(score_map["causal_shuffled"][test], truth[test])
        reversed_auc = _auroc(score_map["causal_reversed"][test], truth[test])
        static_aucs = {
            name: _auroc(score_map[name][test], truth[test]) for name in STATIC
        }
        best_static_name = max(static_aucs, key=static_aucs.get)

        print(
            f"factor={factor:.1f} useful={np.mean(useful):.6f} "
            f"collisions={np.mean(truth[test]):.6f} "
            f"causal_r={causal_r:.6f} causal_auc={causal_auc:.6f} "
            f"best_static={best_static_name}:{static_aucs[best_static_name]:.6f} "
            f"shuffled={shuffled_auc:.6f} reversed={reversed_auc:.6f}"
        )


if __name__ == "__main__":
    main()
