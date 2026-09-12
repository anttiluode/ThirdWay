"""Live T6 continual-learning demonstration.

Run:
    python demo_t6_continual_learning.py

The guarded and accept-all learners receive the exact same candidate matrices.
The guarded learner may reject a permanent edit using the training-calibrated
directional dynamic-risk score.  No displayed value is hard-coded.
"""

from experiments.t6_continual_learning import continual_demo


SKILL_NAME = {0: "A/recent", 1: "B/delayed", 2: "C/order"}


def _fmt(scores: tuple[float, float, float]) -> str:
    return f"A={scores[0]:.3f} B={scores[1]:.3f} C={scores[2]:.3f}"


def main() -> None:
    demo = continual_demo(seed=0)
    print("=== ThirdWay T6: continual learning on one shared recurrent network ===")
    print(f"risk threshold (frozen T6 train split): {demo.risk_threshold:.6f}")
    print(f"BASE             {_fmt(demo.base_accuracy)}")
    print()

    for event in demo.events:
        verdict = "ACCEPT" if event.accepted else "REJECT"
        print(
            f"{event.index:02d} learn {SKILL_NAME[event.skill]:9s} "
            f"candidate={event.candidate_seed:3d} "
            f"target_gain(all)={event.target_improvement_accept_all:+.5f} "
            f"target_gain(guard)={event.target_improvement_guarded:+.5f} "
            f"risk={event.max_risk:.5f} {verdict:6s} {event.reason}"
        )
        print(f"   guarded       {_fmt(event.guarded_accuracy)}")
        print(f"   accept-all    {_fmt(event.accept_all_accuracy)}")

    print()
    print(
        f"FINAL guarded    {_fmt(demo.final_accuracy_guarded)} "
        f"mean={demo.final_mean_accuracy_guarded:.3f} "
        f"min={demo.final_min_accuracy_guarded:.3f} "
        f"switch={demo.final_switch_penalty_guarded:+.5f}"
    )
    print(
        f"FINAL accept-all {_fmt(demo.final_accuracy_accept_all)} "
        f"mean={demo.final_mean_accuracy_accept_all:.3f} "
        f"min={demo.final_min_accuracy_accept_all:.3f} "
        f"switch={demo.final_switch_penalty_accept_all:+.5f}"
    )
    print(
        "old-at-final (A,B) mean: "
        f"guarded={demo.final_old_skill_accuracy_guarded:.3f} "
        f"accept-all={demo.final_old_skill_accuracy_accept_all:.3f}"
    )
    print(
        f"decisions: guarded accepted={demo.accepted} rejected={demo.rejected}; "
        f"accept-all accepted={demo.accept_all_accepted}"
    )


if __name__ == "__main__":
    main()
