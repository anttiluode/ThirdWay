"""Gate T5: a nonlinear collision radar for time-ordered low-rank edits.

T4 was exact and linear. T5 keeps the same causal question but puts it inside a
state-dependent tanh recurrent system. Two finite rank-1 operator edits happen
at different times. Each edit has a measurable isolated effect. The collision
target is how much the first edit changes the *marginal action* of the second.

Four pre-commit signals are compared:

1. parameter cosine between the two rank-1 edits;
2. a Fisher-ish activation proxy: cosine between their isolated state effects;
3. cosine between their local one-step Jacobian changes;
4. propagated causal susceptibility: propagate the first isolated effect through
   the unedited recurrent Jacobians, then ask how strongly that transported
   perturbation enters the marginal Jacobian of the second edit.

Thresholds are fit only on sources 0..7 and evaluated on held-out sources 8..15.
The gate is deliberately a bridge, not a general continual-learning claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Case:
    gap: int
    source: int
    second_site: int
    kind: str
    isolated_i: float
    isolated_j: float
    collision: float
    parameter_score: float
    fisher_score: float
    jacobian_score: float
    causal_score: float


@dataclass(frozen=True)
class ClassifierMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float


@dataclass(frozen=True)
class Summary:
    cases: int
    train_cases: int
    test_cases: int
    collision_threshold: float
    min_isolated_effect: float
    parameter_correlation: float
    fisher_correlation: float
    jacobian_correlation: float
    causal_correlation: float
    parameter_test_accuracy: float
    parameter_test_precision: float
    parameter_test_recall: float
    fisher_test_accuracy: float
    fisher_test_precision: float
    fisher_test_recall: float
    jacobian_test_accuracy: float
    jacobian_test_precision: float
    jacobian_test_recall: float
    causal_test_accuracy: float
    causal_test_precision: float
    causal_test_recall: float
    causal_threshold: float


def shift_operator(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        a[(i + 1) % n, i] = 1.0
    return a


def base_operator(n: int) -> np.ndarray:
    return 0.85 * shift_operator(n) + 0.08 * np.eye(n)


def drive(t: int, n: int) -> np.ndarray:
    b = 0.04 * np.sin(0.4 * t + 0.17 * np.arange(n))
    b = np.asarray(b, dtype=float)
    b[(3 * t + 5) % n] += 0.35
    return b


def initial_state(n: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return 0.30 * np.sin(phase)


def edit_matrix(n: int, site: int, strength: float) -> np.ndarray:
    e = np.zeros(n, dtype=float)
    e[site % n] = 1.0
    return strength * np.outer(e, e)


def simulate(
    *,
    n: int,
    total_steps: int,
    tau_i: int,
    tau_j: int,
    site_i: int,
    site_j: int,
    strength: float,
    use_i: bool,
    use_j: bool,
) -> tuple[np.ndarray, np.ndarray]:
    a = base_operator(n)
    ei = edit_matrix(n, site_i, strength)
    ej = edit_matrix(n, site_j, strength)
    x = initial_state(n)
    trajectory = [x.copy()]

    for t in range(total_steps):
        at = a
        if use_i and t == tau_i:
            at = at + ei
        if use_j and t == tau_j:
            at = at + ej
        x = np.tanh(at @ x + drive(t, n))
        trajectory.append(x.copy())

    return np.asarray(trajectory), a


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-15:
        return 0.0
    return float(abs(np.sum(a * b)) / den)


def step_jacobian(a: np.ndarray, next_state: np.ndarray) -> np.ndarray:
    d = 1.0 - np.asarray(next_state) ** 2
    return d[:, None] * a


def event_jacobian_change(
    *,
    a: np.ndarray,
    x: np.ndarray,
    t: int,
    site: int,
    strength: float,
) -> np.ndarray:
    e = edit_matrix(a.shape[0], site, strength)
    z0 = a @ x + drive(t, a.shape[0])
    z1 = (a + e) @ x + drive(t, a.shape[0])
    y0 = np.tanh(z0)
    y1 = np.tanh(z1)
    j0 = (1.0 - y0**2)[:, None] * a
    j1 = (1.0 - y1**2)[:, None] * (a + e)
    return j1 - j0


def measure_case(
    *,
    gap: int,
    source: int,
    second_site: int,
    kind: str,
    n: int = 32,
    strength: float = 0.8,
) -> Case:
    tau_i = 2
    tau_j = tau_i + gap + 1
    total_steps = tau_j + 3

    base, a = simulate(
        n=n,
        total_steps=total_steps,
        tau_i=tau_i,
        tau_j=tau_j,
        site_i=source,
        site_j=second_site,
        strength=strength,
        use_i=False,
        use_j=False,
    )
    only_i, _ = simulate(
        n=n,
        total_steps=total_steps,
        tau_i=tau_i,
        tau_j=tau_j,
        site_i=source,
        site_j=second_site,
        strength=strength,
        use_i=True,
        use_j=False,
    )
    only_j, _ = simulate(
        n=n,
        total_steps=total_steps,
        tau_i=tau_i,
        tau_j=tau_j,
        site_i=source,
        site_j=second_site,
        strength=strength,
        use_i=False,
        use_j=True,
    )
    both, _ = simulate(
        n=n,
        total_steps=total_steps,
        tau_i=tau_i,
        tau_j=tau_j,
        site_i=source,
        site_j=second_site,
        strength=strength,
        use_i=True,
        use_j=True,
    )

    delta_i = only_i[tau_i + 1] - base[tau_i + 1]
    delta_j = only_j[tau_j + 1] - base[tau_j + 1]

    marginal_j_base = only_j[tau_j + 1] - base[tau_j + 1]
    marginal_j_after_i = both[tau_j + 1] - only_i[tau_j + 1]
    collision = float(np.linalg.norm(marginal_j_after_i - marginal_j_base))

    edit_i = edit_matrix(n, source, strength)
    edit_j = edit_matrix(n, second_site, strength)
    parameter_score = _cosine(edit_i, edit_j)
    fisher_score = _cosine(delta_i, delta_j)

    ji = event_jacobian_change(
        a=a,
        x=base[tau_i],
        t=tau_i,
        site=source,
        strength=strength,
    )
    jj = event_jacobian_change(
        a=a,
        x=base[tau_j],
        t=tau_j,
        site=second_site,
        strength=strength,
    )
    jacobian_score = _cosine(ji, jj)

    transported = delta_i.copy()
    for t in range(tau_i + 1, tau_j):
        transported = step_jacobian(a, base[t + 1]) @ transported

    causal_score = float(np.linalg.norm(jj @ transported))

    return Case(
        gap=int(gap),
        source=int(source),
        second_site=int(second_site),
        kind=kind,
        isolated_i=float(np.linalg.norm(delta_i)),
        isolated_j=float(np.linalg.norm(delta_j)),
        collision=collision,
        parameter_score=parameter_score,
        fisher_score=fisher_score,
        jacobian_score=jacobian_score,
        causal_score=causal_score,
    )


def make_cases(
    *,
    gaps: Iterable[int] = (2, 5, 8, 11),
    sources: Iterable[int] = range(16),
    n: int = 32,
    strength: float = 0.8,
) -> list[Case]:
    cases: list[Case] = []
    for gap in gaps:
        for source in sources:
            options = (
                ("transported", (source + gap) % n),
                ("static_confound", source),
                ("neutral", (source + gap + 1) % n),
            )
            for kind, second_site in options:
                cases.append(
                    measure_case(
                        gap=int(gap),
                        source=int(source),
                        second_site=int(second_site),
                        kind=kind,
                        n=n,
                        strength=strength,
                    )
                )
    return cases


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _choose_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    values = np.unique(scores)
    if values.size == 1:
        candidates = np.asarray([-np.inf, np.inf])
    else:
        mids = (values[:-1] + values[1:]) / 2.0
        candidates = np.concatenate(([-np.inf], mids, [np.inf]))

    best_key: tuple[float, float] | None = None
    best_threshold = float("inf")
    prevalence = float(np.mean(truth))

    for threshold in candidates:
        pred = scores > threshold
        tp = int(np.sum(pred & truth))
        tn = int(np.sum(~pred & ~truth))
        fp = int(np.sum(pred & ~truth))
        fn = int(np.sum(~pred & truth))
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        balanced = 0.5 * (tpr + tnr)
        prevalence_match = -abs(float(np.mean(pred)) - prevalence)
        key = (balanced, prevalence_match)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold


def _metrics(
    train_scores: np.ndarray,
    train_truth: np.ndarray,
    test_scores: np.ndarray,
    test_truth: np.ndarray,
) -> ClassifierMetrics:
    threshold = _choose_threshold(train_scores, train_truth)
    pred = test_scores > threshold
    tp = int(np.sum(pred & test_truth))
    fp = int(np.sum(pred & ~test_truth))
    fn = int(np.sum(~pred & test_truth))
    accuracy = float(np.mean(pred == test_truth))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return ClassifierMetrics(threshold, accuracy, precision, recall)


def run(*, collision_threshold: float = 0.005) -> Summary:
    cases = make_cases()
    train = np.asarray([c.source < 8 for c in cases], dtype=bool)
    test = ~train
    collision = np.asarray([c.collision for c in cases], dtype=float)
    truth = collision > collision_threshold

    scores = {
        "parameter": np.asarray([c.parameter_score for c in cases]),
        "fisher": np.asarray([c.fisher_score for c in cases]),
        "jacobian": np.asarray([c.jacobian_score for c in cases]),
        "causal": np.asarray([c.causal_score for c in cases]),
    }
    fitted = {
        name: _metrics(score[train], truth[train], score[test], truth[test])
        for name, score in scores.items()
    }

    return Summary(
        cases=len(cases),
        train_cases=int(np.sum(train)),
        test_cases=int(np.sum(test)),
        collision_threshold=float(collision_threshold),
        min_isolated_effect=float(
            min(min(c.isolated_i, c.isolated_j) for c in cases)
        ),
        parameter_correlation=_correlation(scores["parameter"], collision),
        fisher_correlation=_correlation(scores["fisher"], collision),
        jacobian_correlation=_correlation(scores["jacobian"], collision),
        causal_correlation=_correlation(scores["causal"], collision),
        parameter_test_accuracy=fitted["parameter"].accuracy,
        parameter_test_precision=fitted["parameter"].precision,
        parameter_test_recall=fitted["parameter"].recall,
        fisher_test_accuracy=fitted["fisher"].accuracy,
        fisher_test_precision=fitted["fisher"].precision,
        fisher_test_recall=fitted["fisher"].recall,
        jacobian_test_accuracy=fitted["jacobian"].accuracy,
        jacobian_test_precision=fitted["jacobian"].precision,
        jacobian_test_recall=fitted["jacobian"].recall,
        causal_test_accuracy=fitted["causal"].accuracy,
        causal_test_precision=fitted["causal"].precision,
        causal_test_recall=fitted["causal"].recall,
        causal_threshold=fitted["causal"].threshold,
    )


def main() -> None:
    s = run()
    print(
        f"cases={s.cases} train={s.train_cases} test={s.test_cases} "
        f"min_isolated_effect={s.min_isolated_effect:.6f}"
    )
    print(
        "correlation with finite nonlinear collision: "
        f"parameter={s.parameter_correlation:.6f} "
        f"fisherish={s.fisher_correlation:.6f} "
        f"jacobian={s.jacobian_correlation:.6f} "
        f"causal={s.causal_correlation:.6f}"
    )
    for name in ("parameter", "fisher", "jacobian", "causal"):
        print(
            f"{name:9s} test_accuracy={getattr(s, name + '_test_accuracy'):.6f} "
            f"precision={getattr(s, name + '_test_precision'):.6f} "
            f"recall={getattr(s, name + '_test_recall'):.6f}"
        )

    passed = (
        s.cases == 192
        and s.min_isolated_effect > 1e-3
        and s.causal_correlation > 0.98
        and s.causal_test_accuracy > 0.95
        and s.causal_test_precision > 0.95
        and s.causal_test_recall > 0.95
        and s.parameter_test_recall < 0.10
        and s.fisher_test_recall < 0.10
        and s.jacobian_test_recall < 0.10
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
