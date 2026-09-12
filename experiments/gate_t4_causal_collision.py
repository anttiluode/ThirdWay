"""Gate T4: compatibility of learning is causal, not merely geometric.

Kompressori found that experience-induced response-operator changes can be
low-rank and that overlapping updates can interfere. CausalHorizon gives the
exact directed interaction coordinate for time-ordered low-rank events:

    Omega_ji = V_j^T Phi(tau_j, tau_i + 1) U_i.

T4 builds the smallest exact laboratory where static overlap is actively
misleading. The base dynamics is a cyclic transport operator. Two local rank-1
edits may be at different coordinates but collide after transport; two edits at
the same coordinate may be causally orthogonal because the first edit has moved
away before the second occurs.

The point is not the ring. The point is the guard variable: if slow learning is
an operator edit, compatibility should be tested after propagation through the
current substrate, not only in raw parameter coordinates.
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
    static_overlap: float
    causal_overlap: float
    nonadditivity: float


@dataclass(frozen=True)
class Summary:
    cases: int
    collision_cases: int
    causal_accuracy: float
    static_accuracy: float
    causal_precision: float
    causal_recall: float
    static_precision: float
    static_recall: float
    max_exact_error: float
    harmless_static_rejected: float
    transported_collision_missed_by_static: float


def shift_operator(n: int, step: int = 1) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        a[(i + step) % n, i] = 1.0
    return a


def ordered_product(mats: list[np.ndarray]) -> np.ndarray:
    out = np.eye(mats[0].shape[0], dtype=float)
    for a in mats:
        out = a @ out
    return out


def _basis(n: int, index: int) -> np.ndarray:
    e = np.zeros(n, dtype=float)
    e[index % n] = 1.0
    return e


def measure_pair(
    *,
    n: int,
    total_steps: int,
    tau_i: int,
    tau_j: int,
    site_i: int,
    site_j: int,
    strength_i: float = 0.2,
    strength_j: float = 0.2,
) -> tuple[float, float, float, float]:
    if not (0 <= tau_i < tau_j < total_steps):
        raise ValueError("event times must satisfy 0 <= tau_i < tau_j < T")

    a = shift_operator(n)
    u_i = _basis(n, site_i)
    v_i = u_i.copy()
    u_j = _basis(n, site_j)
    v_j = u_j.copy()

    base_mats = [a.copy() for _ in range(total_steps)]
    phi = ordered_product(base_mats)

    mats_i = [x.copy() for x in base_mats]
    mats_i[tau_i] = a + strength_i * np.outer(u_i, v_i)
    phi_i = ordered_product(mats_i)

    mats_j = [x.copy() for x in base_mats]
    mats_j[tau_j] = a + strength_j * np.outer(u_j, v_j)
    phi_j = ordered_product(mats_j)

    mats_ij = [x.copy() for x in base_mats]
    mats_ij[tau_i] = mats_i[tau_i]
    mats_ij[tau_j] = mats_j[tau_j]
    phi_ij = ordered_product(mats_ij)

    delta_i = phi_i - phi
    delta_j = phi_j - phi
    delta_ij = phi_ij - phi
    cross = delta_ij - delta_i - delta_j
    nonadditivity = float(np.linalg.norm(cross, "fro"))

    between_steps = tau_j - tau_i - 1
    between = np.linalg.matrix_power(a, between_steps)
    omega = float(abs(v_j @ (between @ u_i)))
    static = float(abs(v_j @ u_i))

    # For this orthogonal rank-1 construction all outer factors have unit norm,
    # so the exact cross-term Frobenius norm is |m_j * Omega * m_i|.
    exact_prediction = abs(strength_i * strength_j) * omega
    exact_error = abs(nonadditivity - exact_prediction)
    return static, omega, nonadditivity, exact_error


def make_cases(
    *,
    n: int = 32,
    gaps: Iterable[int] = (2, 5, 8, 11),
    sources: Iterable[int] = range(16),
    strength: float = 0.2,
) -> list[Case]:
    """Balanced hard set: transported collision, static confound, neutral control."""
    cases: list[Case] = []
    tau_i = 2

    for gap in gaps:
        tau_j = tau_i + gap + 1
        total_steps = tau_j + 5
        for source in sources:
            # 1) True collision after transport, despite zero static overlap.
            causal_site = (source + gap) % n
            # 2) Same raw coordinate, but transport has moved the first edit away.
            static_site = source
            # 3) Neither static nor propagated overlap.
            control_site = (source + gap + 1) % n

            for kind, second_site in (
                ("transported_collision", causal_site),
                ("static_confound", static_site),
                ("neutral", control_site),
            ):
                static, omega, nonadd, _ = measure_pair(
                    n=n,
                    total_steps=total_steps,
                    tau_i=tau_i,
                    tau_j=tau_j,
                    site_i=source,
                    site_j=second_site,
                    strength_i=strength,
                    strength_j=strength,
                )
                cases.append(
                    Case(
                        gap=int(gap),
                        source=int(source),
                        second_site=int(second_site),
                        kind=kind,
                        static_overlap=static,
                        causal_overlap=omega,
                        nonadditivity=nonadd,
                    )
                )
    return cases


def _classification_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    accuracy = float(np.mean(truth == prediction))
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return accuracy, precision, recall


def summarize(cases: list[Case]) -> Summary:
    truth = np.asarray([c.nonadditivity > 1e-12 for c in cases])
    causal_pred = np.asarray([c.causal_overlap > 0.5 for c in cases])
    static_pred = np.asarray([c.static_overlap > 0.5 for c in cases])

    ca, cp, cr = _classification_metrics(truth, causal_pred)
    sa, sp, sr = _classification_metrics(truth, static_pred)

    # Recompute exact identity error explicitly for every case.
    exact_errors = []
    for c in cases:
        predicted = 0.04 * c.causal_overlap
        exact_errors.append(abs(c.nonadditivity - predicted))

    static_confounds = [c for c in cases if c.kind == "static_confound"]
    transported = [c for c in cases if c.kind == "transported_collision"]
    harmless_static_rejected = float(
        np.mean([c.static_overlap > 0.5 and c.nonadditivity <= 1e-12 for c in static_confounds])
    )
    transported_missed = float(
        np.mean([c.static_overlap <= 0.5 and c.nonadditivity > 1e-12 for c in transported])
    )

    return Summary(
        cases=len(cases),
        collision_cases=int(np.sum(truth)),
        causal_accuracy=ca,
        static_accuracy=sa,
        causal_precision=cp,
        causal_recall=cr,
        static_precision=sp,
        static_recall=sr,
        max_exact_error=float(max(exact_errors)),
        harmless_static_rejected=harmless_static_rejected,
        transported_collision_missed_by_static=transported_missed,
    )


def run() -> Summary:
    return summarize(make_cases())


def main() -> None:
    s = run()
    print(f"cases={s.cases} collisions={s.collision_cases}")
    print(
        "causal "
        f"accuracy={s.causal_accuracy:.6f} precision={s.causal_precision:.6f} "
        f"recall={s.causal_recall:.6f}"
    )
    print(
        "static "
        f"accuracy={s.static_accuracy:.6f} precision={s.static_precision:.6f} "
        f"recall={s.static_recall:.6f}"
    )
    print(f"max exact cross-term error={s.max_exact_error:.3e}")
    print(f"harmless static-overlap pairs rejected={s.harmless_static_rejected:.6f}")
    print(
        "transported collisions missed by static overlap="
        f"{s.transported_collision_missed_by_static:.6f}"
    )

    passed = (
        s.causal_accuracy == 1.0
        and s.causal_precision == 1.0
        and s.causal_recall == 1.0
        and s.static_recall == 0.0
        and s.static_precision == 0.0
        and s.static_accuracy < 0.50
        and s.max_exact_error < 1e-12
        and s.harmless_static_rejected == 1.0
        and s.transported_collision_missed_by_static == 1.0
    )
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
