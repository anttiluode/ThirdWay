"""Gate T6: persistent skill compatibility in a shared recurrent model.

One deterministic 24-state tanh recurrent substrate is shared by three temporal
skills. Candidate learning events are persistent rank-1 edits to the recurrent
matrix. Candidate generation is target-only. Compatibility is evaluated only
after useful candidates exist.

T6 compares static parameter-space signals with a directional causal score: the
candidate edit is linearized along trajectories of a retained skill and its
state perturbation is propagated through the ordered recurrent Jacobians. The
world, identity split, collision threshold, predictors, and pass criteria were
frozen before held-out predictor results were observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    hidden_dim: int = 24
    input_dim: int = 6
    sequence_length: int = 16
    noise_scale: float = 0.20
    context_scale: float = 0.25
    recent_cue_scale: float = 0.45
    delayed_cue_scale: float = 0.65
    relation_cue_scale: float = 0.90
    spectral_radius: float = 1.10
    input_scale: float = 0.55
    bias_scale: float = 0.02
    ridge: float = 1.0
    calibration_seed: int = 7
    calibration_examples_per_skill: int = 256


CONFIG = WorldConfig()
AMPLITUDES = (-0.30, -0.20, -0.12, -0.07, 0.07, 0.12, 0.20, 0.30)
DIRECTIONS_PER_CANDIDATE = 8
COLLISION_THRESHOLD = 0.02
PREDICTOR_EXAMPLES = 48


@dataclass(frozen=True)
class Dataset:
    inputs: np.ndarray
    targets: np.ndarray
    skills: np.ndarray

    def for_skill(self, skill: int) -> "Dataset":
        mask = self.skills == int(skill)
        return Dataset(
            inputs=self.inputs[mask].copy(),
            targets=self.targets[mask].copy(),
            skills=self.skills[mask].copy(),
        )


@dataclass(frozen=True)
class Model:
    W: np.ndarray
    B: np.ndarray
    b: np.ndarray
    C: np.ndarray


@dataclass(frozen=True)
class CandidateEdit:
    skill: int
    seed: int
    delta_w: np.ndarray
    base_loss: float
    edited_loss: float

    @property
    def improvement(self) -> float:
        return float(self.base_loss - self.edited_loss)

    @property
    def identity(self) -> tuple[int, int]:
        return (int(self.skill), int(self.seed))


@dataclass(frozen=True)
class Damage:
    old_skill_damage: float
    leakage_delta: float
    switch_cost_delta: float


@dataclass(frozen=True)
class PairCase:
    retained: CandidateEdit
    candidate: CandidateEdit
    split: str
    damage: Damage

    @property
    def candidate_id(self) -> tuple[int, int]:
        return self.candidate.identity

    @property
    def retained_id(self) -> tuple[int, int]:
        return self.retained.identity


@dataclass(frozen=True)
class PredictorMetrics:
    damage_correlation: float
    test_auroc: float
    test_balanced_accuracy: float
    threshold: float


@dataclass(frozen=True)
class Summary:
    cases: int
    train_cases: int
    test_cases: int
    all_candidates_individually_useful: bool
    heldout_collision_fraction: float
    heldout_safe_fraction: float
    predictors: dict[str, PredictorMetrics]
    skill_pair_families_with_positive_causal_advantage: int


def make_dataset(
    seed: int,
    examples_per_skill: int,
    config: WorldConfig = CONFIG,
) -> Dataset:
    """Build three matched-input temporal tasks with explicit context."""
    if examples_per_skill <= 0:
        raise ValueError("examples_per_skill must be positive")

    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    skills: list[int] = []

    for skill in range(3):
        for _ in range(examples_per_skill):
            x = rng.normal(
                0.0,
                config.noise_scale,
                size=(config.sequence_length, config.input_dim),
            )
            x[:, 3 + skill] += config.context_scale

            if skill == 0:
                target = float(rng.choice((-1.0, 1.0)))
                x[13, 0] += config.recent_cue_scale * target
            elif skill == 1:
                target = float(rng.choice((-1.0, 1.0)))
                x[8, 0] += config.delayed_cue_scale * target
            else:
                if rng.random() < 0.5:
                    t_a, t_b = 4, 11
                    target = 1.0
                else:
                    t_a, t_b = 11, 4
                    target = -1.0
                x[t_a, 1] += config.relation_cue_scale
                x[t_b, 2] += config.relation_cue_scale

            inputs.append(x)
            targets.append(target)
            skills.append(skill)

    return Dataset(
        inputs=np.asarray(inputs, dtype=float),
        targets=np.asarray(targets, dtype=float),
        skills=np.asarray(skills, dtype=int),
    )


def _make_body(seed: int, config: WorldConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.normal(
        0.0,
        1.0 / np.sqrt(config.hidden_dim),
        size=(config.hidden_dim, config.hidden_dim),
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    if radius <= 1e-12:
        raise RuntimeError("degenerate recurrent initialization")
    w = w * (config.spectral_radius / radius)
    b_in = rng.normal(
        0.0,
        config.input_scale,
        size=(config.hidden_dim, config.input_dim),
    )
    bias = rng.normal(0.0, config.bias_scale, size=config.hidden_dim)
    return w, b_in, bias


def hidden_trajectory(model: Model, sequence: np.ndarray) -> np.ndarray:
    h = np.zeros(model.W.shape[0], dtype=float)
    states = [h.copy()]
    for x_t in np.asarray(sequence, dtype=float):
        h = np.tanh(model.W @ h + model.B @ x_t + model.b)
        states.append(h.copy())
    return np.asarray(states)


def batch_hidden_trajectories(
    model: Model,
    sequences: np.ndarray,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Run the exact same recurrence for a batch of independent sequences."""
    x = np.asarray(sequences, dtype=float)
    if x.ndim != 3:
        raise ValueError("sequences must have shape (batch, time, input_dim)")
    if initial_states is None:
        h = np.zeros((x.shape[0], model.W.shape[0]), dtype=float)
    else:
        h = np.asarray(initial_states, dtype=float).copy()
        if h.shape != (x.shape[0], model.W.shape[0]):
            raise ValueError("initial_states must have shape (batch, hidden_dim)")
    states = [h.copy()]
    for t in range(x.shape[1]):
        h = np.tanh(h @ model.W.T + x[:, t, :] @ model.B.T + model.b)
        states.append(h.copy())
    return np.stack(states, axis=1)


def final_hidden(model: Model, sequence: np.ndarray) -> np.ndarray:
    return hidden_trajectory(model, sequence)[-1]


def batch_final_hidden(
    model: Model,
    sequences: np.ndarray,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    return batch_hidden_trajectories(model, sequences, initial_states)[:, -1, :]


def _fit_shared_readout(
    w: np.ndarray,
    b_in: np.ndarray,
    bias: np.ndarray,
    config: WorldConfig,
) -> np.ndarray:
    calibration = make_dataset(
        seed=config.calibration_seed,
        examples_per_skill=config.calibration_examples_per_skill,
        config=config,
    )
    temporary = Model(
        W=w,
        B=b_in,
        b=bias,
        C=np.zeros(config.hidden_dim, dtype=float),
    )
    features = batch_final_hidden(temporary, calibration.inputs)
    gram = features.T @ features + config.ridge * np.eye(config.hidden_dim)
    return np.linalg.solve(gram, features.T @ calibration.targets)


def make_seed_model(seed: int, config: WorldConfig = CONFIG) -> Model:
    w, b_in, bias = _make_body(seed, config)
    readout = _fit_shared_readout(w, b_in, bias, config)
    return Model(W=w, B=b_in, b=bias, C=readout)


def predict(model: Model, sequence: np.ndarray) -> float:
    return float(model.C @ final_hidden(model, sequence))


def batch_predict(
    model: Model,
    sequences: np.ndarray,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    return batch_final_hidden(model, sequences, initial_states) @ model.C


def mean_squared_loss(model: Model, dataset: Dataset, skill: int) -> float:
    subset = dataset.for_skill(skill)
    if subset.targets.size == 0:
        raise ValueError(f"dataset has no examples for skill {skill}")
    outputs = batch_predict(model, subset.inputs)
    return float(np.mean((outputs - subset.targets) ** 2))


def evaluate_skill(model: Model, dataset: Dataset, skill: int) -> float:
    subset = dataset.for_skill(skill)
    if subset.targets.size == 0:
        raise ValueError(f"dataset has no examples for skill {skill}")
    outputs = batch_predict(model, subset.inputs)
    guessed = np.where(outputs >= 0.0, 1.0, -1.0)
    return float(np.mean(guessed == subset.targets))


def evaluate_all(model: Model, dataset: Dataset) -> np.ndarray:
    return np.asarray([evaluate_skill(model, dataset, skill) for skill in range(3)])


def apply_edit(model: Model, edit: CandidateEdit) -> Model:
    return Model(
        W=model.W + edit.delta_w,
        B=model.B,
        b=model.b,
        C=model.C,
    )


def generate_candidates(
    model: Model,
    dataset: Dataset,
    skill: int,
    candidate_seeds: Iterable[int],
    *,
    min_improvement: float = 1e-4,
) -> list[CandidateEdit]:
    """Generate useful rank-1 edits using target-skill utility only."""
    target = dataset.for_skill(skill)
    base_loss = mean_squared_loss(model, target, skill)
    hidden_dim = model.W.shape[0]
    edits: list[CandidateEdit] = []

    for candidate_seed in candidate_seeds:
        rng = np.random.default_rng(10_000 * int(skill) + int(candidate_seed))
        best_loss = base_loss
        best_delta: np.ndarray | None = None

        for _ in range(DIRECTIONS_PER_CANDIDATE):
            u = rng.normal(size=hidden_dim)
            v = rng.normal(size=hidden_dim)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            direction = np.outer(u, v)

            for amplitude in AMPLITUDES:
                delta = float(amplitude) * direction
                trial = Model(
                    W=model.W + delta,
                    B=model.B,
                    b=model.b,
                    C=model.C,
                )
                trial_loss = mean_squared_loss(trial, target, skill)
                if trial_loss < best_loss:
                    best_loss = trial_loss
                    best_delta = delta.copy()

        if best_delta is not None and best_loss < base_loss - min_improvement:
            edits.append(
                CandidateEdit(
                    skill=int(skill),
                    seed=int(candidate_seed),
                    delta_w=best_delta,
                    base_loss=float(base_loss),
                    edited_loss=float(best_loss),
                )
            )

    return edits


def _covariance_signature(states: np.ndarray) -> np.ndarray:
    centered = np.asarray(states, dtype=float) - np.mean(states, axis=0, keepdims=True)
    return centered.T @ centered / max(1, centered.shape[0])


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-15:
        return 0.0
    return float(abs(np.sum(a * b)) / den)


def _representation_overlap(
    model: Model,
    dataset: Dataset,
    first_skill: int,
    second_skill: int,
    *,
    examples: int = 48,
) -> float:
    first = dataset.for_skill(first_skill)
    second = dataset.for_skill(second_skill)
    n = min(examples, first.targets.size, second.targets.size)
    h_first = batch_final_hidden(model, first.inputs[:n])
    h_second = batch_final_hidden(model, second.inputs[:n])
    return _cosine_matrix(_covariance_signature(h_first), _covariance_signature(h_second))


def _switch_penalty(
    model: Model,
    dataset: Dataset,
    from_skill: int,
    to_skill: int,
    *,
    examples: int = 24,
) -> float:
    """Measure context-switch transient while recurrent state is carried across episodes."""
    before = dataset.for_skill(from_skill)
    after = dataset.for_skill(to_skill)
    n = min(examples, before.targets.size, after.targets.size)
    carried_state = batch_final_hidden(model, before.inputs[:n])
    carried_output = batch_predict(model, after.inputs[:n], carried_state)
    reset_output = batch_predict(model, after.inputs[:n])
    targets = after.targets[:n]
    carried_loss = (carried_output - targets) ** 2
    reset_loss = (reset_output - targets) ** 2
    return float(np.mean(carried_loss - reset_loss))


def measure_persistent_damage(
    base: Model,
    retained: CandidateEdit,
    candidate: CandidateEdit,
    dataset: Dataset,
) -> Damage:
    """Measure what permanently adding candidate does after retained is installed."""
    if retained.skill == candidate.skill:
        raise ValueError("T6 pair battery compares different skills")

    retained_model = apply_edit(base, retained)
    both_model = apply_edit(retained_model, candidate)

    old_before = mean_squared_loss(retained_model, dataset, retained.skill)
    old_after = mean_squared_loss(both_model, dataset, retained.skill)
    old_damage = float(old_after - old_before)

    overlap_before = _representation_overlap(
        retained_model, dataset, retained.skill, candidate.skill
    )
    overlap_after = _representation_overlap(
        both_model, dataset, retained.skill, candidate.skill
    )
    leakage_delta = float(overlap_after - overlap_before)

    switch_before = _switch_penalty(
        retained_model, dataset, candidate.skill, retained.skill
    )
    switch_after = _switch_penalty(
        both_model, dataset, candidate.skill, retained.skill
    )
    switch_cost_delta = float(switch_after - switch_before)

    return Damage(
        old_skill_damage=old_damage,
        leakage_delta=leakage_delta,
        switch_cost_delta=switch_cost_delta,
    )


def _paired_seed(candidate_seed: int) -> int:
    """Pair within a 16-seed split so no edit identity crosses train/test."""
    start = 0 if candidate_seed < 16 else 16
    return start + ((candidate_seed - start + 7) % 16)


@lru_cache(maxsize=1)
def _default_pair_battery() -> tuple[PairCase, ...]:
    base = make_seed_model(seed=23)
    proposal_data = make_dataset(seed=41, examples_per_skill=128)
    evaluation_data = make_dataset(seed=53, examples_per_skill=128)

    banks: dict[int, dict[int, CandidateEdit]] = {}
    for skill in range(3):
        edits = generate_candidates(
            base,
            proposal_data,
            skill=skill,
            candidate_seeds=range(32),
        )
        banks[skill] = {edit.seed: edit for edit in edits}
        missing = sorted(set(range(32)) - set(banks[skill]))
        if missing:
            raise RuntimeError(
                f"structurally invalid T6 battery: skill {skill} lacks useful edits {missing}"
            )

    cases: list[PairCase] = []
    for candidate_skill in range(3):
        for retained_skill in range(3):
            if retained_skill == candidate_skill:
                continue
            for candidate_seed in range(32):
                retained_seed = _paired_seed(candidate_seed)
                candidate = banks[candidate_skill][candidate_seed]
                retained = banks[retained_skill][retained_seed]
                split = "train" if candidate_seed < 16 else "test"
                damage = measure_persistent_damage(
                    base,
                    retained,
                    candidate,
                    evaluation_data,
                )
                cases.append(
                    PairCase(
                        retained=retained,
                        candidate=candidate,
                        split=split,
                        damage=damage,
                    )
                )

    return tuple(cases)


def build_default_pair_battery() -> list[PairCase]:
    return list(_default_pair_battery())


def _per_example_gradients(
    model: Model,
    dataset: Dataset,
    skill: int,
    *,
    examples: int = PREDICTOR_EXAMPLES,
) -> np.ndarray:
    """Exact BPTT gradients of per-example squared error with respect to W."""
    subset = dataset.for_skill(skill)
    n = min(int(examples), subset.targets.size)
    x = subset.inputs[:n]
    targets = subset.targets[:n]
    states = batch_hidden_trajectories(model, x)
    prediction = states[:, -1, :] @ model.C

    adjoint = 2.0 * (prediction - targets)[:, None] * model.C[None, :]
    gradients = np.zeros((n, model.W.shape[0], model.W.shape[1]), dtype=float)

    for t in range(x.shape[1] - 1, -1, -1):
        local = adjoint * (1.0 - states[:, t + 1, :] ** 2)
        gradients += np.einsum("ni,nj->nij", local, states[:, t, :])
        adjoint = local @ model.W

    return gradients


def fisher_diagonal(
    model: Model,
    dataset: Dataset,
    skill: int,
    *,
    examples: int = PREDICTOR_EXAMPLES,
) -> np.ndarray:
    """Diagonal empirical Fisher-style statistic from squared per-example gradients."""
    gradients = _per_example_gradients(model, dataset, skill, examples=examples)
    return np.mean(gradients**2, axis=0)


def _static_jacobian_signature(
    base: Model,
    edit: CandidateEdit,
    dataset: Dataset,
    *,
    examples: int = PREDICTOR_EXAMPLES,
) -> np.ndarray:
    """Average one-step Jacobian change without time-ordered transport."""
    subset = dataset.for_skill(edit.skill)
    n = min(int(examples), subset.targets.size)
    x = subset.inputs[:n]
    states = batch_hidden_trajectories(base, x)
    edited_w = base.W + edit.delta_w
    signature = np.zeros_like(base.W)
    count = 0

    for t in range(x.shape[1]):
        h_t = states[:, t, :]
        input_t = x[:, t, :]
        z0 = h_t @ base.W.T + input_t @ base.B.T + base.b
        z1 = h_t @ edited_w.T + input_t @ base.B.T + base.b
        y0 = np.tanh(z0)
        y1 = np.tanh(z1)
        d0 = 1.0 - y0**2
        d1 = 1.0 - y1**2
        j0 = d0[:, :, None] * base.W[None, :, :]
        j1 = d1[:, :, None] * edited_w[None, :, :]
        signature += np.sum(j1 - j0, axis=0)
        count += n

    return signature / max(1, count)


def _causal_harm_score(
    current: Model,
    perturbation: np.ndarray,
    dataset: Dataset,
    skill: int,
    *,
    examples: int = PREDICTOR_EXAMPLES,
    shuffled: bool = False,
) -> float:
    """First-order persistent-edit harm on one retained task.

    The candidate acts at every recurrent step. Its induced hidden-state change
    is propagated through the current model's ordered recurrent tangent. The
    shuffled control keeps the same injections but permutes only the Jacobian
    damping sequence used to transport earlier perturbations.
    """
    subset = dataset.for_skill(skill)
    n = min(int(examples), subset.targets.size)
    x = subset.inputs[:n]
    targets = subset.targets[:n]
    states = batch_hidden_trajectories(current, x)
    prediction = states[:, -1, :] @ current.C
    time_steps = x.shape[1]

    damp = 1.0 - states[:, 1:, :] ** 2
    if shuffled:
        order = np.random.default_rng(991).permutation(time_steps)
    else:
        order = np.arange(time_steps)

    delta = np.zeros((n, current.W.shape[0]), dtype=float)
    for t in range(time_steps):
        h_t = states[:, t, :]
        injection = (h_t @ perturbation.T) * damp[:, t, :]
        transported = (delta @ current.W.T) * damp[:, order[t], :]
        delta = transported + injection

    delta_output = delta @ current.C
    first_order_loss_change = 2.0 * (prediction - targets) * delta_output
    return float(np.mean(np.maximum(first_order_loss_change, 0.0)))


def _auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = scores[truth]
    negative = scores[~truth]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = positive[:, None] > negative[None, :]
    equal = positive[:, None] == negative[None, :]
    return float(np.mean(greater) + 0.5 * np.mean(equal))


def _correlation(scores: np.ndarray, target: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=float)
    if scores.size < 2 or np.std(scores) < 1e-15 or np.std(target) < 1e-15:
        return 0.0
    return float(np.corrcoef(scores, target)[0, 1])


def _balanced_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    positive = truth
    negative = ~truth
    tpr = float(np.mean(prediction[positive])) if np.any(positive) else 0.0
    tnr = float(np.mean(~prediction[negative])) if np.any(negative) else 0.0
    return 0.5 * (tpr + tnr)


def _choose_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    values = np.unique(scores)
    if values.size <= 1:
        candidates = np.asarray([-np.inf, np.inf], dtype=float)
    else:
        mids = (values[:-1] + values[1:]) / 2.0
        candidates = np.concatenate(([-np.inf], mids, [np.inf]))

    prevalence = float(np.mean(truth))
    best_key: tuple[float, float] | None = None
    best_threshold = float("inf")
    for threshold in candidates:
        pred = scores > threshold
        balanced = _balanced_accuracy(pred, truth)
        prevalence_match = -abs(float(np.mean(pred)) - prevalence)
        key = (balanced, prevalence_match)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _predictor_metrics(
    scores: np.ndarray,
    damage: np.ndarray,
    truth: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> PredictorMetrics:
    threshold = _choose_threshold(scores[train], truth[train])
    test_prediction = scores[test] > threshold
    return PredictorMetrics(
        damage_correlation=_correlation(scores[test], damage[test]),
        test_auroc=_auroc(scores[test], truth[test]),
        test_balanced_accuracy=_balanced_accuracy(test_prediction, truth[test]),
        threshold=float(threshold),
    )


def _score_battery(
    cases: list[PairCase],
    base: Model,
    data: Dataset,
) -> dict[str, np.ndarray]:
    """Compute the frozen pre-commit signals for every ordered pair."""
    scores: dict[str, list[float]] = {
        "parameter": [],
        "gradient": [],
        "fisher": [],
        "static_jacobian": [],
        "causal": [],
        "causal_shuffled": [],
        "causal_reversed": [],
        "oracle": [],
    }

    retained_cache: dict[tuple[int, int], tuple[Model, np.ndarray, np.ndarray]] = {}
    static_cache: dict[tuple[int, int], np.ndarray] = {}

    def retained_analysis(edit: CandidateEdit) -> tuple[Model, np.ndarray, np.ndarray]:
        if edit.identity not in retained_cache:
            model = apply_edit(base, edit)
            per_example = _per_example_gradients(
                model,
                data,
                edit.skill,
                examples=PREDICTOR_EXAMPLES,
            )
            retained_cache[edit.identity] = (
                model,
                np.mean(per_example, axis=0),
                np.mean(per_example**2, axis=0),
            )
        return retained_cache[edit.identity]

    def static_signature(edit: CandidateEdit) -> np.ndarray:
        if edit.identity not in static_cache:
            static_cache[edit.identity] = _static_jacobian_signature(
                base,
                edit,
                data,
                examples=PREDICTOR_EXAMPLES,
            )
        return static_cache[edit.identity]

    for case in cases:
        retained = case.retained
        candidate = case.candidate
        retained_model, old_gradient, old_fisher = retained_analysis(retained)

        scores["parameter"].append(
            _cosine_matrix(retained.delta_w, candidate.delta_w)
        )
        scores["gradient"].append(
            _cosine_matrix(old_gradient, candidate.delta_w)
        )
        fisher_energy = float(np.sum(old_fisher * candidate.delta_w**2))
        candidate_energy = float(np.sum(candidate.delta_w**2))
        scores["fisher"].append(
            float(np.sqrt(fisher_energy / max(candidate_energy, 1e-15)))
        )
        scores["static_jacobian"].append(
            _cosine_matrix(static_signature(retained), static_signature(candidate))
        )
        scores["causal"].append(
            _causal_harm_score(
                retained_model,
                candidate.delta_w,
                data,
                retained.skill,
                examples=PREDICTOR_EXAMPLES,
                shuffled=False,
            )
        )
        scores["causal_shuffled"].append(
            _causal_harm_score(
                retained_model,
                candidate.delta_w,
                data,
                retained.skill,
                examples=PREDICTOR_EXAMPLES,
                shuffled=True,
            )
        )

        candidate_model = apply_edit(base, candidate)
        scores["causal_reversed"].append(
            _causal_harm_score(
                candidate_model,
                retained.delta_w,
                data,
                candidate.skill,
                examples=PREDICTOR_EXAMPLES,
                shuffled=False,
            )
        )
        scores["oracle"].append(float(case.damage.old_skill_damage))

    return {name: np.asarray(values, dtype=float) for name, values in scores.items()}


@lru_cache(maxsize=1)
def _run_cached() -> Summary:
    cases = build_default_pair_battery()
    base = make_seed_model(seed=23)
    evaluation_data = make_dataset(seed=53, examples_per_skill=128)
    score_map = _score_battery(cases, base, evaluation_data)

    train = np.asarray([case.split == "train" for case in cases], dtype=bool)
    test = ~train
    damage = np.asarray([case.damage.old_skill_damage for case in cases], dtype=float)
    truth = damage > COLLISION_THRESHOLD

    metrics = {
        name: _predictor_metrics(values, damage, truth, train, test)
        for name, values in score_map.items()
    }

    static_names = ("parameter", "gradient", "fisher", "static_jacobian")
    positive_families = 0
    for retained_skill in range(3):
        for candidate_skill in range(3):
            if retained_skill == candidate_skill:
                continue
            family = np.asarray(
                [
                    test[i]
                    and case.retained.skill == retained_skill
                    and case.candidate.skill == candidate_skill
                    for i, case in enumerate(cases)
                ],
                dtype=bool,
            )
            family_truth = truth[family]
            if not np.any(family_truth) or np.all(family_truth):
                continue
            causal_auc = _auroc(score_map["causal"][family], family_truth)
            static_best = max(
                _auroc(score_map[name][family], family_truth) for name in static_names
            )
            if causal_auc > static_best + 1e-12:
                positive_families += 1

    heldout_truth = truth[test]
    all_useful = all(
        case.retained.improvement > 1e-4 and case.candidate.improvement > 1e-4
        for case in cases
    )

    return Summary(
        cases=len(cases),
        train_cases=int(np.sum(train)),
        test_cases=int(np.sum(test)),
        all_candidates_individually_useful=bool(all_useful),
        heldout_collision_fraction=float(np.mean(heldout_truth)),
        heldout_safe_fraction=float(np.mean(~heldout_truth)),
        predictors=metrics,
        skill_pair_families_with_positive_causal_advantage=int(positive_families),
    )


def run() -> Summary:
    return _run_cached()


def _passes_gate(summary: Summary) -> bool:
    causal = summary.predictors["causal"]
    static_names = ("parameter", "gradient", "fisher", "static_jacobian")
    return bool(
        summary.cases == 192
        and summary.test_cases == 96
        and summary.all_candidates_individually_useful
        and summary.heldout_collision_fraction >= 0.20
        and summary.heldout_safe_fraction >= 0.20
        and causal.damage_correlation > 0.70
        and all(
            causal.test_auroc >= summary.predictors[name].test_auroc + 0.10
            for name in static_names
        )
        and summary.skill_pair_families_with_positive_causal_advantage >= 4
        and summary.predictors["oracle"].test_auroc >= causal.test_auroc - 1e-12
        and causal.test_auroc
        >= summary.predictors["causal_shuffled"].test_auroc + 0.10
        and causal.test_auroc
        >= summary.predictors["causal_reversed"].test_auroc + 0.10
    )


def main() -> None:
    summary = run()
    print(
        f"cases={summary.cases} train={summary.train_cases} test={summary.test_cases} "
        f"collision_fraction={summary.heldout_collision_fraction:.6f} "
        f"safe_fraction={summary.heldout_safe_fraction:.6f}"
    )
    for name, metric in summary.predictors.items():
        print(
            f"{name:17s} r={metric.damage_correlation:.6f} "
            f"auroc={metric.test_auroc:.6f} "
            f"balanced_acc={metric.test_balanced_accuracy:.6f} "
            f"threshold={metric.threshold:.6g}"
        )
    print(
        "ordered skill-pair families with causal advantage="
        f"{summary.skill_pair_families_with_positive_causal_advantage}/6"
    )
    passed = _passes_gate(summary)
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
