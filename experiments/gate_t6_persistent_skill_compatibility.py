"""Gate T6: persistent skill compatibility in a shared recurrent model.

The gate starts from one deterministic 24-state recurrent substrate shared by
three temporal skills. Candidate learning events are persistent rank-1 edits to
the recurrent matrix. Candidate generation is deliberately target-only: it may
measure whether an edit helps the skill that proposed it, but it cannot inspect
cross-skill damage before the later compatibility stage.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def batch_hidden_trajectories(model: Model, sequences: np.ndarray) -> np.ndarray:
    """Run the exact same recurrence for a batch of independent sequences."""
    x = np.asarray(sequences, dtype=float)
    if x.ndim != 3:
        raise ValueError("sequences must have shape (batch, time, input_dim)")
    h = np.zeros((x.shape[0], model.W.shape[0]), dtype=float)
    states = [h.copy()]
    for t in range(x.shape[1]):
        h = np.tanh(h @ model.W.T + x[:, t, :] @ model.B.T + model.b)
        states.append(h.copy())
    return np.stack(states, axis=1)


def final_hidden(model: Model, sequence: np.ndarray) -> np.ndarray:
    return hidden_trajectory(model, sequence)[-1]


def batch_final_hidden(model: Model, sequences: np.ndarray) -> np.ndarray:
    return batch_hidden_trajectories(model, sequences)[:, -1, :]


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


def batch_predict(model: Model, sequences: np.ndarray) -> np.ndarray:
    return batch_final_hidden(model, sequences) @ model.C


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
    """Generate useful rank-1 edits using target-skill utility only.

    Each candidate identity owns a fixed bundle of eight random rank-1
    directions. The best direction/amplitude is selected using only a dataset
    slice containing the target skill. Cross-skill performance is inaccessible
    to this function by construction.
    """
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
