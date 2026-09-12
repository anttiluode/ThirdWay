"""Gate T6: persistent skill compatibility in a shared recurrent model.

This first implementation step defines only the deterministic recurrent world
and the three temporal skills. Persistent edits and compatibility scoring are
added by later TDD steps.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def final_hidden(model: Model, sequence: np.ndarray) -> np.ndarray:
    return hidden_trajectory(model, sequence)[-1]


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
    features = np.asarray(
        [final_hidden(temporary, x) for x in calibration.inputs],
        dtype=float,
    )
    gram = features.T @ features + config.ridge * np.eye(config.hidden_dim)
    return np.linalg.solve(gram, features.T @ calibration.targets)


def make_seed_model(seed: int, config: WorldConfig = CONFIG) -> Model:
    w, b_in, bias = _make_body(seed, config)
    readout = _fit_shared_readout(w, b_in, bias, config)
    return Model(W=w, B=b_in, b=bias, C=readout)


def predict(model: Model, sequence: np.ndarray) -> float:
    return float(model.C @ final_hidden(model, sequence))


def evaluate_skill(model: Model, dataset: Dataset, skill: int) -> float:
    subset = dataset.for_skill(skill)
    if subset.targets.size == 0:
        raise ValueError(f"dataset has no examples for skill {skill}")
    outputs = np.asarray([predict(model, x) for x in subset.inputs])
    guessed = np.where(outputs >= 0.0, 1.0, -1.0)
    return float(np.mean(guessed == subset.targets))


def evaluate_all(model: Model, dataset: Dataset) -> np.ndarray:
    return np.asarray([evaluate_skill(model, dataset, skill) for skill in range(3)])
