import copy
import fcntl
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep_kernel import (
    AveragedKernel,
    GaussianKernel,
    LinearKernel,
    TrainingConfig,
    WeisfeilerLehmanKernel,
    fit_logging_policy,
    get_cme_nuisance_estimator,
    learning_kernel_terms,
    mmd2_dr_kpt_stat,
    test_pseudo_features,
    train_deep_outcome_kernel,
    validate,
)
from environment import make_scenario

run_id = os.environ.get("CNME_RUN_ID", "results")
run_dir = Path(__file__).resolve().parents[1] / "out" / run_id
rng = np.random.default_rng(0)
outer_reps = int(os.environ.get("CNME_OUTER_REPS", 20))
seed_offset = int(os.environ.get("CNME_SEED_OFFSET", 0))
all_methods = ["DR-ATE", "DR-KTE", "DR-KTE-Opt", "DR-KTE-Deep"]
default_methods = os.environ.get("CNME_METHODS", ",".join(all_methods)).split(",")


@dataclass
class ExperimentConfig:
    environment: str
    feature_type: str
    scale: float = 0.0
    num_repetitions: int = 200
    epochs: int = 500
    lr: float = 0.001
    ema_decay: float = 0.95
    n_samples: int = 1500
    train_val_test_split: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    alpha: float = 0.05
    validation_every: int = 1
    validation_repetition: int = 1
    bandwidth_q: float = 0.5
    kernel_type: str = "gaussian"
    kernel_composition: str = "additive"
    epsilon_min: float = 1e-09
    feature_lr: float = 0.02
    sigma_reg_lambda: float = 0.0001
    sigma_reg_lambda_test: float = 0.0
    reg_lambda: float = 0.001
    ensemble_size: int = 3
    methods: list[str] = field(default_factory=lambda: list(default_methods))
    distance_float64: bool = False


def split_sample_sizes(config):
    fractions = config.train_val_test_split
    n_train = int(config.n_samples * fractions[0])
    n_validation = int(config.n_samples * fractions[1])
    n_test = config.n_samples - n_train - n_validation
    return (n_train, n_validation, n_test)


def test(config, kernels, cme_estimator, propensity_estimator, learning_outcomes, rng):
    rejections = {method: [] for method in kernels}
    test_statistics = {method: [] for method in kernels}
    p_values = {method: [] for method in kernels}
    learning_terms = {method: learning_kernel_terms(learning_outcomes, kernel) for (method, kernel) in kernels.items()}
    (_, _, n_test) = split_sample_sizes(config)

    for _ in range(config.num_repetitions):
        (covariates, treatments, outcomes) = make_experiment_data(config, n_test, rng, pool="test")
        features = test_pseudo_features(cme_estimator, propensity_estimator, covariates, treatments)

        for method, kernel in kernels.items():
            (learning_features, learning_gram) = learning_terms[method]
            (p_value, test_statistic) = mmd2_dr_kpt_stat(learning_outcomes, outcomes, features, kernel, sigma_reg_lambda=config.sigma_reg_lambda_test, learning_features=learning_features, learning_gram=learning_gram)
            rejections[method].append(float(p_value < config.alpha))
            test_statistics[method].append(float(test_statistic))
            p_values[method].append(float(p_value))

    summaries = {method: (np.mean(rejections[method]), np.mean(test_statistics[method])) for method in kernels}
    return (summaries, test_statistics, p_values)


def make_validation_sets(config, cme_estimator, propensity_estimator, rng):
    (_, n_validation, _) = split_sample_sizes(config)
    sets = []
    data = [make_experiment_data(config, n_validation, rng, pool="validation") for _ in range(config.validation_repetition)]
    for covariates, treatments, outcomes in data:
        sets.append((outcomes, test_pseudo_features(cme_estimator, propensity_estimator, covariates, treatments)))
    return (sets, data[0])


def train_with_validation(config, covariates, treatments, outcomes, feature_type, validation_sets, ensemble_size=1, member_seeds=None):
    kernels = []
    member_test_powers = []
    member_parameters = []
    member_validation_test_stats = []
    training_config = TrainingConfig(feature_type, epochs=config.epochs, lr=config.lr, reg_lambda=config.reg_lambda, sigma_reg_lambda=config.sigma_reg_lambda, bandwidth_q=config.bandwidth_q, ema_decay=config.ema_decay, kernel_type=config.kernel_type, kernel_composition=config.kernel_composition, epsilon_min=config.epsilon_min, feature_lr=config.feature_lr, distance_float64=config.distance_float64)

    for member in range(ensemble_size):
        if member_seeds is not None:
            torch.manual_seed(member_seeds[member])

        validation_test_stats = []
        best_kernel = {"T_theta": -np.inf, "state": None}

        def select_kernel(epoch, learned_kernel):
            if config.validation_every == 0 or epoch % config.validation_every != 0:
                return

            learned_kernel.eval()
            test_statistic = validate(validation_sets, outcomes, learned_kernel, config.sigma_reg_lambda_test)
            validation_test_stats.append(test_statistic)
            learned_kernel.train()

            if test_statistic >= best_kernel["T_theta"]:
                (best_kernel["T_theta"], best_kernel["state"]) = (test_statistic, copy.deepcopy(learned_kernel.state_dict()))

        (kernel, test_powers, parameters) = train_deep_outcome_kernel(covariates, treatments, outcomes, training_config, epoch_callback=select_kernel)

        if best_kernel["state"] is not None:
            kernel.load_state_dict(best_kernel["state"])

        if hasattr(kernel, "q_kernel") and isinstance(kernel.q_kernel, WeisfeilerLehmanKernel):
            kernel.q_kernel.cache_grams = False
            kernel.q_kernel.clear_cache()

        kernels.append(kernel)
        member_test_powers.append(test_powers)
        member_parameters.append(parameters)
        member_validation_test_stats.append(validation_test_stats)

    kernel = kernels[0] if ensemble_size == 1 else AveragedKernel(kernels)
    return (kernel, member_test_powers, member_parameters, member_validation_test_stats)


def make_experiment_data(config, n, rng, pool="train"):
    (covariates, treatments, outcomes, _, _, _) = make_scenario(config.environment, n, config.scale, rng, pool=pool)
    return (covariates, treatments, outcomes)


def run_experiment(config: ExperimentConfig, rng, save_filename="", rep=0):
    (n_train, _, _) = split_sample_sizes(config)
    (covariates, treatments, outcomes) = make_experiment_data(config, n_train, rng)
    cme_estimator = get_cme_nuisance_estimator(covariates, treatments, config.reg_lambda)
    propensity_estimator = fit_logging_policy(covariates, treatments)

    test_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.uint32).max))
    (validation_sets, validation_data) = make_validation_sets(config, cme_estimator, propensity_estimator, rng)

    kernels = {}
    (test_powers, parameters, validation_test_stats) = ([], {}, [])
    (bandwidth_test_powers, bandwidth_parameters, bandwidth_validation) = ([], {}, [])
    member_seeds = []
    methods = config.methods

    if "DR-ATE" in methods:
        kernels["DR-ATE"] = LinearKernel()

    if "DR-KTE" in methods:
        kernels["DR-KTE"] = WeisfeilerLehmanKernel.from_data(outcomes) if config.kernel_type == "wl" else GaussianKernel(np.concatenate((outcomes, validation_data[2])), config.bandwidth_q, distance_float64=config.distance_float64)

    if "DR-KTE-Deep" in methods:
        member_seeds = [int(np.random.SeedSequence([seed_offset, rep, member]).generate_state(1)[0]) for member in range(config.ensemble_size)]
        (learned_kernel, test_powers, parameters, validation_test_stats) = train_with_validation(config, covariates, treatments, outcomes, config.feature_type, validation_sets, config.ensemble_size, member_seeds)
        kernels["DR-KTE-Deep"] = learned_kernel.eval()

    if "DR-KTE-Opt" in methods:
        bandwidth_feature = "token_bandwidth" if config.feature_type == "token_transformer" else "bandwidth"
        (bandwidth_kernel, bandwidth_test_powers, bandwidth_parameters, bandwidth_validation) = train_with_validation(config, covariates, treatments, outcomes, bandwidth_feature, validation_sets)
        kernels["DR-KTE-Opt"] = bandwidth_kernel.eval()

    covariates = np.concatenate((covariates, validation_data[0]))
    treatments = np.concatenate((treatments, validation_data[1]))
    outcomes = np.concatenate((outcomes, validation_data[2]))
    cme_estimator = get_cme_nuisance_estimator(covariates, treatments, config.reg_lambda)
    propensity_estimator = fit_logging_policy(covariates, treatments)

    (summaries, test_statistics, p_values) = test(config, kernels, cme_estimator, propensity_estimator, outcomes, test_rng)
    results = {"test_powers": test_powers, "parameters": parameters, "validation_test_stats": validation_test_stats, "member_seeds": member_seeds, "test_statistics": test_statistics, "p_values": p_values}
    results.update({method: power for (method, (power, _)) in summaries.items()})

    if save_filename is not None:
        save_result(config, results, run_dir / save_filename, seed_offset + rep)

    return ((test_powers, parameters, validation_test_stats), (bandwidth_test_powers, bandwidth_parameters, bandwidth_validation))


def save_result(config, result, path, outer_repetition):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"environment": config.environment, "feature_type": config.feature_type, "outer_repetition": outer_repetition, "repetition": config.num_repetitions, "epochs": config.epochs, "lr": config.lr, "n_samples": config.n_samples, "train_val_test_split": json.dumps(config.train_val_test_split), "alpha": config.alpha, "scale": config.scale, "bandwidth_q": config.bandwidth_q, "kernel_type": config.kernel_type, "kernel_composition": config.kernel_composition, "epsilon_min": config.epsilon_min, "feature_lr": config.feature_lr, "ema_decay": config.ema_decay, "validation_every": config.validation_every, "validation_repetition": config.validation_repetition, "sigma_reg_lambda": config.sigma_reg_lambda, "sigma_reg_lambda_test": config.sigma_reg_lambda_test, "reg_lambda": config.reg_lambda, "optimizer": "adam", "ensemble_size": config.ensemble_size, "member_seeds": json.dumps(result["member_seeds"]), "distance_float64": config.distance_float64, **{method: result.get(method) for method in all_methods}, "test_powers": json.dumps([[float(value) for value in member] for member in result["test_powers"]]), "validation_test_stats": json.dumps([[float(value) for value in member] for member in result["validation_test_stats"]]), "parameters": json.dumps([{name: list(map(float, values)) for (name, values) in member.items()} for member in result["parameters"]]), "test_statistics": json.dumps(result["test_statistics"]), "p_values": json.dumps(result["p_values"])}

    with open(path.with_suffix(".csv"), "a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0, os.SEEK_END)
        pd.DataFrame([row]).to_csv(file, header=file.tell() == 0, index=False)
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)


def run_sweep(job_name, config, variants, outer_reps, rng):
    for rep in range(outer_reps):
        for overrides in variants:
            run_experiment(replace(config, **overrides), rng, save_filename=job_name, rep=rep)


def run_sample_sweep(job_name, config, n_samples, scale=0.0, outer_reps=outer_reps):
    variants = [{"scale": scale, "n_samples": n} for n in n_samples]
    run_sweep(job_name, config, variants, outer_reps, rng)


def run_scale_sweep(job_name, config, scales, outer_reps=outer_reps):
    variants = [{"scale": scale} for scale in scales]
    run_sweep(job_name, config, variants, outer_reps, rng)
