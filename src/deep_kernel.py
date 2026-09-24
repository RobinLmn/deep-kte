from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, pairwise_kernels
from torch.utils.checkpoint import checkpoint
from transformers import BertConfig, BertModel

from utils import pdist2, to_tensor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class ConvolutionalFeature(nn.Module):
    def __init__(self, in_channels, output_dim=100):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.GroupNorm(8, 32), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2), nn.GroupNorm(8, 64), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.Flatten(), nn.LazyLinear(output_dim))

    def forward(self, outcomes):
        return self.layers(outcomes)


class LinearFeature(nn.Module):
    def __init__(self, input_dim, output_dim=50, hidden=50):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden), nn.Softplus(), nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, output_dim))

    def forward(self, outcomes):
        return self.layers(outcomes.flatten(1))


class TreeGNNFeature(nn.Module):
    def __init__(self, label_buckets=256, hidden=50, layers=3, output_dim=50):
        super().__init__()
        self.label_buckets = label_buckets
        self.embedding = nn.Embedding(label_buckets + 1, hidden, padding_idx=0)
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, outcomes):
        if outcomes.ndim == 4:
            outcomes = outcomes.squeeze(1)

        outcomes = outcomes[:, outcomes[..., 0].any(0)]
        labels = outcomes[..., 0].long()
        parents = outcomes[..., 1].long() - 1
        present = labels != 0
        buckets = torch.where(present, labels.remainder(self.label_buckets) + 1, 0)
        hidden = self.embedding(buckets)

        parent_indices = parents.clamp_min(0).unsqueeze(-1).expand_as(hidden)
        children = present & (parents >= 0)

        for update in self.updates:
            child_messages = torch.zeros_like(hidden)
            child_messages.scatter_add_(1, parent_indices, hidden * children.unsqueeze(-1))
            parent_messages = hidden.gather(1, parent_indices) * children.unsqueeze(-1)
            hidden = update(hidden + child_messages + parent_messages) * present.unsqueeze(-1)

        counts = present.sum(1, keepdim=True).clamp_min(1)
        return self.output(hidden.sum(1) / torch.sqrt(counts))


class TransformerFeature(nn.Module):
    def __init__(self, vocab_size, max_tokens, model_dim=128, heads=4, layers=2, output_dim=50):
        super().__init__()
        config = BertConfig(vocab_size=1, hidden_size=model_dim, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=2 * model_dim, max_position_embeddings=max_tokens, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.project = nn.Linear(model_dim, model_dim)
        self.encoder = BertModel(config, add_pooling_layer=False)
        self.query = nn.Parameter(torch.randn(model_dim) / np.sqrt(model_dim))
        self.head = nn.Linear(model_dim, output_dim)
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)

    def forward(self, outcomes):
        ids = outcomes.reshape(len(outcomes), -1).long()
        features = [checkpoint(self.encode, chunk, use_reentrant=False) if torch.is_grad_enabled() else self.encode(chunk) for chunk in ids.split(64)]
        return torch.cat(features)

    def encode(self, ids):
        mask = ids > 0
        hidden = self.encoder(inputs_embeds=self.project(self.embedding(ids)), attention_mask=mask.long()).last_hidden_state
        scores = (hidden @ self.query).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.head((weights * hidden).sum(1))


class LearnedDeepKernel(nn.Module):
    def __init__(self, zeta_theta, sigma_init=1.0, sigma_not_init=0.1, q_kernel=None, kernel_composition="additive", epsilon_min=0.001, distance_float64=False):
        super().__init__()
        self.distance_float64 = distance_float64
        self.zeta_theta = zeta_theta

        if q_kernel is None:
            self.sqrt_sigma = nn.Parameter(torch.tensor(np.sqrt(sigma_init), dtype=torch.float32))

        self.sqrt_sigma_not = nn.Parameter(torch.tensor(np.sqrt(sigma_not_init), dtype=torch.float32))
        self.epsilon_logit = nn.Parameter(torch.zeros(()))
        self.epsilon_min = epsilon_min
        self.q_kernel = q_kernel
        self.kernel_composition = kernel_composition
        self.q_cache = {}

    def feature(self, outcomes):
        return self.zeta_theta(outcomes)

    def k_theta(self, first_features, second_features, first_outcomes, second_outcomes):
        kappa = torch.exp(-pdist2(first_features, second_features, self.distance_float64) / self.sqrt_sigma_not**2)

        if self.q_kernel is None:
            base_gram = torch.exp(-pdist2(first_outcomes.flatten(1), second_outcomes.flatten(1), self.distance_float64) / self.sqrt_sigma**2)
        else:
            q_key = (first_outcomes.data_ptr(), second_outcomes.data_ptr())
            if not self.training or q_key not in self.q_cache:
                base_gram = self.q_kernel.k_theta(self.q_kernel.feature(first_outcomes), self.q_kernel.feature(second_outcomes), first_outcomes, second_outcomes)
                if self.training:
                    self.q_cache[q_key] = base_gram
            else:
                base_gram = self.q_cache[q_key]

        return self.combine(kappa, base_gram)

    def epsilon(self):
        return self.epsilon_min + (1.0 - self.epsilon_min) * torch.sigmoid(self.epsilon_logit)

    def combine(self, kappa, base_gram):
        epsilon = self.epsilon()
        if self.kernel_composition == "additive":
            return (1 - epsilon) * kappa + epsilon * base_gram
        if self.kernel_composition == "product":
            return ((1 - epsilon) * kappa + epsilon) * base_gram

    def forward(self, first_outcomes, second_outcomes):
        return self.k_theta(self.feature(first_outcomes), self.feature(second_outcomes), first_outcomes, second_outcomes)

    def parameter_values(self):
        values = {"epsilon": self.epsilon().item(), "sigma_not": self.sqrt_sigma_not.item() ** 2}
        if self.q_kernel is None:
            values["sigma"] = self.sqrt_sigma.item() ** 2
        return values


class LearnedGaussianKernelBandwidth(nn.Module):
    def __init__(self, sigma_init=1.0, distance_float64=False):
        super().__init__()
        self.distance_float64 = distance_float64
        self.sqrt_sigma = nn.Parameter(torch.tensor(np.sqrt(sigma_init), dtype=torch.float32))

    def feature(self, outcomes):
        return outcomes.flatten(1)

    def k_theta(self, first_features, second_features, first_outcomes, second_outcomes):
        return torch.exp(-pdist2(first_features, second_features, self.distance_float64) / self.sqrt_sigma**2)

    def forward(self, first_outcomes, second_outcomes):
        return self.k_theta(self.feature(first_outcomes), self.feature(second_outcomes), first_outcomes, second_outcomes)

    def parameter_values(self):
        return {"sigma": self.sqrt_sigma.item() ** 2}


class GaussianKernel:
    def __init__(self, learning_outcomes, bandwidth_q=0.5, distance_float64=False):
        self.distance_float64 = distance_float64
        learning_features = to_tensor(learning_outcomes, device=device).flatten(1)
        self.sigma2 = torch.quantile(torch.sqrt(pdist2(learning_features, learning_features, self.distance_float64)).flatten(), bandwidth_q).item() ** 2

    def feature(self, outcomes):
        return outcomes.flatten(1)

    def eval(self):
        return self

    def k_theta(self, first_features, second_features, first_outcomes, second_outcomes):
        return torch.exp(-pdist2(first_features, second_features, self.distance_float64) / (self.sigma2 + 1e-12))


class LinearKernel:
    def feature(self, outcomes):
        return outcomes.flatten(1)

    def k_theta(self, first_features, second_features, first_outcomes, second_outcomes):
        return first_features @ second_features.T


class WeisfeilerLehmanKernel:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.cache_grams = False
        self.feature_cache = {}
        self.gram_cache = {}

    @classmethod
    def from_data(cls, outcomes, max_samples=500):
        kernel = cls()
        sample = to_tensor(outcomes[: min(max_samples, len(outcomes))], device=device)
        gram = kernel.k_theta(kernel.feature(sample), kernel.feature(sample), sample, sample)
        kernel.scale = max(torch.median(torch.diag(gram)).item(), 1.0)
        return kernel

    def feature(self, outcomes):
        if outcomes.ndim == 4 and outcomes.shape[1] == 1:
            outcomes = outcomes.squeeze(1)

        array = outcomes[..., 2] if outcomes.ndim == 3 and outcomes.shape[-1] >= 3 else outcomes
        array = array.detach().cpu().numpy().astype(np.int64)
        key = (array.shape, blake2b(array.tobytes(), digest_size=16).digest())

        if self.cache_grams and key in self.feature_cache:
            return (key, self.feature_cache[key])

        (row_indices, positions) = np.nonzero(array)
        labels = array[row_indices, positions]
        width = int(labels.max()) + 1 if len(labels) else 1
        features = csr_matrix((np.ones(len(labels), dtype=np.float32), (row_indices, labels)), shape=(len(array), width), dtype=np.float32)
        features.sum_duplicates()

        if self.cache_grams:
            self.feature_cache[key] = features

        return (key, features)

    def eval(self):
        return self

    def k_theta(self, features1, features2, first_outcomes, second_outcomes):
        (key1, features1) = features1
        (key2, features2) = features2
        cache_key = (key1, key2, str(first_outcomes.device))

        if self.cache_grams and cache_key in self.gram_cache:
            return self.gram_cache[cache_key]

        reverse_key = (key2, key1, str(first_outcomes.device))

        if self.cache_grams and reverse_key in self.gram_cache:
            return self.gram_cache[reverse_key].T

        shared_width = min(features1.shape[1], features2.shape[1])
        gram = (features1[:, :shared_width] @ features2[:, :shared_width].T).toarray()
        gram = gram / self.scale
        result = torch.tensor(gram, dtype=torch.float32, device=first_outcomes.device)

        if self.cache_grams:
            self.gram_cache[cache_key] = result

        return result

    def clear_cache(self):
        self.feature_cache.clear()
        self.gram_cache.clear()


def make_token_kernel_nn(outcomes, feature_type, bandwidth_q, kernel_composition, epsilon_min=1e-09, distance_float64=False, vocabulary_size=151670):
    sample = to_tensor(outcomes[: min(500, len(outcomes))], device=device)

    if feature_type in {"bandwidth", "token_bandwidth"}:
        with torch.no_grad():
            distances = pdist2(sample.flatten(1), sample.flatten(1), distance_float64)
            sigma_init = max(torch.quantile(distances[distances > 0], bandwidth_q).item(), 1e-06)
        return LearnedGaussianKernelBandwidth(sigma_init=sigma_init, distance_float64=distance_float64).to(device)

    feature = TransformerFeature(vocabulary_size, max_tokens=outcomes.shape[-2]).to(device)

    with torch.no_grad():
        features = feature(sample)
        distances = pdist2(features, features, distance_float64)
        sigma_not_init = max(torch.quantile(distances[distances > 0], bandwidth_q).item(), 1e-06)
        outcomes = to_tensor(outcomes, device=device)
        distances = torch.sqrt(pdist2(outcomes.flatten(1), outcomes.flatten(1), distance_float64)).flatten()
        sigma_init = max(torch.quantile(distances, bandwidth_q).item() ** 2, 1e-06)

    return LearnedDeepKernel(feature, sigma_init=sigma_init, sigma_not_init=sigma_not_init, kernel_composition=kernel_composition, epsilon_min=epsilon_min, distance_float64=distance_float64).to(device)


def make_kernel_nn(outcomes, feature_type, bandwidth_q, kernel_type, kernel_composition, epsilon_min=1e-09, distance_float64=False):
    if feature_type in {"token_transformer", "token_bandwidth"}:
        return make_token_kernel_nn(outcomes, feature_type, bandwidth_q, kernel_composition, epsilon_min, distance_float64)

    sample = to_tensor(outcomes[: min(500, len(outcomes))])

    if feature_type == "bandwidth":
        with torch.no_grad():
            squared_distances = pdist2(sample.flatten(1), sample.flatten(1), distance_float64)
            sigma_init = max(torch.quantile(squared_distances[squared_distances > 0], bandwidth_q).item(), 1e-06)
        return LearnedGaussianKernelBandwidth(sigma_init=sigma_init, distance_float64=distance_float64).to(device)

    if feature_type == "gnn":
        deep_feature = TreeGNNFeature()
    elif feature_type == "linear":
        input_dim = np.prod(outcomes.shape[1:]) if outcomes.ndim > 1 else 1
        deep_feature = LinearFeature(input_dim)
    elif feature_type == "cnn":
        input_channels = outcomes.shape[3] if outcomes.ndim == 4 else 1
        deep_feature = ConvolutionalFeature(input_channels)

    with torch.no_grad():
        feature_distances = pdist2(deep_feature(sample), deep_feature(sample), distance_float64)
        sigma_not_init = max(torch.quantile(feature_distances[feature_distances > 0], bandwidth_q).item(), 1e-06)

    if device == "cuda" and feature_type == "cnn":
        deep_feature = torch.compile(deep_feature)

    q_kernel = WeisfeilerLehmanKernel.from_data(outcomes) if kernel_type == "wl" else None
    sigma_init = None

    if q_kernel is None:
        outcomes = to_tensor(outcomes, device=device)
        with torch.no_grad():
            distances = torch.sqrt(pdist2(outcomes.flatten(1), outcomes.flatten(1), distance_float64)).flatten()
            sigma_init = max(torch.quantile(distances, bandwidth_q).item() ** 2, 1e-06)

    return LearnedDeepKernel(deep_feature, sigma_init=sigma_init, sigma_not_init=sigma_not_init, q_kernel=q_kernel, kernel_composition=kernel_composition, epsilon_min=epsilon_min, distance_float64=distance_float64).to(device)


class AveragedKernel:
    def __init__(self, kernels):
        self.kernels = kernels

    def eval(self):
        for kernel in self.kernels:
            kernel.eval()
        return self

    def feature(self, outcomes):
        return [kernel.feature(outcomes) for kernel in self.kernels]

    def k_theta(self, features1, features2, first_outcomes, second_outcomes):
        if all(isinstance(kernel, LearnedDeepKernel) and isinstance(kernel.q_kernel, WeisfeilerLehmanKernel) for kernel in self.kernels):
            q_kernel = self.kernels[0].q_kernel
            base_gram = q_kernel.k_theta(q_kernel.feature(first_outcomes), q_kernel.feature(second_outcomes), first_outcomes, second_outcomes)
            matrices = []
            for kernel, first_features, second_features in zip(self.kernels, features1, features2):
                kappa = torch.exp(-pdist2(first_features, second_features, kernel.distance_float64) / kernel.sqrt_sigma_not**2)
                matrices.append(kernel.combine(kappa, base_gram))
            return sum(matrices) / len(matrices)

        matrices = [kernel.k_theta(first_features, second_features, first_outcomes, second_outcomes) for (kernel, first_features, second_features) in zip(self.kernels, features1, features2)]
        return sum(matrices) / len(matrices)


class EstimatedLoggingPolicy:
    def __init__(self, covariates, treatments):
        self.model = LogisticRegression()
        self.model.fit(covariates.reshape(len(covariates), -1), (treatments.reshape(-1) + 1) // 2)

    def get_propensities(self, covariates, treatment):
        probabilities = self.model.predict_proba(covariates.reshape(len(covariates), -1))[:, 1]
        treatment = ((treatment + 1) // 2).astype(int)
        return np.where(treatment == 1, probabilities, 1 - probabilities)


def fit_logging_policy(covariates, treatments):
    return EstimatedLoggingPolicy(covariates, treatments)


def logging_policy_weights(propensity_estimator, covariates, treatments):
    actions = treatments.reshape(-1)
    propensities = propensity_estimator.get_propensities(covariates, actions)
    return ((actions == 1).astype(float) - (actions == 0).astype(float)) / propensities


def get_cme_nuisance_estimator(covariates, treatments, reg_lambda=0.1):
    m = len(covariates)
    covariates = covariates.reshape(m, -1)
    treatments = treatments.reshape(m, -1)
    covariate_bandwidth = np.median(pairwise_distances(covariates, covariates, metric="euclidean")) ** 2
    treatment_bandwidth = np.median(pairwise_distances(treatments, treatments, metric="euclidean")) ** 2

    if treatment_bandwidth == 0:
        treatment_bandwidth = 1.0

    covariate_gram = pairwise_kernels(covariates, metric="rbf", gamma=1.0 / covariate_bandwidth)
    treatment_gram = pairwise_kernels(treatments, metric="rbf", gamma=1.0 / treatment_bandwidth)
    cholesky_factor = cho_factor(np.multiply(covariate_gram, treatment_gram) + m * reg_lambda * np.eye(m))

    def estimate(test_covariates, test_treatments):
        test_covariates = test_covariates.reshape(len(test_covariates), -1)
        test_treatments = test_treatments.reshape(len(test_treatments), -1)
        covariate_cross_gram = pairwise_kernels(covariates, test_covariates, metric="rbf", gamma=1.0 / covariate_bandwidth)
        treatment_cross_gram = pairwise_kernels(treatments, test_treatments, metric="rbf", gamma=1.0 / treatment_bandwidth)
        return cho_solve(cholesky_factor, np.multiply(covariate_cross_gram, treatment_cross_gram))

    return estimate


def pseudo_map_weights(cme_nuisance_estimator, propensity_estimator, covariates, treatments):
    weights = logging_policy_weights(propensity_estimator, covariates, treatments)
    (treated_actions, control_actions) = (np.ones(len(treatments)), np.zeros(len(treatments)))
    observed_embedding = cme_nuisance_estimator(covariates, treatments)
    treated_embedding = cme_nuisance_estimator(covariates, treated_actions)
    control_embedding = cme_nuisance_estimator(covariates, control_actions)
    correction = treated_embedding - control_embedding - observed_embedding * weights[np.newaxis, :]
    return (correction, weights)


def get_nuisance_parameters(outcomes, covariates, treatments, reg_lambda, fit_indices, first_selection_indices, second_selection_indices):
    cme_nuisance_estimator = get_cme_nuisance_estimator(covariates[fit_indices], treatments[fit_indices], reg_lambda)
    propensity_estimator = fit_logging_policy(covariates[fit_indices], treatments[fit_indices])
    (first_correction, first_weights) = pseudo_map_weights(cme_nuisance_estimator, propensity_estimator, covariates[first_selection_indices], treatments[first_selection_indices])
    (second_correction, second_weights) = pseudo_map_weights(cme_nuisance_estimator, propensity_estimator, covariates[second_selection_indices], treatments[second_selection_indices])
    return (outcomes[fit_indices], outcomes[first_selection_indices], outcomes[second_selection_indices], *to_tensor(first_correction, first_weights, second_correction, second_weights, device=device))


def get_training_folds(outcomes, covariates, treatments, reg_lambda):
    m = len(outcomes) // 3
    folds = [np.arange(0, m), np.arange(m, 2 * m), np.arange(2 * m, len(outcomes))]
    return [get_nuisance_parameters(outcomes, covariates, treatments, reg_lambda, folds[fit_index], folds[first_index], folds[second_index]) for (fit_index, first_index, second_index) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]


def test_power_loss(fit_outcomes, first_selection_outcomes, second_selection_outcomes, first_correction, first_weights, second_correction, second_weights, learned_kernel, sigma_reg_lambda=1e-08):
    fit_features = learned_kernel.feature(fit_outcomes)
    first_selection_features = learned_kernel.feature(first_selection_outcomes)
    second_selection_features = learned_kernel.feature(second_selection_outcomes)

    fit_gram = learned_kernel.k_theta(fit_features, fit_features, fit_outcomes, fit_outcomes)
    first_fit_gram = learned_kernel.k_theta(first_selection_features, fit_features, first_selection_outcomes, fit_outcomes)
    fit_second_gram = learned_kernel.k_theta(fit_features, second_selection_features, fit_outcomes, second_selection_outcomes)
    selection_gram = learned_kernel.k_theta(first_selection_features, second_selection_features, first_selection_outcomes, second_selection_outcomes)

    cross_gram = first_weights * selection_gram * second_weights.T + first_weights * (first_fit_gram @ second_correction) + first_correction.T @ fit_second_gram * second_weights.T + first_correction.T @ fit_gram @ second_correction

    mean_features = torch.mean(cross_gram, dim=0)
    cross_mean = torch.mean(mean_features)
    variance = torch.mean((mean_features - cross_mean) ** 2) + sigma_reg_lambda
    power_objective = cross_mean / torch.sqrt(variance)
    return -power_objective


def test_pseudo_features(cme_estimator, propensity_estimator, test_covariates, test_treatments):
    n_test = len(test_covariates)
    m = n_test // 2
    first_indices = np.arange(m)
    second_indices = np.arange(m, n_test)
    (first_correction, first_weights) = pseudo_map_weights(cme_estimator, propensity_estimator, test_covariates[first_indices], test_treatments[first_indices])
    (second_correction, second_weights) = pseudo_map_weights(cme_estimator, propensity_estimator, test_covariates[second_indices], test_treatments[second_indices])
    return (first_indices, second_indices, first_correction, first_weights, second_correction, second_weights)


@torch.no_grad()
def learning_kernel_terms(learning_outcomes, outcome_kernel):
    learning_outcomes = to_tensor(learning_outcomes, device=device)
    learning_features = outcome_kernel.feature(learning_outcomes)

    learning_gram = outcome_kernel.k_theta(learning_features, learning_features, learning_outcomes, learning_outcomes).cpu().numpy()
    return (learning_features, learning_gram)


@torch.no_grad()
def mmd2_dr_kpt_stat(learning_outcomes, test_outcomes, features, outcome_kernel, sigma_reg_lambda=0.0, learning_features=None, learning_gram=None):
    (first_indices, second_indices, first_correction, first_weights, second_correction, second_weights) = features
    m = len(first_indices)
    (learning_outcomes, first_outcomes, second_outcomes) = (to_tensor(outcomes, device=device) for outcomes in (learning_outcomes, test_outcomes[first_indices], test_outcomes[second_indices]))
    learning_features = outcome_kernel.feature(learning_outcomes) if learning_features is None else learning_features
    first_features = outcome_kernel.feature(first_outcomes)
    second_features = outcome_kernel.feature(second_outcomes)

    learning_gram = outcome_kernel.k_theta(learning_features, learning_features, learning_outcomes, learning_outcomes).cpu().numpy() if learning_gram is None else learning_gram
    first_learning_gram = outcome_kernel.k_theta(first_features, learning_features, first_outcomes, learning_outcomes).cpu().numpy()
    learning_second_gram = outcome_kernel.k_theta(learning_features, second_features, learning_outcomes, second_outcomes).cpu().numpy()
    test_gram = outcome_kernel.k_theta(first_features, second_features, first_outcomes, second_outcomes).cpu().numpy()

    first_weights = first_weights[:, None]
    second_weights = second_weights[:, None]

    cross_gram = first_weights * test_gram * second_weights.T + first_weights * (first_learning_gram @ second_correction) + first_correction.T @ learning_second_gram * second_weights.T + first_correction.T @ learning_gram @ second_correction

    mean_features = np.sum(cross_gram, axis=0) / m
    cross_mean = np.sum(mean_features) / m
    variance = np.sum((mean_features - cross_mean) ** 2) / m + sigma_reg_lambda

    test_statistic = np.sqrt(m) * cross_mean / np.sqrt(variance)

    p_value = 1.0 - norm.cdf(test_statistic)
    return (p_value, test_statistic)


def validate(validation_sets, learning_outcomes, outcome_kernel, sigma_reg_lambda=1e-08):
    if isinstance(outcome_kernel, LearnedDeepKernel) and isinstance(outcome_kernel.q_kernel, WeisfeilerLehmanKernel):
        outcome_kernel.q_kernel.cache_grams = True
    (learning_features, learning_gram) = learning_kernel_terms(learning_outcomes, outcome_kernel)
    return np.mean([mmd2_dr_kpt_stat(learning_outcomes, test_outcomes, features, outcome_kernel, sigma_reg_lambda=sigma_reg_lambda, learning_features=learning_features, learning_gram=learning_gram)[1] for (test_outcomes, features) in validation_sets])


@dataclass
class TrainingConfig:
    feature_type: str
    epochs: int = 1000
    lr: float = 0.001
    reg_lambda: float = 0.1
    sigma_reg_lambda: float = 1e-08
    bandwidth_q: float = 0.5
    ema_decay: float = 0.99
    kernel_type: str = "gaussian"
    kernel_composition: str = "additive"
    epsilon_min: float = 1e-09
    feature_lr: float = 0.02
    distance_float64: bool = False


def make_optimizers(learned_kernel, training_config):
    feature_params = list(learned_kernel.zeta_theta.parameters()) if hasattr(learned_kernel, "zeta_theta") else []
    kernel_params = [parameter for parameter in learned_kernel.parameters() if all(parameter is not other for other in feature_params)]
    return ([torch.optim.Adam(feature_params, lr=training_config.feature_lr)] if feature_params else []) + [torch.optim.Adam(kernel_params, lr=training_config.lr)]


def run_epoch_callback(epoch, learned_kernel, ema, epoch_callback):
    if epoch_callback is None:
        return

    if ema is None:
        epoch_callback(epoch, learned_kernel)
        return

    state = {name: value.detach().clone() for (name, value) in learned_kernel.state_dict().items()}
    learned_kernel.load_state_dict(ema)
    epoch_callback(epoch, learned_kernel)
    learned_kernel.load_state_dict(state)


def update_ema(ema, learned_kernel, decay):
    if ema is None:
        return

    with torch.no_grad():
        for name, value in learned_kernel.state_dict().items():
            if value.dtype.is_floating_point:
                ema[name].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema[name].copy_(value)


def train_deep_outcome_kernel(covariates, treatments, outcomes, training_config, epoch_callback=None):
    learned_kernel = make_kernel_nn(outcomes, training_config.feature_type, training_config.bandwidth_q, training_config.kernel_type, training_config.kernel_composition, training_config.epsilon_min, training_config.distance_float64)
    optimizers = make_optimizers(learned_kernel, training_config)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, training_config.epochs), eta_min=optimizer.defaults["lr"] * 0.01) for optimizer in optimizers]
    ema = {name: value.detach().clone() for (name, value) in learned_kernel.state_dict().items()} if training_config.ema_decay > 0 else None
    test_powers = []
    parameters = {}
    outcome_tensor = to_tensor(outcomes, device=device)
    training_folds = get_training_folds(outcome_tensor, covariates, treatments, training_config.reg_lambda)

    for epoch in range(training_config.epochs + 1):
        with torch.no_grad():
            loss = sum(test_power_loss(*fold, learned_kernel, training_config.sigma_reg_lambda) for fold in training_folds) / len(training_folds)
        test_powers.append(-loss.item())

        for name, value in learned_kernel.parameter_values().items():
            parameters.setdefault(name, []).append(value)

        if epoch == training_config.epochs:
            run_epoch_callback(epoch, learned_kernel, ema, epoch_callback)
            break

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = sum(test_power_loss(*fold, learned_kernel, training_config.sigma_reg_lambda) for fold in training_folds) / len(training_folds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learned_kernel.parameters(), max_norm=1.0)

        for optimizer in optimizers:
            optimizer.step()

        for scheduler in schedulers:
            scheduler.step()

        update_ema(ema, learned_kernel, training_config.ema_decay)
        run_epoch_callback(epoch, learned_kernel, ema, epoch_callback)

    if ema is not None:
        learned_kernel.load_state_dict(ema)

    return (learned_kernel, test_powers, parameters)
