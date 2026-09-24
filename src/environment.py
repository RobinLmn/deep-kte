import pickle
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ArrayDataset:
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return (self.images[index], self.targets[index])


@cache
def pool_permutation(n):
    return np.random.default_rng(n).permutation(n)


@cache
def get_mnist():
    return datasets.MNIST(root=DATA_DIR / "images" / "mnist", train=True, download=False)


@cache
def get_mnist_real():
    mnist = get_mnist()
    image_arrays = [np.array(mnist[index][0].resize((32, 32))) for index in range(len(mnist))]
    return np.stack(image_arrays).astype(np.float64) / 255.0


@cache
def get_mnist_fake():
    with open(DATA_DIR / "images" / "fake_mnist" / "generated.pckl", "rb") as file:
        fake = np.asarray(pickle.load(file)[0]).squeeze(1)
    return (fake + 1.0) / 2.0


@cache
def get_mnist_real_dataset():
    mnist = get_mnist()
    return ArrayDataset(get_mnist_real(), np.asarray(mnist.targets))


@cache
def get_mnist_fake_dataset():
    mnist = get_mnist()
    real = get_mnist_real()
    fake = get_mnist_fake()
    classifier = SGDClassifier(loss="log_loss", max_iter=20, tol=None, random_state=0, n_jobs=-1)
    classifier.fit(real.reshape(len(real), -1), np.asarray(mnist.targets))
    targets = classifier.predict(fake.reshape(len(fake), -1))
    return ArrayDataset(fake, targets)


@cache
def get_retinamnist(size=28):
    import medmnist
    from medmnist import INFO

    root = DATA_DIR / "images" / "medmnist"
    dataset_class = getattr(medmnist, INFO["retinamnist"]["python_class"])
    (images, targets) = ([], [])

    for split in ("train", "val", "test"):
        dataset = dataset_class(root=root, split=split, download=False, size=size)
        images.append(dataset.imgs.astype(np.float64) / 255.0)
        targets.append(dataset.labels.flatten())

    return ArrayDataset(np.concatenate(images), np.concatenate(targets))


@cache
def get_busi(size=32):
    base = DATA_DIR / "images" / "busi"
    (images, targets) = ([], [])

    for label, name in enumerate(["normal", "benign", "malignant"]):
        files = sorted((base / name).glob("*.png"))
        image_arrays = [np.array(Image.open(file).convert("L").resize((size, size))) for file in files]
        images.append(np.stack(image_arrays).astype(np.float64) / 255.0)
        targets.append(np.full(len(image_arrays), label))

    return ArrayDataset(np.concatenate(images), np.concatenate(targets))


@cache
def get_text_dataset(name, representation="tokens"):
    with np.load(DATA_DIR / "text" / name / f"{representation}.npz") as data:
        covariates = data["X"].astype(np.float32)
        outcomes = np.stack([data["negative"], data["neutral"], data["positive"]], axis=1)
        return (covariates, outcomes.astype(np.float32))


@cache
def get_graph_dataset(environment, scale):
    (name, kernel) = environment.rsplit("_", 1)
    dataset = {"code_graph": "llm_code_repairs", "contact_graph": "contact_networks"}[name]
    with np.load(DATA_DIR / "graphs" / dataset / kernel / f"delta{scale:g}.npz") as data:
        groups = np.unique(np.rec.fromarrays([data["project"], data["bug_id"]]), return_inverse=True)[1] if "bug_id" in data.files else None
        return (data["X"], data["Y0"], data["Y1"], groups)


def split_dataset(dataset, pool):
    indices = np.array_split(pool_permutation(len(dataset)), 3)[("train", "validation", "test").index(pool)]
    return dataset[indices]


def assign_treatment(covariates, alpha, rng):
    treatment_probability = np.clip(1.0 / (1.0 + np.exp(-(covariates @ alpha))), 0.07, 0.93)
    treatments = rng.binomial(1, treatment_probability, size=len(covariates)).astype(int)

    observed_propensity = np.where(treatments == 1, treatment_probability, 1.0 - treatment_probability)
    return (treatments, observed_propensity)


def generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng):
    control_noise = rng.normal(0, noise_std, size=len(covariates))
    treated_noise = rng.normal(0, noise_std, size=len(covariates))
    control_outcomes = covariates @ beta + control_noise
    treated_outcomes = covariates @ beta + effect_strength + treated_noise
    return (control_outcomes, treated_outcomes)


_class_indices = {}


def class_indices(dataset, label, pool):
    key = (id(dataset), int(label), pool)
    if key not in _class_indices:
        indices = np.where(np.array(dataset.targets) == label)[0]
        _class_indices[key] = split_dataset(indices, pool)
    return _class_indices[key]


def sample_from_dataset(label, dataset, rng, pool):
    indices = class_indices(dataset, label, pool)
    index = indices[rng.integers(0, len(indices)).item()]
    (image, _) = dataset[index]
    return np.array(image)


def observed_outcome(treatments, control_outcomes, treated_outcomes):
    select = treatments.reshape((-1,) + (1,) * (control_outcomes.ndim - 1))
    return np.where(select == 1, treated_outcomes, control_outcomes)


def discretize_continuous_outcomes(control_outcomes, treated_outcomes, low, high):
    (minimum_outcome, maximum_outcome) = (min(control_outcomes.min(), treated_outcomes.min()), max(control_outcomes.max(), treated_outcomes.max()))

    def discretize(outcomes):
        return np.clip(np.round((outcomes - minimum_outcome) / (maximum_outcome - minimum_outcome) * (high - low)), low, high).astype(int)

    return (discretize(control_outcomes), discretize(treated_outcomes))


def convert_to_structure_outcomes(control_outcomes, treated_outcomes, dataset, rng, pool):
    def sample(outcomes):
        return np.stack([sample_from_dataset(y, dataset, rng, pool) for y in outcomes])

    return (sample(control_outcomes), sample(treated_outcomes))


def make_scenario_default(environment, n, effect_strength, rng, pool):
    covariate_dimension = 5
    alpha = np.array([0.9, -0.75, 0.55, -0.4, 0.3])
    beta = np.array([0.6, -0.5, 0.4, -0.3, 0.2])

    covariates = rng.normal(0, 1, size=(n, covariate_dimension))
    (treatments, observed_propensity) = assign_treatment(covariates, alpha, rng)
    noise_std = 0.1

    if environment.startswith("continuous"):
        if environment == "continuous_III":
            effect_strength *= 2.0 * rng.binomial(n=1, p=0.5, size=n) - 1
        elif environment == "continuous_IV":
            effect_strength *= 4.0 * rng.binomial(n=1, p=0.5, size=n)
        (control_outcomes, treated_outcomes) = generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng)
    elif environment.startswith("mnist"):
        if environment == "mnist_III":
            effect_strength *= 2.0 * rng.binomial(n=1, p=0.5, size=n) - 1
        (control_outcomes, treated_outcomes) = generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng)
        (control_outcomes, treated_outcomes) = discretize_continuous_outcomes(control_outcomes, treated_outcomes, 0, 9)
        (control_outcomes, treated_outcomes) = convert_to_structure_outcomes(control_outcomes, treated_outcomes, get_mnist(), rng, pool)
    elif environment.startswith("retinamnist"):
        if environment == "retinamnist_III":
            effect_strength *= 2.0 * rng.binomial(n=1, p=0.5, size=n) - 1
        (control_outcomes, treated_outcomes) = generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng)
        (control_outcomes, treated_outcomes) = discretize_continuous_outcomes(control_outcomes, treated_outcomes, 0, 4)
        (control_outcomes, treated_outcomes) = convert_to_structure_outcomes(control_outcomes, treated_outcomes, get_retinamnist(), rng, pool)
    elif environment == "busi":
        effect_strength *= 2.0 * rng.binomial(n=1, p=0.5, size=n) - 1
        (control_outcomes, treated_outcomes) = generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng)
        (control_outcomes, treated_outcomes) = discretize_continuous_outcomes(control_outcomes, treated_outcomes, 0, 2)
        (control_outcomes, treated_outcomes) = convert_to_structure_outcomes(control_outcomes, treated_outcomes, get_busi(), rng, pool)

    outcomes = observed_outcome(treatments, control_outcomes, treated_outcomes)
    return (covariates, treatments, outcomes, observed_propensity, control_outcomes, treated_outcomes)


def make_scenario_fake_mnist(n, scale, rng, pool):
    labels = rng.integers(0, 10, size=n)
    covariates = labels[:, None].astype(np.float32)
    centered_labels = (covariates - 4.5) / np.sqrt(8.25)
    (treatments, observed_propensity) = assign_treatment(centered_labels, np.array([0.9]), rng)

    real = get_mnist_real_dataset()
    fake = get_mnist_fake_dataset()
    control_outcomes = np.stack([sample_from_dataset(label, real, rng, pool) for label in labels])
    treated_real = np.stack([sample_from_dataset(label, real, rng, pool) for label in labels])
    treated_fake = np.stack([sample_from_dataset(label, fake, rng, pool) for label in labels])
    replacement_probability = np.clip(scale * (labels + 1) / 40.0, 0.0, 1.0)
    replaced = rng.binomial(1, replacement_probability).reshape(n, 1, 1)

    treated_outcomes = np.where(replaced == 1, treated_fake, treated_real)
    outcomes = observed_outcome(treatments, control_outcomes, treated_outcomes)
    return (covariates, treatments, outcomes, observed_propensity, control_outcomes, treated_outcomes)


def make_scenario_causal_medmnist(dataset, n, scale, rng, pool):
    from causal_medmnist import Scenario, loaders

    loaders._ROOT = str(DATA_DIR / "images" / "medmnist")
    seed = int(rng.integers(0, 2**32))
    split = "val" if pool == "validation" else pool
    sample = Scenario(dataset, effect_strength=scale, center_effect=False).generate(n, seed=seed, split=split, replace=True)

    observed_propensity = np.where(sample.A == 1, sample.propensity, 1.0 - sample.propensity)
    return (sample.X.astype(np.float32), sample.A.astype(int), sample.Y, observed_propensity.astype(np.float32), sample.Y0, sample.Y1)


BLOB_LOCATIONS = np.array([(x, y) for x in range(3) for y in range(3)], dtype=np.float64)


def build_blob_covariances(effect_strength):
    control_covariance = 0.03 * np.eye(2)
    treated_covariances = np.tile(control_covariance, (9, 1, 1))

    for m in range(9):
        if m < 4:
            rho = -(0.02 + 0.002 * m)
        elif m == 4:
            rho = 0.0
        else:
            rho = 0.02 + 0.002 * (m - 5)

        treated_covariances[m, 0, 1] = effect_strength * rho
        treated_covariances[m, 1, 0] = effect_strength * rho

    return (control_covariance, treated_covariances)


def sample_blob_outcomes(modes, control_covariance, treated_covariances, rng):
    n = len(modes)
    control_outcomes = np.empty((n, 2))
    treated_outcomes = np.empty((n, 2))

    for index in range(n):
        m = modes[index]
        control_outcomes[index] = rng.multivariate_normal(BLOB_LOCATIONS[m], control_covariance)
        treated_outcomes[index] = rng.multivariate_normal(BLOB_LOCATIONS[m], treated_covariances[m])

    return (control_outcomes, treated_outcomes)


def make_scenario_blob(n, scale, rng, covariate_noise=0.3):
    alpha = np.array([1.2, -0.8])
    modes = rng.integers(0, 9, size=n)
    covariates = BLOB_LOCATIONS[modes] + rng.normal(0.0, covariate_noise, size=(n, 2))
    (treatments, observed_propensity) = assign_treatment(covariates, alpha, rng=rng)

    (control_covariance, treated_covariances) = build_blob_covariances(effect_strength=scale)
    (control_outcomes, treated_outcomes) = sample_blob_outcomes(modes, control_covariance, treated_covariances, rng)
    outcomes = observed_outcome(treatments, control_outcomes, treated_outcomes)
    return (covariates.astype(np.float32), treatments.astype(int), outcomes.astype(np.float32), observed_propensity.astype(np.float32), control_outcomes, treated_outcomes)


def make_scenario_text(environment, n, effect_strength, rng, pool):
    alpha = np.array([0.9, -0.75, 0.55, -0.4, 0.3])
    beta = np.array([0.6, -0.5, 0.4, -0.3, 0.2])
    noise_std = 0.1

    representation = "embeddings" if environment.endswith("_embeddings") else "tokens"
    name = environment.removesuffix("_embeddings").removesuffix("_tokens")
    (covariates, outcomes) = get_text_dataset(name, representation)

    indices = rng.choice(split_dataset(np.arange(len(covariates)), pool), size=n, replace=False)
    covariates = covariates[indices]
    (treatments, observed_propensity) = assign_treatment(covariates, alpha, rng)
    effect_strength = effect_strength * (2.0 * rng.binomial(n=1, p=0.5, size=n) - 1)
    (control_outcomes, treated_outcomes) = generate_scalar_outcome(covariates, beta, effect_strength, noise_std, rng)
    (control_outcomes, treated_outcomes) = (np.digitize(control_outcomes, [-1.0, 1.0]), np.digitize(treated_outcomes, [-1.0, 1.0]))
    (control_outcomes, treated_outcomes) = (outcomes[indices, control_outcomes], outcomes[indices, treated_outcomes])
    outcomes = observed_outcome(treatments, control_outcomes, treated_outcomes)
    return (covariates.astype(np.float32), treatments.astype(int), outcomes.astype(np.float32), observed_propensity.astype(np.float32), control_outcomes.astype(np.float32), treated_outcomes.astype(np.float32))


def make_scenario_graph(environment, n, scale, rng, pool):
    (covariates, control_outcomes, treated_outcomes, groups) = get_graph_dataset(environment, scale)
    indices = split_dataset(np.arange(len(covariates)), pool) if groups is None else np.flatnonzero(np.isin(groups, split_dataset(np.unique(groups), pool)))

    indices = rng.choice(indices, size=n, replace=False)
    (covariates, control_outcomes, treated_outcomes) = (covariates[indices, -1:], control_outcomes[indices], treated_outcomes[indices])
    alpha = np.zeros(covariates.shape[1])
    alpha[-1] = 1.0
    (treatments, observed_propensity) = assign_treatment(covariates, alpha, rng)
    outcomes = observed_outcome(treatments, control_outcomes, treated_outcomes)
    return (covariates, treatments, outcomes, observed_propensity, control_outcomes, treated_outcomes)


def make_scenario(environment, n, scale, rng, pool="train"):
    if environment == "fake_mnist":
        return make_scenario_fake_mnist(n, scale, rng, pool)
    elif environment in {"breastmnist", "chestmnist_cardiomegaly", "chestmnist_effusion", "chestmnist_mass", "dermamnist", "octmnist_cnv", "octmnist_dme", "octmnist_drusen", "pneumoniamnist", "retinamnist_severe"}:
        return make_scenario_causal_medmnist(environment, n, scale, rng, pool)
    elif environment == "blob":
        return make_scenario_blob(n, scale, rng)
    elif environment in {"oncology_notes", "oncology_notes_tokens", "oncology_notes_embeddings", "movie_reviews", "movie_reviews_tokens", "movie_reviews_embeddings", "clinical_notes", "clinical_notes_tokens", "clinical_notes_embeddings"}:
        return make_scenario_text(environment, n, scale, rng, pool)
    elif environment in {"code_llm_repair", "code_llm_repair_gaussian", "code_llm_repair_wl"}:
        return make_scenario_graph("code_graph_wl" if environment.endswith("_wl") else "code_graph_gaussian", n, scale, rng, pool)
    elif environment.startswith("contact_graph_"):
        return make_scenario_graph(environment, n, scale, rng, pool)

    return make_scenario_default(environment, n, scale, rng, pool)
