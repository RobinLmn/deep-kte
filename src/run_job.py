import os
import sys

import numpy as np

import experiments
from experiments import ExperimentConfig, run_sample_sweep, run_scale_sweep

JOBS = []


def job(name, kind, run):
    JOBS.append((name, kind, run))


# Medical Image Outcomes
job("breastmnist_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("breastmnist", "cnn"), [1500], scale=0.0))
job("breastmnist_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("breastmnist", "cnn"), [1500], scale=0.7))
job("chestmnist_cardiomegaly_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_cardiomegaly", "cnn"), [1500], scale=0.0))
job("chestmnist_cardiomegaly_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_cardiomegaly", "cnn"), [1500], scale=1.4))
job("chestmnist_effusion_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_effusion", "cnn"), [1500], scale=0.0))
job("chestmnist_effusion_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_effusion", "cnn"), [1500], scale=1.2))
job("chestmnist_mass_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_mass", "cnn"), [1500], scale=0.0))
job("chestmnist_mass_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("chestmnist_mass", "cnn"), [1500], scale=1.0))
job("dermamnist_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("dermamnist", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=0.0))
job("dermamnist_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("dermamnist", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=1.4))
job("octmnist_cnv_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_cnv", "cnn"), [1500], scale=0.0))
job("octmnist_cnv_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_cnv", "cnn"), [1500], scale=0.8))
job("octmnist_dme_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_dme", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=0.0))
job("octmnist_dme_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_dme", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=0.4))
job("octmnist_drusen_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_drusen", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=0.0))
job("octmnist_drusen_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("octmnist_drusen", "cnn"), [500, 1000, 1500, 2000, 2500, 3000], scale=0.4))
job("pneumoniamnist_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("pneumoniamnist", "cnn"), [1500], scale=0.0))
job("pneumoniamnist_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("pneumoniamnist", "cnn"), [1500], scale=1.0))
job("retinamnist_severe_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("retinamnist_severe", "cnn"), [1500], scale=0.0))
job("retinamnist_severe_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("retinamnist_severe", "cnn"), [1500], scale=1.0))
job("octmnist_dme/scale", "gpu", lambda name: run_scale_sweep(name, ExperimentConfig("octmnist_dme", "cnn"), [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))
job("octmnist_drusen/scale", "gpu", lambda name: run_scale_sweep(name, ExperimentConfig("octmnist_drusen", "cnn"), [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))
job("dermamnist/scale", "gpu", lambda name: run_scale_sweep(name, ExperimentConfig("dermamnist", "cnn"), [1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))

# Text Outcomes
job("oncology_notes_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("oncology_notes", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1000, 1500, 2000, 2500, 3000, 3500], scale=0.0))
job("oncology_notes_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("oncology_notes", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1000, 1500, 2000, 2500, 3000, 3500], scale=1.0))
job("movie_reviews_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("movie_reviews", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1000, 1500, 2000, 2500, 3000, 3500], scale=0.0))
job("movie_reviews_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("movie_reviews", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1000, 1500, 2000, 2500, 3000, 3500], scale=1.0))
job("clinical_notes_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("clinical_notes", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1500], scale=0.0))
job("clinical_notes_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("clinical_notes", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1500], scale=2.6))
job("oncology_notes/scale", "gpu", lambda name: run_scale_sweep(name, ExperimentConfig("oncology_notes", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))
job("movie_reviews/scale", "gpu", lambda name: run_scale_sweep(name, ExperimentConfig("movie_reviews", "token_transformer", epochs=1000, feature_lr=0.001, lr=0.0001, distance_float64=True), [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]))

# Graph Outcomes
job("code_llm_repair_gaussian_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("code_llm_repair_gaussian", "gnn", epochs=1000), [1500, 2000], scale=0.0))
job("code_llm_repair_gaussian_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("code_llm_repair_gaussian", "gnn", epochs=1000), [1500, 2000], scale=1.0))
job("contact_graph_gaussian_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("contact_graph_gaussian", "gnn", epochs=1000), [500, 1000, 1500, 2000, 2500, 3000], scale=0.0))
job("contact_graph_gaussian_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("contact_graph_gaussian", "gnn", epochs=1000), [500, 1000, 1500, 2000, 2500, 3000], scale=0.2))

# Low Dimensional Outcomes
job("continuous_I/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("continuous_I", "linear", epochs=1000, ensemble_size=5), [1500, 3000], scale=0.0))
job("continuous_II/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("continuous_II", "linear", epochs=1000, ensemble_size=5), [1500, 3000], scale=0.8))
job("continuous_III/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("continuous_III", "linear", epochs=1000, ensemble_size=5), [1500, 3000], scale=0.8))
job("continuous_IV/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("continuous_IV", "linear", epochs=1000, ensemble_size=5), [1500, 3000], scale=0.8))
job("blob_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("blob", "linear", epochs=1000, bandwidth_q=0.05, lr=0.01, ensemble_size=5), [1500, 3000], scale=0.0))
job("blob_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("blob", "linear", epochs=1000, bandwidth_q=0.05, lr=0.01, ensemble_size=5), [1500, 3000], scale=1.0))

# Image Outcomes
job("mnist_I/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("mnist_I", "cnn"), [1500, 3000], scale=0.0))
job("mnist_II/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("mnist_II", "cnn"), [1500, 3000], scale=1.0))
job("mnist_III/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("mnist_III", "cnn"), [1500, 3000], scale=1.5))
job("retinamnist_I/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("retinamnist_I", "cnn"), [1500, 3000], scale=0.0))
job("retinamnist_II/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("retinamnist_II", "cnn"), [1500, 3000], scale=2.0))
job("retinamnist_III/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("retinamnist_III", "cnn"), [1500, 3000], scale=4.0))
job("busi_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("busi", "cnn"), [1500, 3000], scale=0.0))
job("busi_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("busi", "cnn"), [1500, 3000], scale=4.0))
job("fake_mnist_H0/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("fake_mnist", "cnn"), [1500, 3000], scale=0.0))
job("fake_mnist_H1/sample", "gpu", lambda name: run_sample_sweep(name, ExperimentConfig("fake_mnist", "cnn"), [1500, 3000], scale=4.0))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        for index, (name, kind, _) in enumerate(JOBS):
            print(f"{index}\t{name}\t{kind}")
        raise SystemExit(0)

    task_id = int(sys.argv[1])
    (name, _, run) = JOBS[task_id]
    seed_offset = int(os.environ.get("CNME_SEED_OFFSET", 0))
    experiments.rng = np.random.default_rng([task_id, seed_offset])
    run(name)
