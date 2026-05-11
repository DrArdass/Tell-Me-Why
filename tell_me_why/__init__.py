"""Tools to graft classifiers onto an adversarial autoencoder."""

__version__ = "0.0.1"

from .core import AAEAdapter, GraftedAAEClassifier, LatentBatch, collect_latents, freeze_module
from .fastai_aae import (
    FastaiLatentResult,
    HUMAN_CENTERED_XAI_BRANCH,
    HUMAN_CENTERED_XAI_REPO_URL,
    HumanCenteredAAEConfig,
    HumanCenteredFeatures,
    build_human_centered_aae,
    ensure_human_centered_xai_repo,
    extract_human_aae_features,
    extract_latents_from_learner,
    graft_classifier_to_human_aae,
    load_human_centered_aae_module,
    make_grafted_learner,
    resolve_human_centered_aae_path,
)
from .visualization import (
    plot_feature_distributions,
    plot_feature_grid,
    plot_latent_gaussian,
)

__all__ = [
    "AAEAdapter",
    "GraftedAAEClassifier",
    "FastaiLatentResult",
    "HUMAN_CENTERED_XAI_BRANCH",
    "HUMAN_CENTERED_XAI_REPO_URL",
    "HumanCenteredAAEConfig",
    "HumanCenteredFeatures",
    "LatentBatch",
    "build_human_centered_aae",
    "collect_latents",
    "ensure_human_centered_xai_repo",
    "extract_human_aae_features",
    "extract_latents_from_learner",
    "freeze_module",
    "graft_classifier_to_human_aae",
    "load_human_centered_aae_module",
    "make_grafted_learner",
    "plot_feature_distributions",
    "plot_feature_grid",
    "plot_latent_gaussian",
    "resolve_human_centered_aae_path",
]
