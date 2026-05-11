# Tell Me Why

Librairie nbdev/fastai pour greffer un classifieur fourni par un développeur sur
ton AAE `modelAAE_DROPOUT.py`, puis inspecter l'espace latent gaussien et les
features produites.

La source officielle du modèle est la branche `Arda` du repo
[`LucaLaFisca/Human-Centered-xAI`](https://github.com/LucaLaFisca/Human-Centered-xAI/tree/Arda).
La librairie cherche par défaut le clone local frère:

```text
../Human-Centered-xAI/modelAAE_DROPOUT.py
```

Ce fichier définit `AAE.forward`: il encode l'image dans `self.zi`, puis appelle
`self.linear(self.zi)`. La greffe consiste donc à remplacer proprement
`model.linear` par le classifieur du développeur, tout en gardant le `Learner`
fastai, la reconstruction, le discriminateur latent et les callbacks possibles.

## Installation développeur

```bash
conda env create -f environment.yml
conda activate tell-me-why
python -m pip install -e ".[dev]"
```

L'environnement est aligné sur `Human-Centered-xAI/env_mac.yml`: Python 3.10,
fastai 2.7, PyTorch/torchvision, `pytorch-msssim` et nbdev.

Commandes nbdev utiles:

```bash
nbdev-export
nbdev-prepare
nbdev-test
nbdev-preview
```

Si le repo `Human-Centered-xAI` n'est pas déjà cloné à côté:

```bash
git clone --branch Arda --single-branch \
  https://github.com/LucaLaFisca/Human-Centered-xAI.git \
  ../Human-Centered-xAI
```

Ou depuis Python:

```python
from tell_me_why import ensure_human_centered_xai_repo

ensure_human_centered_xai_repo()
```

## Greffer un classifieur fastai

```python
from torch import nn

from tell_me_why import HumanCenteredAAEConfig, make_grafted_learner

classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),
)

learn = make_grafted_learner(
    dls,
    classifier,
    config=HumanCenteredAAEConfig(encoding_dims=128, classes=2),
    pretrained_weights="CL_AAE_model_128",
    freeze_aae_body=True,
)

learn.fine_tune(3)
```

Si le dossier `Human-Centered-xAI` n'est pas le dossier frère de ce repo:

```python
learn = make_grafted_learner(
    dls,
    classifier,
    model_path="/chemin/vers/Human-Centered-xAI/modelAAE_DROPOUT.py",
)
```

## Visualiser l'espace latent

```python
from tell_me_why import extract_latents_from_learner, plot_latent_gaussian

latent_batch = extract_latents_from_learner(learn, dl=test_dl)
fig, ax = plot_latent_gaussian(
    latent_batch.latents,
    labels=latent_batch.targets,
    projection="first2",
)
```

Pour un latent de grande dimension, `projection="pca"` donne une vue 2D rapide.

## Visualiser des features

```python
from tell_me_why import extract_human_aae_features, plot_feature_grid, plot_feature_distributions

features = extract_human_aae_features(learn.model, xb)

fig, axes = plot_feature_grid(features.encoder_features, n=12, channel=0)
fig, ax = plot_feature_distributions(features.latents, max_features=8)
```

## Structure nbdev

Règle de travail: on modifie d'abord les notebooks dans `nbs/`, puis on lance
`nbdev-export`. Les fichiers `tell_me_why/*.py` sont générés et ne doivent pas
être édités directement.

- `nbs/00_source.ipynb`: source officielle GitHub, branche `Arda`, import dynamique du modèle.
- `nbs/01_config.ipynb`: dataclasses de configuration et résultats.
- `nbs/02_tensors.ipynb`: helpers de conversion et déplacement des tenseurs.
- `nbs/03_adapters.ipynb`: adapters PyTorch génériques.
- `nbs/04_graft.ipynb`: greffe du classifieur et création du `Learner` fastai.
- `nbs/05_callbacks.ipynb`: callbacks fastai pour collecter `zi`.
- `nbs/06_latent.ipynb`: extraction de l'espace latent.
- `nbs/07_features.ipynb`: extraction des features internes de l'AAE.
- `nbs/08_plots.ipynb`: visualisations latent/features.
- `nbs/09_tutorial.ipynb`: exemple d'utilisation sans export.
- `nbs/90_core_compat.ipynb`, `91_fastai_aae_compat.ipynb`, `92_visualization_compat.ipynb`: façades de compatibilité exportées.
- `nbs/99_init.ipynb`: API publique exportée dans `tell_me_why/__init__.py`.
- `tell_me_why/`: paquet Python exporté par nbdev depuis les notebooks.
- `tests/`: tests rapides de l'API publique.

Cette organisation suit le principe visible dans FasterAI de Nathan Hubens:
un module court par responsabilité, puis une API publique simple ré-exportée à
la racine du package.
