# Tell Me Why

Librairie nbdev/fastai pour greffer un classifieur fourni par un développeur sur
ton AAE `modelAAE_DROPOUT.py`, puis inspecter l'espace latent gaussien et les
features produites.

La librairie s'appuie explicitement sur le modèle situé ici par défaut:

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
nbdev_export
nbdev_test
nbdev_preview
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

- `nbs/`: notebooks source nbdev.
- `tell_me_why/fastai_aae.py`: intégration dédiée à `modelAAE_DROPOUT.py`.
- `tell_me_why/`: paquet Python exporté par nbdev.
- `tests/`: tests rapides de l'API publique.
