import torch
from torch import nn

from tell_me_why import (
    HUMAN_CENTERED_XAI_BRANCH,
    HUMAN_CENTERED_XAI_REPO_URL,
    ensure_human_centered_xai_repo,
    graft_classifier_to_human_aae,
    resolve_human_centered_aae_path,
)


class TinyHumanAAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 3)
        self.linear = nn.Linear(3, 2)
        self.classes = 2

    def forward(self, inputs):
        self.zi = self.body(inputs)
        return self.linear(self.zi)


def test_resolve_human_centered_aae_path_accepts_explicit_file(tmp_path):
    model_path = tmp_path / "modelAAE_DROPOUT.py"
    model_path.write_text("class AAE: pass\n")

    path = resolve_human_centered_aae_path(model_path)

    assert path == model_path.resolve()
    assert path.name == "modelAAE_DROPOUT.py"


def test_resolve_human_centered_aae_path_accepts_repo_dir_env(tmp_path, monkeypatch):
    repo_dir = tmp_path / "Human-Centered-xAI"
    repo_dir.mkdir()
    model_path = repo_dir / "modelAAE_DROPOUT.py"
    model_path.write_text("class AAE: pass\n")
    monkeypatch.setenv("HUMAN_CENTERED_XAI_REPO_DIR", str(repo_dir))

    assert resolve_human_centered_aae_path() == model_path.resolve()


def test_source_metadata_points_to_official_arda_branch():
    assert HUMAN_CENTERED_XAI_REPO_URL == "https://github.com/LucaLaFisca/Human-Centered-xAI.git"
    assert HUMAN_CENTERED_XAI_BRANCH == "Arda"


def test_ensure_human_centered_xai_repo_returns_existing_directory(tmp_path):
    repo_dir = tmp_path / "Human-Centered-xAI"
    repo_dir.mkdir()

    assert ensure_human_centered_xai_repo(repo_dir) == repo_dir.resolve()


def test_graft_replaces_fastai_aae_linear_head_and_freezes_body():
    aae = TinyHumanAAE()
    classifier = nn.Linear(3, 4)

    grafted = graft_classifier_to_human_aae(aae, classifier, freeze_aae_body=True)
    output = grafted(torch.randn(2, 4))

    assert grafted.linear is classifier
    assert grafted.classes == 4
    assert output.shape == (2, 4)
    assert not grafted.body.weight.requires_grad
    assert grafted.linear.weight.requires_grad
