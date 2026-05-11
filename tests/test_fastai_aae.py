import torch
from torch import nn

from tell_me_why import graft_classifier_to_human_aae, resolve_human_centered_aae_path


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
