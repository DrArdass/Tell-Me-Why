import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tell_me_why import AAEAdapter, GraftedAAEClassifier, collect_latents


class TinyAAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 2)

    def encode(self, inputs):
        return self.encoder(inputs)


def test_grafted_classifier_returns_logits_and_latents():
    model = GraftedAAEClassifier(TinyAAE(), nn.Linear(2, 3))

    logits, latents = model(torch.randn(5, 4), return_latent=True)

    assert logits.shape == (5, 3)
    assert latents.shape == (5, 2)


def test_adapter_can_use_encoder_attribute():
    class EncoderOnlyAAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 2)

    adapter = AAEAdapter(EncoderOnlyAAE(), encoder_attr="encoder")

    assert adapter(torch.randn(5, 4)).shape == (5, 2)


def test_collect_latents_from_dataloader():
    dataset = TensorDataset(torch.randn(10, 4), torch.arange(10))
    dataloader = DataLoader(dataset, batch_size=5)

    batch = collect_latents(AAEAdapter(TinyAAE()), dataloader)

    assert batch.latents.shape == (10, 2)
    assert batch.labels.shape == (10,)
