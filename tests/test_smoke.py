import pytest
import torch.nn as nn

import tell_me_why
from tell_me_why.model_aae import AAE
from tell_me_why.user_encoder import EncoderWithAAEBlocks


def test_package_has_version():
    assert tell_me_why.__version__


def test_aae_rejects_non_binary_classes():
    with pytest.raises(AssertionError, match="binary image classification only"):
        AAE(classes=3)


def test_encoder_with_aae_blocks_rejects_non_binary_classes():
    encoder = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    with pytest.raises(AssertionError, match="binary image classification only"):
        EncoderWithAAEBlocks(encoder=encoder, classes=10)
