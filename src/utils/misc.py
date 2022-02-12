import typing as tp

import numpy as np
import torch

from torch import nn


def cut_last_layers(
    model: nn.Module, 
    count_del_layer: tp.Optional[int] = 2,
) -> nn.Module:
    """
    Remove and pool layers 

    Args:
        last_layer_for_del (tp.Optional[int], optional): Count of last layers for delete from pretrained model
    """

    modules = list(model.children())[:-count_del_layer] 
    model = nn.Sequential(*modules) 
    return model


def freeze_layers(
    model: nn.Module, 
    count_freeze_layers: int,
) -> nn.Module:
    """
    Freeze first count_freeze_layers layers in model

    Args:
        model (nn.Module) model
        count_freeze_layers (int): Number of the last layers for which the gradient will not be considered
    """
    for p in model.parameters():
        p.requires_grad = False
    for c in list(model.children())[-count_freeze_layers:]:
        for p in c.parameters():
            p.requires_grad = True
    return model


def load_pretrained_embeddings(
    model_embedding: nn.Module, 
    embeddings: tp.Union[np.ndarray, torch.Tensor],
):
    """
    Loads embedding layer with pre-trained embeddings.

    Args:
        model_embedding (nn.Module): layer with embeddings
        embeddings (tp.Union[np.ndarray, torch.Tensor]): pre-trained embeddings
    """

    assert model_embedding.weight.shape == embeddings.shape, "Dimensions must be equal!"
    model_embedding.weight = nn.Parameter(embeddings)
