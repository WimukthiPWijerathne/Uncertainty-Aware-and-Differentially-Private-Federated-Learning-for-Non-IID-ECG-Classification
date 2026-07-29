"""
Utilities for converting PyTorch model parameters
to and from NumPy arrays.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn


def get_model_parameters(
    model: nn.Module,
) -> list[np.ndarray]:
    """Return model state as NumPy arrays."""
    return [
        tensor.detach().cpu().numpy().copy()
        for tensor in model.state_dict().values()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: list[np.ndarray],
) -> None:
    """Load NumPy arrays into a PyTorch model."""
    current_state = model.state_dict()

    if len(parameters) != len(current_state):
        raise ValueError(
            "Parameter count mismatch: "
            f"received {len(parameters)}, "
            f"expected {len(current_state)}."
        )

    new_state = OrderedDict()

    for (
        parameter_name,
        current_tensor,
    ), parameter_array in zip(
        current_state.items(),
        parameters,
    ):
        new_tensor = torch.from_numpy(
            np.asarray(parameter_array)
        ).to(
            device=current_tensor.device,
            dtype=current_tensor.dtype,
        )

        if new_tensor.shape != current_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {parameter_name}: "
                f"received {tuple(new_tensor.shape)}, "
                f"expected {tuple(current_tensor.shape)}."
            )

        new_state[parameter_name] = new_tensor

    model.load_state_dict(
        new_state,
        strict=True,
    )