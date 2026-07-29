"""
Uncertainty-aware sample-weighted Federated Averaging.
"""

from __future__ import annotations

import math

import numpy as np


ClientResult = tuple[
    list[np.ndarray],
    int,
    float,
]


def aggregate_uncertainty_fedavg(
    client_results: list[ClientResult],
    epsilon: float = 1e-8,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[float],
]:
    """
    Aggregate client models using sample count and uncertainty.

    For client k:

        raw_weight_k = n_k / (u_k + epsilon)

        final_weight_k =
            raw_weight_k / sum(raw_weight_j)

    Args:
        client_results:
            List containing:
            (
                model_parameters,
                sample_count,
                uncertainty_score,
            )

        epsilon:
            Small positive number preventing division by zero.

    Returns:
        aggregated_parameters:
            The uncertainty-aware global parameters.

        normalized_weights:
            Final aggregation weight for each client.

        raw_weights:
            Unnormalized sample/uncertainty scores.
    """
    if not client_results:
        raise ValueError(
            "No client results were supplied."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    reference_parameters = client_results[0][0]

    raw_weights: list[float] = []

    for (
        parameters,
        sample_count,
        uncertainty,
    ) in client_results:
        if sample_count <= 0:
            raise ValueError(
                "Every client sample count must be positive."
            )

        if not math.isfinite(uncertainty):
            raise ValueError(
                "Client uncertainty must be finite."
            )

        if uncertainty < 0:
            raise ValueError(
                "Client uncertainty cannot be negative."
            )

        if len(parameters) != len(
            reference_parameters
        ):
            raise ValueError(
                "Clients returned different numbers "
                "of model tensors."
            )

        for index, parameter in enumerate(
            parameters
        ):
            if (
                parameter.shape
                != reference_parameters[index].shape
            ):
                raise ValueError(
                    "Client tensor shape mismatch at "
                    f"parameter index {index}."
                )

        raw_weight = (
            sample_count
            / (uncertainty + epsilon)
        )

        raw_weights.append(
            float(raw_weight)
        )

    total_raw_weight = sum(raw_weights)

    if (
        total_raw_weight <= 0
        or not math.isfinite(total_raw_weight)
    ):
        raise ValueError(
            "The total uncertainty-aware weight "
            "must be finite and positive."
        )

    normalized_weights = [
        raw_weight / total_raw_weight
        for raw_weight in raw_weights
    ]

    # Used for non-floating-point state tensors,
    # such as BatchNorm num_batches_tracked.
    largest_weight_client_index = int(
        np.argmax(normalized_weights)
    )

    aggregated_parameters: list[np.ndarray] = []

    for parameter_index, reference in enumerate(
        reference_parameters
    ):
        if np.issubdtype(
            reference.dtype,
            np.floating,
        ):
            aggregated = np.zeros_like(
                reference,
                dtype=np.float64,
            )

            for (
                parameters,
                _,
                _,
            ), normalized_weight in zip(
                client_results,
                normalized_weights,
            ):
                aggregated += (
                    parameters[
                        parameter_index
                    ].astype(
                        np.float64,
                        copy=False,
                    )
                    * normalized_weight
                )

            aggregated_parameters.append(
                aggregated.astype(
                    reference.dtype,
                    copy=False,
                )
            )

        else:
            selected_parameters = client_results[
                largest_weight_client_index
            ][0]

            aggregated_parameters.append(
                selected_parameters[
                    parameter_index
                ].copy()
            )

    return (
        aggregated_parameters,
        normalized_weights,
        raw_weights,
    )