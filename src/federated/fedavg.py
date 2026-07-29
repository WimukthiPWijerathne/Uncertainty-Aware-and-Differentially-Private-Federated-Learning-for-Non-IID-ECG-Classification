"""
Sample-weighted Federated Averaging.
"""

from __future__ import annotations

import numpy as np


def aggregate_fedavg(
    client_results: list[
        tuple[list[np.ndarray], int]
    ],
) -> list[np.ndarray]:
    """
    Aggregate client model parameters according to
    each client's number of local training records.
    """
    if not client_results:
        raise ValueError(
            "No client results were supplied."
        )

    total_samples = sum(
        sample_count
        for _, sample_count in client_results
    )

    if total_samples <= 0:
        raise ValueError(
            "Total sample count must be positive."
        )

    reference_parameters = client_results[0][0]

    aggregated_parameters = [
        np.zeros_like(
            parameter,
            dtype=np.float64,
        )
        for parameter in reference_parameters
    ]

    for parameters, sample_count in client_results:
        if len(parameters) != len(
            reference_parameters
        ):
            raise ValueError(
                "Clients returned different "
                "numbers of parameters."
            )

        client_weight = (
            sample_count / total_samples
        )

        for index, parameter in enumerate(
            parameters
        ):
            if (
                parameter.shape
                != reference_parameters[index].shape
            ):
                raise ValueError(
                    "Client parameter shapes do not match."
                )

            aggregated_parameters[index] += (
                parameter.astype(
                    np.float64,
                    copy=False,
                )
                * client_weight
            )

    return [
        aggregated.astype(
            reference.dtype,
            copy=False,
        )
        for aggregated, reference in zip(
            aggregated_parameters,
            reference_parameters,
        )
    ]