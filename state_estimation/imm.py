"""This module implements the Interacting Multiple Model (IMM) filter equations.

Throughout this module, "r" represents the number of models,
"n" represents the dimension of the state vector,
and "m" represents the dimension of the measurement vector.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Vec: TypeAlias = npt.NDArray[np.float64]


def compute_mixing_probabilities(
    previous_model_probs: Vec,
    mode_transition_probs: Vec,
) -> Vec:
    """Compute the mixing probabilities.
    
    Args:
        previous_model_probs (Vec): Previous model probabilities (r vector)
        mode_transition_probs (Vec): Mode transition probability matrix (r x r matrix). Row idx = prev model, 
            col idx = current model.
    
    Returns:
        Vec: Mixing probabilities for each model (r x r matrix). Row idx = prev model, col idx = current model.
    """

def compute_mixed_initial_conditions(
    previous_state_estimates: Vec,
    previous_covariances: Vec,
    mixing_probs: Vec,
) -> tuple[Vec, Vec]:
    """Compute the mixed initial conditions for each model.

    Args:
        previous_state_estimates (Vec): Previous state estimates (n x r matrix)
        previous_covariances (Vec): Previous state estimate covariance matrices (n x n x r matrix)
        mixing_probs (Vec): Mixing probabilities (r x r matrix)
    
    Returns:
        mixed_state_estimates (Vec): Mixed state estimates (n x r matrix)
        mixed_covariances (Vec): Mixed state estimate covariance matrices (n x n x r matrix)
    """

def kalman_filter_update(
    state_transition_matrix: Vec,
    process_noise_covariance: Vec,
    measurement_matrix: Vec,
    measurement_covariance: Vec,
    measurement: Vec,
    state_estimate: Vec,
    state_estimate_covariance: Vec,
) -> tuple[Vec, Vec, float]:
    """Perform a Kalman filter update for a single model.

    Args:
        state_transition_matrix (Vec): State transition matrix (n x n)
        process_noise_covariance (Vec): Process noise covariance matrix (n x n)
        measurement_matrix (Vec): Measurement matrix (m x n)
        measurement_covariance (Vec): Measurement noise covariance matrix (m x m)
        measurement (Vec): Measurement (m vector)
        state_estimate (Vec): State estimate (n vector)
        state_estimate_covariance (Vec): State estimate covariance (n x n matrix)

    Returns:
        state_estimate (Vec): The updated state estimate (n vector).
        covariance (Vec): The updated state estimate covariance (n x n matrix).
        likelihood (float): The likelihood function value (scalar).
    """

def update_model_probabilities(
    previous_model_probs: Vec,
    mode_transition_probs: Vec,
    likelihoods: Vec
) -> Vec:
    """Update the model probabilities based on the previous model probabilities, mode transition probabilities, and likelihoods.

    Args:
        previous_model_probs (Vec): Previous model probabilities (r vector)
        mode_transition_probs (Vec): Mode transition probability matrix (r x r matrix)
        likelihoods (Vec): Likelihood function values (r vector)

    Returns:
        next_model_probs (Vec): Next model probabilities (r vector)
    """

def combine_state_estimates(
    state_estimates: Vec,
    covariances: Vec,
    model_probs: Vec
) -> tuple[Vec, Vec]:
    """Combine state estimates and covariances based on model probabilities.

    Args:
        state_estimates (Vec): State estimate for each model (n x r matrix)
        covariances (Vec): Covariance for each model (n x n x r matrix)
        model_probs (Vec): Model probabilities (r vector)

    Returns:
        combined_state_estimate (Vec): Combined state estimate (n vector)
        combined_covariance (Vec): Combined covariance (n x n matrix)
    """

def imm_update():
    """
    Args:
        Previous state estimates (n x r matrix)
        Previous state estimate covariance matrices (n x n x r matrix)
        mode transition probability matrix (r x r matrix)
        Previous model probabilities (r vector)
        Measurement (m vector)
    Returns:
        State estimate for each model (n x r matrix)
        Covariance for each model (n x n x r matrix)
        Updated model probabilities (r vector)
        Combined state estimate (n vector)
        Combined covariance (n x n matrix)
    """
