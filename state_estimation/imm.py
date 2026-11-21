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
    # Input validation
    r = previous_model_probs.shape[0]
    if mode_transition_probs.shape != (r, r):
        raise ValueError(f"Invalid shape for mode transition probability matrix! Got {mode_transition_probs.shape}, expected ({r, r})")

    # Compute normalization constants
    normalization_consts = np.zeros(r)
    for j in range(r):  # Current model idx
        normalization_consts[j] = np.dot(previous_model_probs, mode_transition_probs[:, j])

    # Compute mixing probabilities
    mixing_probs = np.zeros((r, r))
    for i in range(r):  # Previous model idx
        for j in range(r):  # Current model idx
            mixing_probs[i, j] = mode_transition_probs[i, j] * previous_model_probs[i]

    return mixing_probs


def compute_mixed_initial_states(
    initial_states: Vec,
    mixing_probs: Vec,
) -> Vec:
    """Compute the mixed initial state for each model.

    Args:
        initial_states (Vec): Previous state estimates (n x r matrix)
        mixing_probs (Vec): Mixing probabilities (r x r matrix)
    
    Returns:
        mixed_initial_states (Vec): Mixed initial state (n x r matrix)
    """
    # Input validation
    (n, r) = initial_states.shape
    if mixing_probs.shape != (r, r):
        raise ValueError(f"Invalid shape for mixing probabilities! Got {mixing_probs.shape}, expected ({r, r})")

    # Computed mixed initial states
    mixed_initial_states = np.zeros((n, r))
    for j in range(r):  # Current model
        sum = np.zeros(n)
        for i in range(r):  # Previous model
            sum += initial_states[:, i] * mixing_probs[i, j]
        mixed_initial_states[:, j] = sum

    return mixed_initial_states


def compute_mixed_initial_covariances(
    initial_states: Vec,
    mixed_initial_states: Vec,
    initial_covariances: Vec,
    mixing_probs: Vec,
) -> Vec:
    """Compute the mixed initial covariances of each model.

    Args:
        initial_states (Vec): Previous state estimates (n x r matrix)
        mixed_initial_states (Vec): Mixed initial states for each model (n x r matrix)
        initial_covariances (Vec): Previous state estimate covariances (n x n x r matrix)
        mixing_probs (Vec): Mixing probabilities (r x r matrix)
    
    Returns:
        mixed_initial_covariances (Vec): Mixed initial covariance matrices (n x n x r matrix)
    """
    # Input validation
    (n, r) = mixed_initial_states.shape
    if initial_covariances.shape != (n, n, r):
        raise ValueError(f"Invalid shape for initial covariances! Got {initial_covariances.shape}, expected ({n, n, r})")
    if mixing_probs.shape != (r, r):
        raise ValueError(f"Invalid shape for mixing probabilities! Got {mixing_probs.shape}, expected ({r, r})")

    # Compute mixed initial covariances
    mixed_initial_covariances = np.zeros((n, n, r))
    for j in range(r):  # current model
        sum = np.zeros((n, n))
        for i in range(r):  # previous model
            diff = initial_states[:, i] - mixed_initial_states[:, j]
            sum += mixing_probs[i, j] * (initial_covariances[:, :, i] + np.outer(diff, diff))
        mixed_initial_covariances[:, :, j] = sum

    return mixed_initial_covariances


def kalman_filter_update(
    state_transition_matrix: Vec,
    process_noise_covariance: Vec,
    measurement_matrix: Vec,
    measurement_covariance: Vec,
    measurement: Vec,
    initial_state: Vec,
    initial_covariance: Vec,
) -> tuple[Vec, Vec, float]:
    """Perform a Kalman filter update for a single model.

    Args:
        state_transition_matrix (Vec): State transition matrix (n x n)
        process_noise_covariance (Vec): Process noise covariance matrix (n x n)
        measurement_matrix (Vec): Measurement matrix (m x n)
        measurement_covariance (Vec): Measurement noise covariance matrix (m x m)
        measurement (Vec): Measurement (m vector)
        initial_state (Vec): Initial state (n vector)
        initial_covariance (Vec): Initial covariance (n x n matrix)

    Returns:
        updated_state (Vec): The updated state estimate (n vector).
        updated_covariance (Vec): The updated state estimate covariance (n x n matrix).
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
