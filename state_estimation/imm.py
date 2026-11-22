"""This module implements the Interacting Multiple Model (IMM) filter equations.

Throughout this module, "r" represents the number of models,
"n" represents the dimension of the state vector,
and "m" represents the dimension of the measurement vector.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


Vec: TypeAlias = npt.NDArray[np.float64]


@dataclass
class ImmResult:
    """Dataclass to hold the results of the IMM update.
    
    Attributes:
        states (Vec): State estimate for each model (n x r matrix)
        covariances (Vec): Covariance of the state estimate for each model (n x n x r matrix)
        model_probabilities (Vec): Updated model probabilities (r vector)
        combined_state (Vec): Combined state estimate (n vector)
        combined_covariance (Vec): Combined covariance (n x n matrix)
    """
    states: Vec
    covariances: Vec
    model_probabilities: Vec
    combined_state: Vec
    combined_covariance: Vec


def _validate_array_shape(array: Vec, desired_dim: list[int], label: str = "array") -> None:
    """Validate the dimensions of a numpy array.

    Args:
        array (Vec): The array to validate.
        desired_dim (list[int]): The desired dimensions of the array.
        label (str, optional): A label for the array. Defaults to "array".

    Raises:
        ValueError: If the dimensions of the array do not match the desired dimensions.
    """
    if array.shape != tuple(desired_dim):
        raise ValueError(f"Invalid shape for {label}! Got {array.shape}, expected {tuple(desired_dim)}")


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
    _validate_array_shape(mode_transition_probs, [r, r], "mode transition probability matrix")

    # Compute normalization factors for each model
    normalization_factors = np.zeros(r)
    for j in range(r):  # Current model idx
        normalization_factors[j] = np.dot(previous_model_probs, mode_transition_probs[:, j])

    # Compute mixing probabilities
    mixing_probs = np.zeros((r, r))
    for i in range(r):  # Previous model idx
        for j in range(r):  # Current model idx
            mixing_probs[i, j] = mode_transition_probs[i, j] * previous_model_probs[i] / normalization_factors[j]

    return mixing_probs


def compute_mixed_initial_conditions(
    initial_states: Vec,
    initial_covariances: Vec,
    mixing_probs: Vec,
) -> tuple[Vec, Vec]:
    """Compute the mixed initial state and covariance for each model.

    Args:
        initial_states (Vec): Previous state estimates (n x r matrix)
        initial_covariances (Vec): Previous state estimate covariances (n x n x r matrix)
        mixing_probs (Vec): Mixing probabilities (r x r matrix)

    Returns:
        mixed_initial_states (Vec): Mixed initial state (n x r matrix)
        mixed_initial_covariances (Vec): Mixed initial covariance matrices (n x n x r matrix)
    """
    # Input validation
    (n, r) = initial_states.shape
    _validate_array_shape(initial_covariances, [n, n, r], "initial covariances")
    _validate_array_shape(mixing_probs, [r, r], "mixing probabilities")

    # Computed mixed initial states
    mixed_initial_states = np.zeros((n, r))
    for j in range(r):  # Current model
        sum = np.zeros(n)
        for i in range(r):  # Previous model
            sum += initial_states[:, i] * mixing_probs[i, j]
        mixed_initial_states[:, j] = sum

    # Compute mixed initial covariances
    mixed_initial_covariances = np.zeros((n, n, r))
    for j in range(r):  # current model
        sum = np.zeros((n, n))
        for i in range(r):  # previous model
            diff = initial_states[:, i] - mixed_initial_states[:, j]
            sum += mixing_probs[i, j] * (initial_covariances[:, :, i] + np.outer(diff, diff))
        mixed_initial_covariances[:, :, j] = sum

    return mixed_initial_states, mixed_initial_covariances


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
    # Input validation
    n = state_transition_matrix.shape[0]
    m = measurement_matrix.shape[0]
    _validate_array_shape(state_transition_matrix, [n, n], "state transition matrix")
    _validate_array_shape(process_noise_covariance, [n, n], "process noise covariance")
    _validate_array_shape(measurement_matrix, [m, n], "measurement matrix")
    _validate_array_shape(measurement_covariance, [m, m], "measurement covariance")
    _validate_array_shape(measurement, [m], "measurement")
    _validate_array_shape(initial_state, [n], "initial state")
    _validate_array_shape(initial_covariance, [n, n], "initial covariance")

    # State prediction
    predicted_state = state_transition_matrix @ initial_state
    predicted_covariance = (
        state_transition_matrix @ initial_covariance @ state_transition_matrix.T
        + process_noise_covariance
    )

    # Measurement update
    innovation = measurement - measurement_matrix @ predicted_state
    innovation_covariance = measurement_matrix @ predicted_covariance @ measurement_matrix.T + measurement_covariance
    kalman_gain = predicted_covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
    updated_state = predicted_state + kalman_gain @ innovation
    I = np.eye(n)
    updated_covariance = (
        (I - kalman_gain @ measurement_matrix) @ predicted_covariance @ (I - kalman_gain @ measurement_matrix).T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )

    # Likelihood function value
    likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(m), cov=innovation_covariance)

    return updated_state, updated_covariance, likelihood


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
    # Input validation
    r = previous_model_probs.shape[0]
    _validate_array_shape(mode_transition_probs, [r, r], "mode transition probability matrix")
    _validate_array_shape(likelihoods, [r], "likelihoods")

    # Calculate normalization factors for each model
    normalization_factors = np.zeros(r)
    for j in range(r):  # Current model idx
        normalization_factors[j] = np.dot(previous_model_probs, mode_transition_probs[:, j])

    # Update the model probabilities
    next_model_probs = normalization_factors * likelihoods / np.dot(normalization_factors, likelihoods)

    return next_model_probs


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
    # Input validation
    (n, r) = state_estimates.shape
    _validate_array_shape(covariances, [n, n, r], "covariances")
    _validate_array_shape(model_probs, [r], "model probabilities")

    # Compute combined state estimate
    combined_state_estimate = np.zeros(n)
    for j in range(r):  # Current model idx
        combined_state_estimate += model_probs[j] * state_estimates[:, j]

    # Compute combined covariance
    combined_covariance = np.zeros((n, n))
    for j in range(r):  # Current model idx
        diff = state_estimates[:, j] - combined_state_estimate
        combined_covariance += model_probs[j] * (covariances[:, :, j] + np.outer(diff, diff))

    return combined_state_estimate, combined_covariance


def imm_update(
    initial_states: Vec,
    initial_covariances: Vec,
    mode_transition_probs: Vec,
    previous_model_probs: Vec,
    measurement: Vec,
    state_transition_matrices: Vec,
    process_noise_covariances: Vec,
    measurement_matrices: Vec,
    measurement_covariances: Vec,
) -> ImmResult:
    """Perform a full update of the interacting multiple model filter (IMM).

    Args:
        initial_states (Vec): Previous state estimates (n x r matrix)
        initial_covariances (Vec): Previous state estimate covariance matrices (n x n x r matrix)
        mode_transition_probs (Vec): Mode transition probability matrix (r x r matrix)
        previous_model_probs (Vec): Previous model probabilities (r vector)
        measurement (Vec): Measurement (m vector)
        state_transition_matrices (Vec): State transition matrices for all models (n x n x r)
        process_noise_covariances (Vec): Process noise covariance matrices for all models (n x n x r)
        measurement_matrices (Vec): Measurement matrices for all models (m x n x r)
        measurement_covariances (Vec): Measurement noise covariances for all models (m x m x r)
    
    Returns:
        result (ImmResult): A dataclass containing:
            - State estimate for each model (n x r matrix)
            - Covariance for each model (n x n x r matrix)
            - Updated model probabilities (r vector)
            - Combined state estimate (n vector)
            - Combined covariance (n x n matrix)
    """
    # Input validation
    (n, r) = initial_states.shape
    (m,) = measurement.shape
    _validate_array_shape(initial_covariances, [n, n, r], "initial covariances")
    _validate_array_shape(mode_transition_probs, [r, r], "mode transition probabilities")
    _validate_array_shape(previous_model_probs, [r], "previous model probabilities")
    _validate_array_shape(measurement, [m], "measurement")
    _validate_array_shape(state_transition_matrices, [n, n, r], "state transition matrices")
    _validate_array_shape(process_noise_covariances, [n, n, r], "process noise covariances")
    _validate_array_shape(measurement_matrices, [m, n, r], "measurement matrices")
    _validate_array_shape(measurement_covariances, [m, m, r], "measurement covariances")

    # Step 1: Compute mixing probabilities
    mixing_probs = compute_mixing_probabilities(previous_model_probs, mode_transition_probs)

    # Step 2: Compute mixed initial conditions
    mixed_initial_states, mixed_initial_covariances = compute_mixed_initial_conditions(
        initial_states, initial_covariances, mixing_probs
    )

    # Step 3: Perform a Kalman filter update for each model
    updated_states = np.zeros((n, r))
    updated_covariances = np.zeros((n, n, r))
    likelihoods = np.zeros(r)
    for j in range(r):  # Current model idx
        updated_states[:, j], updated_covariances[:, :, j], likelihoods[j] = kalman_filter_update(
            state_transition_matrices[:, :, j],
            process_noise_covariances[:, :, j],
            measurement_matrices[:, :, j],
            measurement_covariances[:, :, j],
            measurement,
            mixed_initial_states[:, j],
            mixed_initial_covariances[:, :, j]
        )

    # Step 4: Update model probabilities
    updated_model_probs = update_model_probabilities(previous_model_probs, mode_transition_probs, likelihoods)

    # Step 5: Combine state estimates and covariances
    combined_state, combined_covariance = combine_state_estimates(updated_states, updated_covariances, updated_model_probs)

    return ImmResult(
        states=updated_states,
        covariances=updated_covariances,
        model_probabilities=updated_model_probs,
        combined_state=combined_state,
        combined_covariance=combined_covariance
    )
