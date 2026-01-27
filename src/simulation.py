import numpy as np


def flip_top_positives(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    p: float
) -> np.ndarray:
    """
    Flip p proportion of positive outcomes (1 -> 0), targeting those with
    highest predicted probabilities first.
    """
    outcomes_modified = outcomes.copy()

    # Find indices where outcome is 1
    pos_indices = np.where(outcomes == 1)[0]

    if len(pos_indices) == 0:
        return outcomes_modified

    # Get predictions for positives and rank them
    pos_predictions = predictions[pos_indices]

    # Number to flip
    n_to_flip = int(np.ceil(len(pos_indices) * p))

    if n_to_flip == 0:
        return outcomes_modified

    # Get indices of top predictions among positives (highest first)
    top_pos_indices = pos_indices[np.argsort(pos_predictions)[-n_to_flip:]]

    # Flip those outcomes (1 -> 0)
    outcomes_modified[top_pos_indices] = 0

    return outcomes_modified
