from typing import List

import numpy as np
import torch


def batch_to_multihot(label: List[List[int]], num_labels: int) -> torch.tensor:
    """Converts label to multihot format.

    Args:
        label: [batch size, *]
        num_labels: total number of labels

    Returns:
        multihot: [batch size, num_labels]
    """
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def multihot_to_indices(multihot: np.ndarray) -> List[int]:
    """Converts multihot format back to indices.

    Args:
        multihot: [num_labels]

    Returns:
        indices: [*]
    """
    indices = np.nonzero(multihot)[0].tolist()  # 获取值为1的索引
    return indices

def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        last_visit_indices = torch.sum(mask.long(), 1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_visit_indices]
