from typing import List

import numpy as np


# TODO: this metric is very ad-hoc, need to be improved


def ddi_rate_score(medications: List[np.ndarray], ddi_matrix: np.ndarray) -> float:
    """DDI rate score.

    Args:
        medications: list of medications for each patient, where each medication
            is represented by the corresponding index in the ddi matrix.
        ddi_matrix: array-like of shape (n_classes, n_classes).

    Returns:
        result: DDI rate score.
    """
    all_cnt = 0
    ddi_cnt = 0
    for sample in medications:
        for i, med_i in enumerate(sample):
            for j, med_j in enumerate(sample):
                if j <= i: continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    ddi_cnt += 1
    if all_cnt == 0:
        return 0
    return ddi_cnt / all_cnt


def ddi_rate_score_per_sample(medications: List[np.ndarray], ddi_matrix: np.ndarray) -> List[float]:
    """DDI rate score for each sample.

    Args:
        medications: list of medications for each patient, where each medication
            is represented by the corresponding index in the ddi matrix.
        ddi_matrix: array-like of shape (n_classes, n_classes).

    Returns:
        result: List of DDI rate scores for each sample.
    """
    results = []
    for sample in medications:
        all_cnt = 0
        ddi_cnt = 0
        for i, med_i in enumerate(sample):
            for j, med_j in enumerate(sample):
                if j <= i: continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    ddi_cnt += 1
        if all_cnt == 0:
            results.append(0)
        else:
            results.append(ddi_cnt / all_cnt)
    return results