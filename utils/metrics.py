"""
Evaluation metrics for rhythm quantization.

We report:
  • Onset F1 / Precision / Recall  (frame-level, threshold-based)
  • Onset F1 with tolerance window  (±1 grid slot)
  • Hamming accuracy
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def frame_metrics(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute frame-level precision, recall, F1 for onset detection.

    Parameters
    ----------
    logits  : (N,) or (B, T)  raw logits (pre-sigmoid).
    targets : same shape, binary {0, 1}.
    threshold : sigmoid threshold for positive prediction.

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy
    """
    logits = _to_numpy(logits).flatten()
    targets = _to_numpy(targets).flatten()

    preds = (1.0 / (1.0 + np.exp(-logits))) >= threshold   # sigmoid + thresh
    targets = targets >= 0.5

    tp = np.sum(preds & targets).item()
    fp = np.sum(preds & ~targets).item()
    fn = np.sum(~preds & targets).item()
    tn = np.sum(~preds & ~targets).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def tolerant_f1(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    tolerance: int = 1,
) -> dict[str, float]:
    """Onset F1 with a ± tolerance window (in grid slots).

    A predicted onset is correct if there is a true onset within ±tolerance.
    """
    logits = _to_numpy(logits).flatten()
    targets = _to_numpy(targets).flatten()

    preds = (1.0 / (1.0 + np.exp(-logits))) >= threshold
    gt = targets >= 0.5

    pred_idxs = set(np.where(preds)[0].tolist())
    gt_idxs = set(np.where(gt)[0].tolist())

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    for p in sorted(pred_idxs):
        for offset in range(0, tolerance + 1):
            for sign in (0, 1, -1):
                candidate = p + sign * offset
                if candidate in gt_idxs and candidate not in matched_gt:
                    matched_pred.add(p)
                    matched_gt.add(candidate)
                    break
            if p in matched_pred:
                break

    tp = len(matched_pred)
    fp = len(pred_idxs) - tp
    fn = len(gt_idxs) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(precision_tol=precision, recall_tol=recall, f1_tol=f1)
