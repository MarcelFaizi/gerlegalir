from typing import List, Set, Any
import math
import numpy as np

def precision(pred: List[Any], true: Set[Any]) -> float:
    """
    Calculate precision score.

    Args:
        pred (List[Any]): Predicted items.
        true (Set[Any]): True relevant items.

    Returns:
        float: Precision score.
    """
    if not pred:
        return 0.0
    return len(set(pred) & true) / len(pred)

def recall(pred: List[Any], true: Set[Any]) -> float:
    """
    Calculate recall score.

    Args:
        pred (List[Any]): Predicted items.
        true (Set[Any]): True relevant items.

    Returns:
        float: Recall score.
    """
    if not true:
        return 0.0
    return len(set(pred) & true) / len(true)

def f1_score(pred: List[Any], true: Set[Any]) -> float:
    """
    Calculate F1 score.

    Args:
        pred (List[Any]): Predicted items.
        true (Set[Any]): True relevant items.

    Returns:
        float: F1 score.
    """
    p = precision(pred, true)
    r = recall(pred, true)
    return 2 * (p * r) / (p + r) if p + r > 0 else 0.0

def dcg(relevances: List[float]) -> float:
    """
    Compute Discounted Cumulative Gain.

    Args:
        relevances (List[float]): List of relevance scores.

    Returns:
        float: DCG score.
    """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg(pred: List[Any], true: Set[Any]) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.

    Args:
        pred (List[Any]): Predicted items.
        true (Set[Any]): True relevant items.

    Returns:
        float: NDCG score.
    """
    relevances = [1 if item in true else 0 for item in pred]
    ideal_relevances = sorted([1] * len(true) + [0] * (len(pred) - len(true)), reverse=True)
    
    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_relevances)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0
