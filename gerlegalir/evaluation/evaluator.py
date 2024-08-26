import time
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

from gerlegalir.evaluation.metrics import precision, recall, f1_score, ndcg
from gerlegalir.retrieval_systems import (
    RetrievalSystem,
    BiEncoderRetrievalSystem,
    BM25RetrievalSystem
)
from gerlegalir.utils.embedding_models import EmbeddingModel
from gerlegalir.utils.mongodb_connector import MongoDBConnector
from gerlegalir.config import PRETRAINED_BIENCODER, FINETUNED_BIENCODER

def evaluate_system(
    retrieval_system: RetrievalSystem,
    queries: List[str],
    relevant_documents: List[List[int]],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Evaluate a single retrieval system using various metrics.
    
    Args:
    - retrieval_system: The retrieval system to evaluate
    - queries: List of query strings
    - relevant_documents: List of lists containing relevant document indices for each query
    - k_values: List of k values for which to compute metrics
    
    Returns:
    - Dictionary containing computed metrics
    """
    results = {f"{metric}_at{k}": [] for metric in ['precision', 'recall', 'f1'] for k in k_values}
    results['ndcg'] = []

    for query, true_relevant in tqdm(zip(queries, relevant_documents), total=len(queries), desc="Evaluating queries"):
        retrieved = retrieval_system.get_relevant_documents(query, max(k_values))
        
        for k in k_values:
            retrieved_at_k = retrieved[:k]
            results[f'precision_at{k}'].append(precision(retrieved_at_k, set(true_relevant)))
            results[f'recall_at{k}'].append(recall(retrieved_at_k, set(true_relevant)))
            results[f'f1_at{k}'].append(f1_score(retrieved_at_k, set(true_relevant)))
        
        results['ndcg'].append(ndcg(retrieved, set(true_relevant)))

    # Compute averages
    return {metric: sum(values) / len(values) for metric, values in results.items()}

def evaluate_retrieval_systems(
    test_df: pd.DataFrame,
    document_base: List[str],
    model_names: List[str],
    mdbc: MongoDBConnector,
    collection_name: str,
    device: str
) -> None:
    """
    Evaluate multiple retrieval systems and store results in MongoDB.
    
    Args:
    - test_df: DataFrame containing test queries and relevant document indices
    - document_base: List of all documents in the corpus
    - model_names: List of model names to evaluate
    - mdbc: MongoDB connector instance
    - collection_name: Name of the MongoDB collection to store results
    - device: Device to use for computations (e.g., 'cpu', 'cuda')
    """
    for model_name in model_names:
        print(f"Evaluating: {model_name}")
        
        if mdbc.is_result_present(model_name, "default;biencoder", "default", collection_name):
            print("Result already present, skipping...")
            continue
        
        stime = time.time()
        
        if model_name == "BM25":
            system = BM25RetrievalSystem(document_base, f"bm25legal.pkl")
        else:
            em = EmbeddingModel(model_name=model_name, device=device)
            system = BiEncoderRetrievalSystem(document_base, embedding_model=em)
        
        metrics = evaluate_system(system, test_df['text'].tolist(), test_df['labels'].tolist())
        
        entry = mdbc.create_entry(
            f1_score_at5=metrics['f1_at5'],
            f1_score_at10=metrics['f1_at10'],
            precision_at5=metrics['precision_at5'],
            precision_at10=metrics['precision_at10'],
            recall_at5=metrics['recall_at5'],
            recall_at10=metrics['recall_at10'],
            ndcg=metrics['ndcg'],
            model_name=model_name,
            model_type="default;biencoder",
            dataset_name="default",
            dataset_type=f"default-rnd{collection_name.split('experiment')[1]}",
            duration=time.time() - stime,
            device=device,
            jobid="N/A",
            nodeid="N/A"
        )
        
        mdbc.upload_result(entry, collection_name)
        print(f"Evaluation completed for {model_name}")
#
# def evaluate_ensemble_methods(
#     test_df: pd.DataFrame,
#     document_base: List[str],
#     pretrained_models: List[str],
#     finetuned_models: List[str],
#     mdbc: MongoDBConnector,
#     collection_name: str,
#     device: str
# ) -> None:
#     """
#     Evaluate ensemble methods (e.g., majority voting) and store results in MongoDB.
#
#     Args:
#     - test_df: DataFrame containing test queries and relevant document indices
#     - document_base: List of all documents in the corpus
#     - pretrained_models: List of pretrained model names
#     - finetuned_models: List of finetuned model names
#     - mdbc: MongoDB connector instance
#     - collection_name: Name of the MongoDB collection to store results
#     - device: Device to use for computations (e.g., 'cpu', 'cuda')
#     """
#     ensemble_configs = [
#         ("ALL", pretrained_models + finetuned_models),
#         ("ALL-PRETRAINED", pretrained_models),
#         ("ALL-FINETUNED", finetuned_models),
#         ("BEST-PRETRAINED-FINETUNED", [pretrained_models[0], finetuned_models[0]])
#     ]
#
#     for ensemble_name, model_list in ensemble_configs:
#         print(f"Evaluating ensemble: {ensemble_name}")
#
#         if mdbc.is_result_present(ensemble_name, "default;moe,majorityvote", "default", collection_name):
#             print("Result already present, skipping...")
#             continue
#
#         stime = time.time()
#
#         retrieval_systems = [
#             BiEncoderRetrievalSystem(document_base, embedding_model=EmbeddingModel(name, device=device))
#             for name in model_list
#         ]
#
#         ensemble_system = EnsembleRetrievalSystem(retrieval_systems)
#
#         metrics = evaluate_system(ensemble_system, test_df['text'].tolist(), test_df['labels'].tolist())
#
#         entry = mdbc.create_entry(
#             f1_score_at5=metrics['f1_at5'],
#             f1_score_at10=metrics['f1_at10'],
#             precision_at5=metrics['precision_at5'],
#             precision_at10=metrics['precision_at10'],
#             recall_at5=metrics['recall_at5'],
#             recall_at10=metrics['recall_at10'],
#             ndcg=metrics['ndcg'],
#             model_name=ensemble_name,
#             model_type="default;moe,majorityvote",
#             dataset_name="default",
#             dataset_type=f"default-rnd{collection_name.split('experiment')[1]}",
#             duration=time.time() - stime,
#             device=device,
#             jobid="N/A",
#             nodeid="N/A"
#         )
#
#         mdbc.upload_result(entry, collection_name)
#         print(f"Evaluation completed for ensemble: {ensemble_name}")