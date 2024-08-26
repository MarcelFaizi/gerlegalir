import os
import torch
import logging

from sentence_transformers import CrossEncoder
from tqdm import tqdm
import time
import sys
import argparse

from config import (
    SEEDS, PRETRAINED_BIENCODER, FINETUNED_BIENCODER, EVALUATION_K_VALUES,
    BM25_FILEPATH, DEFAULT_DEVICE, MONGODB_DATABASE, LOGGING_LEVEL, get_model_path,
    get_embedding_filename, get_experiment_collection_name, ENSEMBLE_CONFIGS, BIENCODER_CROSSENCODER_CONFIGS,
    BM25_CROSSENCODER_CONFIGS
)
from gerlegalir.utils.data_loader import GerLayQADataset, LegalTextDataset
from gerlegalir.utils.embedding_models import EmbeddingModel
from gerlegalir.retrieval_systems import (
    TFIDFRetrievalSystem,
    BM25RetrievalSystem,
    BiEncoderRetrievalSystem,
    BiEncoderCrossEncoderRetrievalSystem,
    Bm25CrossEncoderRetrievalSystem,
    MajorityVoteRetrievalSystem,
)
from gerlegalir.evaluation.evaluator import evaluate_system
from gerlegalir.utils.mongodb_connector import MongoDBConnector

# Set up logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_retrieval_systems(documents, device, seed, system_type):
    """Initialize specified retrieval systems."""
    systems = {}

    if system_type == 'pretrained':
        for model_name in PRETRAINED_BIENCODER:
            embedding_model = EmbeddingModel(model_name, device=device)
            systems[model_name] = BiEncoderRetrievalSystem(documents, embedding_model)
    elif system_type == 'finetuned':
        for model_name in FINETUNED_BIENCODER:
            embedding_model = EmbeddingModel(model_name=get_model_path(model_name=model_name, seed=seed), device=device)
            systems[model_name] = BiEncoderRetrievalSystem(documents, embedding_model, seed=seed)


    # elif system_type == 'tf-idf':
    #     systems['TF-IDF'] = TFIDFRetrievalSystem(documents)
    # elif system_type == 'bm25':
    #     systems['BM25'] = BM25RetrievalSystem(documents, bm25_filename=BM25_FILEPATH)
    # elif system_type == 'biencoder-crossencoder':
    #     for config in BIENCODER_CROSSENCODER_CONFIGS:
    #         biencoder_model = BiEncoderRetrievalSystem(
    #             documents,
    #             EmbeddingModel(config['biencoder_name'], device=device),
    #             seed=seed
    #         )
    #         crossencoder_model = CrossEncoder(config['crossencoder_name'], device=device)
    #         systems[f"{config['name']}"] = BiEncoderCrossEncoderRetrievalSystem(
    #             documents,
    #             biencoder_model,
    #             crossencoder_model,
    #             a=config['a']
    #         )
    # elif system_type == 'bm25-crossencoder':
    #     for config in BM25_CROSSENCODER_CONFIGS:
    #         bm25_model = BM25RetrievalSystem(documents, bm25_filename=BM25_FILEPATH)
    #         crossencoder_model = CrossEncoder(config['crossencoder_name'], device=device)
    #         systems[f"{config['name']}"] = Bm25CrossEncoderRetrievalSystem(
    #             documents,
    #             bm25_model,
    #             crossencoder_model,
    #             a=config['a']
    #         )


    elif system_type == 'ensemble':
        # First, initialize all individual models
        all_models = {}
        for model_name in PRETRAINED_BIENCODER + FINETUNED_BIENCODER:
            if model_name in PRETRAINED_BIENCODER:
                embedding_model = EmbeddingModel(model_name, device=device)
            else:
                embedding_model = EmbeddingModel(model_name=get_model_path(model_name=model_name, seed=seed),
                                                 device=device)
            all_models[model_name] = BiEncoderRetrievalSystem(documents, embedding_model, seed=seed)

        # Now create ensemble systems
        for ensemble_name, model_names in ENSEMBLE_CONFIGS:
            ensemble_systems = [all_models[name] for name in model_names]
            systems[ensemble_name] = MajorityVoteRetrievalSystem(ensemble_systems)
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    return systems


def main(system_type):
    # Load datasets
    glqa = GerLayQADataset()
    legaltext = LegalTextDataset()
    document_base = legaltext.get_documents()

    # Initialize MongoDB connector
    mdbc = MongoDBConnector(database=MONGODB_DATABASE)

    # Determine device
    device = torch.device(DEFAULT_DEVICE)
    logger.info(f"Using device: {device}")

    for seed in SEEDS:
        logger.info(f"Starting evaluation for seed {seed}")
        # Initialize retrieval systems
        retrieval_systems = initialize_retrieval_systems(document_base, device, seed, system_type)
        # Prepare dataset
        dataset_df = glqa.get_dataset()
        train_df = dataset_df.sample(frac=0.95, random_state=seed)
        test_df = dataset_df.drop(train_df.index)
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        # Get experiment collection name
        collection_name = get_experiment_collection_name(seed)

        # Evaluate each retrieval system
        for system_name, system in tqdm(retrieval_systems.items(), desc="Evaluating systems"):
            logger.info(f"Evaluating system: {system_name}")
            stime = time.time()
            # Check if result already exists
            if mdbc.is_result_present(system_name, "default", "default", collection_name):
                logger.info(f"Result for {system_name} already present, skipping...")
                continue

            # Evaluate the system
            evaluation_results = evaluate_system(
                system,
                test_df['text'].tolist(),
                test_df['labels'].tolist(),
                k_values=EVALUATION_K_VALUES
            )

            # Prepare and upload results
            entry = mdbc.create_entry(
                f1_score_at5=evaluation_results['f1_at5'],
                f1_score_at10=evaluation_results['f1_at10'],
                precision_at5=evaluation_results['precision_at5'],
                precision_at10=evaluation_results['precision_at10'],
                recall_at5=evaluation_results['recall_at5'],
                recall_at10=evaluation_results['recall_at10'],
                ndcg=evaluation_results['ndcg'],
                model_name=system_name,
                model_type=system_type,
                dataset_name="default",
                dataset_type=f"default-rnd{seed}",
                duration=time.time() - stime,
                device=str(device),
                jobid="N/A",
                nodeid="N/A"
            )
            mdbc.upload_result(entry, collection_name)

            logger.info(f"Evaluation for {system_name} completed and results uploaded")

    logger.info("All evaluations completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval system evaluations")
    parser.add_argument("system_type", choices=['pretrained', 'finetuned', 'ensemble'],
                        help="Type of retrieval system to evaluate")
    args = parser.parse_args()

    main(args.system_type)