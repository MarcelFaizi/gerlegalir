import os

import torch
from gerlegalir.utils.preprocessing import preprocess_text
# Random seeds for reproducibility
SEEDS = [0, 1337, 42]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMBEDDING_DIR = os.path.join(BASE_DIR, 'embeddings')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Dataset location
LEGAL_TEXT_FILENAME = 'GerLegalText.pkl'
GER_LAY_QA_FILENAME = 'GerLayQA.json'
GERLEGALTEXT_PATH = os.path.join(DATA_DIR, LEGAL_TEXT_FILENAME)
GERLAYQA_PATH = os.path.join(DATA_DIR, GER_LAY_QA_FILENAME)

# Evaluation settings
EVALUATION_K_VALUES = [5, 10]

# Dataset settings
TRAIN_FRACTION = 0.95

# MongoDB settings
MONGODB_URI = "" # Add your MongoDB URI here
MONGODB_DATABASE = "results"

# Compute settings
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 32

# Logging settings
LOGGING_LEVEL = "INFO"

# File naming conventions
def get_embedding_filename(model_name: str, seed: int = None):
    model_name = preprocess_text(model_name)
    if seed is not None:
        return os.path.join(EMBEDDING_DIR, f"legaltext_embeddings({model_name}_seed-{seed}).pkl")
    else:
        return os.path.join(EMBEDDING_DIR, f"legaltext_embeddings({model_name}).pkl")

def get_model_path(model_name: str, seed: int):
    return os.path.join(MODELS_DIR, f"seed_{seed}/{model_name}")

def get_experiment_collection_name(seed):
    return f"experiment{seed}"

# BM25 settings
BM25_FILEPATH = os.path.join(DATA_DIR, 'bm25legal.pkl')

# Model names
PRETRAINED_BIENCODER = [
    'sentence-transformers/all-MiniLM-L6-v2',
#    'intfloat/multilingual-e5-large',
#    'PM-AI/bi-encoder_msmarco_bert-base_german',
#    'Snowflake/snowflake-arctic-embed-l',
#    'Snowflake/snowflake-arctic-embed-m',
#    'sentence-transformers/all-mpnet-base-v2'
]

# Make sure the models are available under the specified names under ./models/seed_{seed}/{model_name}/
# Also, make sure the models are uniquely named e.g. add '_trained' to the model name
FINETUNED_BIENCODER = [
    #'all-mpnet-base-v2_trained',
    #'bi-encoder_msmarco_bert-base_german_trained',
   'multilingual-e5-small_trained'
]



# Ensemble configurations
ENSEMBLE_CONFIGS = [
    ("ALL", PRETRAINED_BIENCODER + FINETUNED_BIENCODER),
    ("ALL-PRETRAINED", PRETRAINED_BIENCODER),
    ("ALL-FINETUNED", FINETUNED_BIENCODER),
    ("BEST-PRETRAINED-FINETUNED", [PRETRAINED_BIENCODER[0], FINETUNED_BIENCODER[0]])
]

# BiEncoder-CrossEncoder configurations(biencoder_name and crossencoder_name should either be the paths to the models or model names from huggingface)
BIENCODER_CROSSENCODER_CONFIGS = [
    {
        'name': 'biencoder-crossencoder',
        'biencoder_name': 'all-MiniLM-L6-v2',
        'crossencoder_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'a': 15  # Number of documents to pre-filter using bi-encoder
    },
]

# BM25-CrossEncoder configurations(crossencoder_name should either be a path to a model or a model name from huggingface)
BM25_CROSSENCODER_CONFIGS = [
    {
        'name': 'bm25_crossencoder',
        'crossencoder_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'a': 1000  # Number of documents to pre-filter using BM25
    }
]



# Add any other configuration variables as needed