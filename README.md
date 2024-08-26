# GerLegalIR: German Legal Information Retrieval System

GerLegalIR is a comprehensive information retrieval system designed for German legal documents. It implements various retrieval methods, including TF-IDF, BM25, and neural approaches using bi-encoders and cross-encoders.

## Data Usage and Copyright Notice
This repository contains data that has been collected exclusively from publicly accessible websites. The process of data collection was conducted with respect to the websites' terms of use and is intended for educational and scientific research purposes only.

Disclaimer: We do not claim ownership or hold any copyrights over the data provided herein. All data remains the property of their respective owners. The data shared in this repository is offered without any warranties, and users are solely responsible for its use.

Usage Restrictions: The data hosted in this repository is strictly for non-commercial, scientific research purposes. Any use of this data for commercial purposes is expressly prohibited.
## Requirements

- Python 3.7+
- PyTorch
- Sentence Transformers
- sklearn
- numpy
- tqdm
- pymongo
## Setup

1. Clone the repository(it is important to name the folder gerlegalir):
   ```
   git clone https://github.com/MarcelFaizi/gerlegalir ./gerlegalir
   cd gerlegalir
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, you may use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install the project in editable mode:
   ```
   pip install -e .
   ```

5. Set up MongoDB:
   - Install MongoDB on your system if not already installed.
   - Update the `MONGODB_URI` in `config.py` with your MongoDB connection string.

## Configuration

The `config.py` file contains various settings that can be adjusted:

- `SEEDS`: Random seeds for reproducibility
- `PRETRAINED_BIENCODER`: List of pretrained bi-encoder model names
- `FINETUNED_BIENCODER`: List of finetuned bi-encoder model names
- `EVALUATION_K_VALUES`: K values for evaluation metrics
- `BM25_FILENAME`: Filename for saving/loading BM25 model
- `MONGODB_URI`: URI for MongoDB connection
- `MONGODB_DATABASE`: Name of the MongoDB database
- `ENSEMBLE_CONFIGS`: Configurations for ensemble methods

## Usage

1. Prepare your dataset:
   - Obtain the GerLegalText dataset by running /script/gerlegaltext_scraper.py
   - Obtain the GerLayQA dataset(Büttner and Habernal, Answering legal questions from laymen in German civil law system, 2024) by visiting: https://github.com/trusthlt/eacl24-german-legal-questions and downloading /data/GerLayQA.json
   - (alternatively, you can use your own dataset that has the data "Question_text" and "Paragraphs" for each entry)
   - Ensure your legal documents are in the correct format and location.
   - Update the `GerLayQADataset` and `LegalTextDataset` classes in `data_loader.py` if necessary.

2. Run the main evaluation script:
   ```
   python main.py <system_type>
   ```
   Where `<system_type>` can be one of:
   - pretrained
   - finetuned
   - ensemble

   This script will:
   - Initialize the specified class of retrieval systems
   - Evaluate each system on the test set
   - Store results in MongoDB

3. To run a specific experiment or modify parameters, you can edit the `main.py` file or create a new script using the provided classes and functions.

## Adding New Retrieval Systems

To add a new retrieval system:

1. Create a new class in `retrieval_systems.py` that inherits from `RetrievalSystem`.
2. Implement the `get_relevant_documents` method in your new class.
3. Add the new system to the `initialize_retrieval_systems` function in `main.py`.

## Results and Analysis
The system evaluates retrieval performance using the following metrics:
- Precision@5 and Precision@10
- Recall@5 and Recall@10
- F1-score@5 and F1-score@10
- NDCG (Normalized Discounted Cumulative Gain)

Results are stored in MongoDB. You can use these results to:
- Compare the performance of different retrieval methods
- Analyze the impact of different model configurations


## Contributing

Contributions to GerLegalIR are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request
