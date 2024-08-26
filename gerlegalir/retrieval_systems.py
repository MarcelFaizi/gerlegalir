from typing import List, Optional, Union, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random

from gerlegalir.utils.embedding_models import EmbeddingModel, HuggingFaceEmbeddingModel
from gerlegalir.config import get_embedding_filename

class RetrievalSystem:
    """Base class for retrieval systems."""

    def __init__(self, documents: List[str]):
        """
        Initialize the RetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
        """
        self.documents = documents

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text.

        Args:
            text (str): Text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        return text

    def get_relevant_documents(self, query: str) -> List[int]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query (str): Query to search for.

        Returns:
            List[int]: List of indices of relevant documents.
        """
        raise NotImplementedError("Subclasses should implement this method")


class TFIDFRetrievalSystem(RetrievalSystem):
    """Class to perform retrieval using TF-IDF."""

    def __init__(self, documents: List[str]):
        """
        Initialize the TFIDFRetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
        """
        super().__init__(documents)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve relevant documents using TF-IDF.

        Args:
            query (str): Query to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        query_clean = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query_clean])
        scores = np.dot(query_vector, self.tfidf_matrix.T).toarray().flatten()

        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        return top_k_indices.tolist()

class BM25RetrievalSystem(RetrievalSystem):
    """Class to perform retrieval using BM25."""

    def __init__(self, documents: List[str], bm25_filename: str):
        """
        Initialize the BM25RetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
            bm25_filename (str): Filename to save/load the BM25 model.
        """
        super().__init__(documents)
        self.tokenized_documents = [self.preprocess_text(doc) for doc in self.documents]
        try:
            with open(bm25_filename, 'rb') as f:
                self.bm25 = pickle.load(f)
        except FileNotFoundError:
            self.bm25 = BM25Okapi(self.tokenized_documents)
            with open(bm25_filename, 'wb') as f:
                pickle.dump(self.bm25, f)

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve relevant documents using BM25.

        Args:
            query (str): Query to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        query = self.preprocess_text(query)
        scores = self.bm25.get_scores(query)
        top_k = np.argpartition(scores, -k)[-k:]
        bm25_scores = [{'id': idx, 'score': scores[idx]} for idx in top_k]
        bm25_scores = sorted(bm25_scores, key=lambda x: x['score'], reverse=True)
        return [score['id'] for score in bm25_scores]

class BiEncoderRetrievalSystem(RetrievalSystem):
    """Class to perform retrieval using Bi-encoder."""

    def __init__(self, documents: List[str], embedding_model: Union[EmbeddingModel, HuggingFaceEmbeddingModel], seed: Optional[int] = None, embeddings: Optional[np.ndarray] = None):
        """
        Initialize the BiEncoderRetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
            embedding_model (Union[EmbeddingModel, HuggingFaceEmbeddingModel]): Embedding model to use.
            seed (Optional[int]): Seed (only use if finetuned model is used, to ensure we're using the Paragraph embeddings from the right model).
            embeddings (Optional[np.ndarray]): Pre-computed document embeddings.
        """
        super().__init__(documents)
        self.embedding_model = embedding_model
        self.document_embeddings = embeddings
        self.seed = seed
        self.embedding_filename = get_embedding_filename(self.embedding_model.model_name, self.seed)
        if embeddings is None:
            try:
                with open(self.embedding_filename, 'rb') as f:
                    self.document_embeddings = pickle.load(f)
            except FileNotFoundError:
                print(f"Embeddings file not found: {self.embedding_filename}")
                print("Computing document embeddings...")
                self.document_embeddings = self.embedding_model.encode_and_save(documents, self.embedding_filename)



    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve relevant documents using Bi-encoder.

        Args:
            query (str): Query to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        query_vector_embedding = self.embedding_model.encode(query).reshape(1, -1)
        cosine_sim = cosine_similarity(query_vector_embedding, self.document_embeddings)
        ranks = np.argsort(-cosine_sim)
        return ranks[0][:k].tolist()

class Bm25CrossEncoderRetrievalSystem(RetrievalSystem):
    """Class to perform retrieval using BM25 for initial filtering and Cross-encoder for final ranking."""

    def __init__(self, documents: List[str], bm25_model: Optional[BM25RetrievalSystem] = None, crossencoder_model: Optional[CrossEncoder] = None, a: int = 1000):
        """
        Initialize the Bm25CrossEncoderRetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
            bm25_model (Optional[BM25RetrievalSystem]): BM25 model to use.
            crossencoder_model (Optional[CrossEncoder]): Cross-encoder model to use.
            a (int): Number of documents to prefilter using BM25.
        """
        super().__init__(documents)
        self.bm25 = BM25RetrievalSystem(documents) if bm25_model is None else bm25_model
        self.crossencoder_model = crossencoder_model
        self.prefiltration = a

    def set_prefiltration(self, a: int) -> None:
        """
        Set the prefiltration parameter.

        Args:
            a (int): Number of documents to prefilter using BM25.
        """
        self.prefiltration = a

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve relevant documents using BM25 and Cross-encoder.

        Args:
            query (str): Query to search for.
            k (int): Number of top documents to retrieve.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        bm25_top_a = self.bm25.get_relevant_documents(query, self.prefiltration)
        first_selection_docs = [self.documents[i] for i in bm25_top_a]
        model_inputs = [[query, doc] for doc in first_selection_docs]
        scores = self.crossencoder_model.predict(model_inputs)

        results = [{"input": inp, "score": score, "id": idx} for inp, score, idx in zip(model_inputs, scores, bm25_top_a)]
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return [result["id"] for result in results][:k]

class BiEncoderCrossEncoderRetrievalSystem(RetrievalSystem):
    """Class to perform retrieval using Bi-encoder for initial filtering and Cross-encoder for final ranking."""

    def __init__(self, documents: List[str],
                 biencoder_model: Optional[BiEncoderRetrievalSystem] = None,
                 crossencoder_model: Optional[CrossEncoder] = None,
                 a: int = 10):
        """
        Initialize the BiEncoderCrossEncoderRetrievalSystem.

        Args:
            documents (List[str]): List of documents to be used in the retrieval system.
            biencoder_model (Optional[BiEncoderRetrievalSystem]): Pre-initialized bi-encoder model.
                If None, a new BiEncoderRetrievalSystem will be created.
            crossencoder_model (Optional[CrossEncoder]): Pre-initialized cross-encoder model.
            a (int): Number of documents to pre-filter using the bi-encoder (default: 10).
        """
        super().__init__(documents)
        self.documents = documents
        self.biencoder = BiEncoderRetrievalSystem(documents) if biencoder_model is None else biencoder_model
        self.crossencoder_model = crossencoder_model
        self.prefiltration = a

    def set_prefiltration(self, a: int) -> None:
        """
        Set the number of documents to pre-filter using the bi-encoder.

        Args:
            a (int): Number of documents to pre-filter.
        """
        self.prefiltration = a

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve relevant documents using bi-encoder for initial filtering and cross-encoder for final ranking.

        Args:
            query (str): The query string for retrieval.
            k (int): Number of top documents to return (default: 10).

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        # Step 1: Use bi-encoder to get initial set of relevant documents
        biencoder_top_a = self.biencoder.get_relevant_documents(query, self.prefiltration)
        first_selection_docs = [self.documents[i] for i in biencoder_top_a]

        # Step 2: Prepare inputs for cross-encoder
        model_inputs = [[query, doc] for doc in first_selection_docs]

        # Step 3: Use cross-encoder to score query-document pairs
        scores = self.crossencoder_model.predict(model_inputs)

        # Step 4: Combine results and sort by score
        results = [{"input": inp, "score": score, "id": idx}
                   for inp, score, idx in zip(model_inputs, scores, biencoder_top_a)]
        results.sort(key=lambda x: x["score"], reverse=True)

        # Step 5: Return top-k document indices
        return [result["id"] for result in results[:k]]

class MajorityVoteRetrievalSystem(RetrievalSystem):
    """
    Class to perform retrieval using a majority vote among multiple retrieval systems.

    This system combines results from multiple retrieval systems and ranks documents
    based on their frequency of appearance across all systems.
    """

    def __init__(self, retrieval_systems: List[Any]):
        """
        Initialize the MajorityVoteRetrievalSystem.

        Args:
            retrieval_systems (List[Any]): List of instantiated retrieval systems.
                Each system should have a get_relevant_documents method.
        """
        super().__init__(documents=None)  # No documents at this level, each system has its own
        self.retrieval_systems = retrieval_systems

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve documents based on a majority vote among the retrieval systems.

        This method queries all the retrieval systems and combines their results,
        ranking documents based on how frequently they appear across all systems.

        Args:
            query (str): The query string for retrieval.
            k (int): Number of top documents to return. Defaults to 10.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        # Step 1: Collect results from all retrieval systems
        all_retrieved_docs = []
        for system in self.retrieval_systems:
            retrieved_docs = system.get_relevant_documents(query, k)
            all_retrieved_docs.extend(retrieved_docs)

        # Step 2: Count occurrences of each document
        doc_counter = Counter(all_retrieved_docs)

        # Step 3: Get the most common documents
        common_docs = doc_counter.most_common()

        # Step 4: Rank documents based on frequency and handle ties
        top_docs = []
        for doc, _ in common_docs:
            top_docs.append(doc)
            if len(top_docs) == k:
                break

        return top_docs

class MajorityVoteRetrievalSystemWeighted(RetrievalSystem):
    """
    Class to perform retrieval using a weighted majority vote among multiple retrieval systems.

    This system combines results from multiple retrieval systems, applying weights to each system's
    contributions, and ranks documents based on their weighted frequency of appearance across all systems.
    """

    def __init__(self, retrieval_systems: List[Any], weights: Optional[List[float]] = None):
        """
        Initialize the MajorityVoteRetrievalSystemWeighted.

        Args:
            retrieval_systems (List[Any]): List of instantiated retrieval systems.
                Each system should have a get_relevant_documents method.
            weights (Optional[List[float]]): List of float weights for each retrieval system.
                If not provided, equal weights will be assigned. Weights should sum to 1.

        Raises:
            AssertionError: If the number of weights doesn't match the number of retrieval systems,
                            or if the weights don't sum to 1.
        """
        super().__init__(documents=None)  # No documents at this level, each system has its own
        self.retrieval_systems = retrieval_systems
        self.weights = weights if weights else [1 / len(retrieval_systems)] * len(retrieval_systems)
        assert len(self.weights) == len(self.retrieval_systems), "Weights and retrieval systems must be of same length"
        assert np.isclose(sum(self.weights), 1.0), "Weights must sum to 1"

    def get_relevant_documents(self, query: str, k: int = 10) -> List[int]:
        """
        Retrieve documents based on a weighted majority vote among the retrieval systems.

        This method queries all the retrieval systems and combines their results,
        ranking documents based on their weighted frequency of appearance across all systems.

        Args:
            query (str): The query string for retrieval.
            k (int): Number of top documents to return. Defaults to 10.

        Returns:
            List[int]: List of indices of the top-k relevant documents.
        """
        # Step 1: Collect weighted results from all retrieval systems
        all_retrieved_docs = []
        for system, weight in zip(self.retrieval_systems, self.weights):
            retrieved_docs = system.get_relevant_documents(query, k)
            # Multiply by 100 to avoid fraction counts
            all_retrieved_docs.extend(retrieved_docs * int(weight * 100))

        # Step 2: Count occurrences of each document
        doc_counter = Counter(all_retrieved_docs)

        # Step 3: Get the most common documents
        common_docs = doc_counter.most_common()

        # Step 4: Rank documents based on weighted frequency and handle ties
        top_docs = []
        for doc, _ in common_docs:
            top_docs.append(doc)
            if len(top_docs) == k:
                break

        # Step 5: If there's a tie or not enough documents, fill randomly
        if len(top_docs) < k:
            remaining_docs = list(set(all_retrieved_docs) - set(top_docs))
            if len(remaining_docs) > 0:
                additional_docs = random.sample(remaining_docs, min(k - len(top_docs), len(remaining_docs)))
                top_docs.extend(additional_docs)

        return top_docs
