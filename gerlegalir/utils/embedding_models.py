from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pickle

class EmbeddingModel:
    """Class to load a sentence transformer model and encode text."""

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the EmbeddingModel.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
            device (str): Device to use for computations (e.g., 'cpu', 'cuda').
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.device = device
        self.model.to(device)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text input.

        Args:
            text (str): Text to encode.

        Returns:
            np.ndarray: Encoded text as a numpy array.
        """
        return self.model.encode(text, device=self.device)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of text inputs.

        Args:
            texts (List[str]): List of texts to encode.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Encoded texts as a numpy array.
        """
        return self.model.encode(texts, device=self.device, batch_size=batch_size, show_progress_bar=True)

    def encode_and_save(self, documents: List[str], filename: str, batch_size: int = 32) -> np.ndarray:
        """
        Encode a dataset of documents and save the embeddings.

        Args:
            documents (List[str]): List of documents to encode.
            filename (str): Filename to save the embeddings.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Encoded documents as a numpy array.
        """
        embeddings = self.encode_batch(documents, batch_size)
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

class HuggingFaceEmbeddingModel:
    """Class to load a Hugging Face model and encode text."""

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the HuggingFaceEmbeddingModel.

        Args:
            model_name (str): Name of the Hugging Face model to use.
            device (str): Device to use for computations (e.g., 'cpu', 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.device = device
        self.model.to(device)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text input.

        Args:
            text (str): Text to encode.

        Returns:
            np.ndarray: Encoded text as a numpy array.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of text inputs.

        Args:
            texts (List[str]): List of texts to encode.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Encoded texts as a numpy array.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def encode_and_save(self, documents: List[str], filename: str, batch_size: int = 32) -> np.ndarray:
        """
        Encode a dataset of documents and save the embeddings.

        Args:
            documents (List[str]): List of documents to encode.
            filename (str): Filename to save the embeddings.
            batch_size (int): Batch size for encoding.

        Returns:
            np.ndarray: Encoded documents as a numpy array.
        """
        embeddings = self.encode_batch(documents, batch_size)
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings
