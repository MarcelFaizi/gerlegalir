import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import models
from gerlegalir.utils.data_loader import GerLayQADataset, LegalTextDataset
import re


def remove_backslashes_and_prefix(text):
    pattern = r'.*?/'
    return re.sub(pattern, '', text)


def create_train_examples(train_df, document_base):
    train_examples = []
    for _, row in train_df.iterrows():
        query = row["text"]
        relevant_documents = row["labels"]
        if not relevant_documents or relevant_documents == [-1]:
            continue

        for document in relevant_documents:
            rand_neg = np.random.choice([x for x in range(len(document_base)) if x not in relevant_documents])
            train_examples.append({
                "anchor": query,
                "positive": document_base[document],
                "negative": document_base[rand_neg],
            })
    return train_examples


def train_model(model_name, seed):
    # Model initialization
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Dataset preparation
    glqa = GerLayQADataset()
    legaltext = LegalTextDataset()
    document_base = legaltext.get_documents()
    dataset_df = glqa.get_dataset()
    train_df = dataset_df.sample(frac=0.95, random_state=seed)

    # Create training examples
    train_examples = create_train_examples(train_df, document_base)
    dataset = [InputExample(texts=[example["anchor"], example["positive"], example["negative"]]) for example in
               train_examples]
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

    # Training setup
    train_loss = losses.MultipleNegativesRankingLoss(model)
    num_epochs = 4
    output_path = f'./output/{seed}/{remove_backslashes_and_prefix(model_name)}_trained'

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=int(len(train_dataloader) * num_epochs * 0.1),
        output_path=output_path
    )

    print(f"Model training completed and saved at {output_path}")


def main():
    model_names = [
        'intfloat/multilingual-e5-small',
        'PM-AI/bi-encoder_msmarco_bert-base_german',
        'sentence-transformers/all-mpnet-base-v2'
    ]
    seeds = [0,42, 1337]

    for seed in seeds:
        for model_name in model_names:
            train_model(model_name, seed)


if __name__ == "__main__":
    main()