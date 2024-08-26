import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from gerlegalir.utils.data_loader import GerLayQADataset, LegalTextDataset


def create_examples(df, document_base, is_train=True):
    examples = []
    for _, row in df.iterrows():
        query = row["text"]
        relevant_documents = row["labels"]
        if not relevant_documents or relevant_documents == [-1]:
            continue

        for document in relevant_documents:
            rand_neg = np.random.choice([x for x in range(len(document_base)) if x not in relevant_documents])
            examples.append(InputExample(texts=[query, document_base[document]], label=1))
            examples.append(InputExample(texts=[query, document_base[rand_neg]], label=0))

    return examples


def train_model(model_name):
    model = CrossEncoder(model_name, max_length=512, num_labels=1)

    # Load datasets
    glqa = GerLayQADataset()
    legaltext = LegalTextDataset()
    document_base = legaltext.get_documents()
    dataset_df = glqa.get_dataset()

    # Split dataset
    train_df = dataset_df.sample(frac=0.95, random_state=42)
    test_df = dataset_df.drop(train_df.index)

    # Create examples
    train_examples = create_examples(train_df, document_base)
    test_examples = create_examples(test_df, document_base, is_train=False)

    # Create data loader and evaluator
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_examples, name="eval")

    # Training settings
    num_epochs = 4
    output_path = f'./output/cross/{model_name}_trained'

    # Train the model
    model.fit(
        loss_fct=torch.nn.MSELoss(),
        evaluator=evaluator,
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        evaluation_steps=5000,
        warmup_steps=int(len(train_dataloader) * num_epochs * 0.1),
        output_path=output_path,
        optimizer_params={"lr": 7e-6},
        use_amp=True
    )

    print(f"Model training completed and saved at {output_path}")


def main():
    model_names = [
        'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'svalabs/cross-electra-ms-marco-german-uncased',
        'PM-AI/bi-encoder_msmarco_bert-base_german'
    ]

    for model_name in model_names:
        train_model(model_name)


if __name__ == "__main__":
    main()