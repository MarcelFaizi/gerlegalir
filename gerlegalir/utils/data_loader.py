import pandas as pd
from gerlegalir.config import GERLAYQA_PATH, GERLEGALTEXT_PATH
class GerLayQADataset():
    def __init__(self):
        self.glqa_df = pd.read_json(GERLAYQA_PATH, encoding='utf-8')

        self.labels = self.generate_labels()
        self.dataset_df = pd.DataFrame({'text': self.glqa_df['Question_text'], 'labels': self.labels})
        self.query_idx_mapping = self.generate_mapping()
        self.rel_docs_mapping = self.generate_rel_docs_mapping()
    def generate_labels(self):
        df_legal = pd.read_pickle(GERLEGALTEXT_PATH)
        df_legal_bgb = df_legal.loc[df_legal['name'] == "BGB"]
        labels = []
        for idx, row in self.glqa_df.iterrows():
            question_paragraph_doc_ids = []
            for paragraph in row['Paragraphs']:
                try:
                    question_paragraph_doc_ids.append(df_legal_bgb.loc[df_legal_bgb['paragraph'] == paragraph].index[0])
                except:
                    #print(f"Paragraph {paragraph} not found")
                    question_paragraph_doc_ids.append(-1)
                    continue
            labels.append(question_paragraph_doc_ids)
        return labels

    def get_dataset(self):
        return self.dataset_df
    def get_input(self, idx:int):
        return self.glqa_df.iloc[idx]['Question_text']

    def get_label(self, idx:int):
        return self.labels[idx]

    def generate_mapping(self):
        mapping = {}
        for idx, row in self.dataset_df.iterrows():
            mapping[idx] = row['text']
        return mapping

    def generate_rel_docs_mapping(self):
        mapping = {}
        for idx, row in self.dataset_df.iterrows():
            mapping[idx] = row['labels']
        return mapping


class LegalTextDataset():
    def __init__(self):
        self.df = pd.read_pickle(GERLEGALTEXT_PATH)
        self.doc_id_mapping = self.generate_mapping()

    def get_documents(self):
        return self.df['text'].tolist()

    def generate_mapping(self):
        mapping = {}
        for idx, row in self.df.iterrows():
            mapping[idx] = row['text']
        return mapping





