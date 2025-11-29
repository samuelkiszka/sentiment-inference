import os
import math
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import DistilBertTokenizer, DistilBertModel

DATA_MODEL =    768     # DistilBERT base hidden size
N_CLASSES =     2       # Positive and Negative sentiment

class DataLoader:
    def __init__(self, split="train", batch_size=32, num_per_class=250):
        self.split = split
        self.batch_size = batch_size
        self.num_per_class = num_per_class
        self.dataset = self.__load_dataset(num_per_class)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]

            embeddings = batch['embeddings']
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            embeddings.astype(np.float32)

            mask = batch['attention_mask']
            if isinstance(mask, tuple):
                mask = mask[0]
            mask.astype(bool)

            labels = np.array(batch['label'], dtype=np.int32)

            yield embeddings, mask, labels

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __load_dataset(self, num_per_class):
        # Check if embedded dataset already exists
        path = f"data/sample_{num_per_class}/{self.split}_embedded"
        if os.path.exists(path):
            print(f"Loading existing embedded dataset from {path} ...")
            return load_from_disk(path)

        # Prepare embedded dataset
        dataset = self.__get_dataset(num_per_class)
        encoded_dataset = self.__encode_dataset(dataset)
        embedded_dataset = self.__embed_dataset(encoded_dataset)
        return embedded_dataset

    def __get_dataset(self, num_per_class):
        # Load full IMDB dataset
        dataset = load_dataset("imdb")

        def sample_class(split, label, n):
            ds = dataset[split].filter(lambda x: x['label'] == label)
            ds = ds.shuffle().select(range(n))
            return ds

        # Sample balanced dataset
        pos = sample_class(self.split, label=1, n=num_per_class)
        neg = sample_class(self.split, label=0, n=num_per_class)
        sample_dataset = concatenate_datasets([pos, neg]).shuffle(seed=42)

        return sample_dataset

    def __encode_dataset(self, dataset):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        def encode(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

        # Tokenize data
        encoded = dataset.map(encode, batched=True)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return encoded

    def __embed_dataset(self, dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        def embed_batch(batch):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return {'embeddings': outputs.last_hidden_state.cpu()}

        # Generate dataset embeddings
        embedded = dataset.map(embed_batch, batched=True)
        # Convert to numpy which is used later for training
        embedded.set_format(type='numpy', columns=['embeddings', 'attention_mask', 'label'])
        # Save embedded dataset to disk for future use
        os.makedirs(f"data/sample_{self.num_per_class}", exist_ok=True)
        embedded.save_to_disk(f"data/sample_{self.num_per_class}/{self.split}_embedded")

        return embedded
