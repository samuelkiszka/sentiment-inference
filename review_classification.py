import numpy as np
import argparse

from src.transformer_encoder import TransformerClassifier
from transformers import DistilBertTokenizer, DistilBertModel
import torch


N_MODEL = 768   # Given by DistilBERT base hidden size
N_CLASSES = 2   # Positive and Negative sentiment
N_LAYERS = 2    # Default value from training
N_HEADS = 4     # Default value from training
D_FF = 64       # Default value from training


parser = argparse.ArgumentParser(description="Review Classification Inference")
parser.add_argument("-p", "--model_path", type=str, default='models/sample_250/best_model_l2h4ff64b32.npz', help='Path to the trained model file')
parser.add_argument("-m", "--mode", type=str, default='test', help="Mode: 'test' for test set inference, 'interactive' for live input inference")
args = parser.parse_args()


model = TransformerClassifier(d_model=N_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF, n_classes=N_CLASSES)
model.load(args.model_path)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

def inference_on_test_split():
    from src.data_loader import DataLoader
    dataloader = DataLoader(split="test", batch_size=32, num_per_class=250)

    total = 0
    correct = 0

    for embeddings, mask, labels in dataloader:
        logits = model.forward(embeddings, mask)
        preds = np.argmax(logits, axis=1)

        correct += (preds == labels).sum()
        total += len(labels)

    accuracy = correct / total * 100
    print(f"Test Set Accuracy: {accuracy:.2f}%")

def interactive_inference():
    print("Enter a review (type 'exit' to quit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            break

        # Tokenize and get embeddings
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.numpy()
        attention_mask = inputs['attention_mask'].numpy()

        # Get prediction
        logits = model.forward(embeddings, attention_mask)
        pred = np.argmax(logits)
        sentiment = 'Positive' if pred == 1 else 'Negative'
        print(f"Predicted Sentiment: {sentiment}")



if __name__ == '__main__':
    if args.mode == 'test':
        inference_on_test_split()
    elif args.mode == 'interactive':
        interactive_inference()

