import numpy as np
import argparse

from src.transformer_encoder import TransformerClassifier


N_MODEL = 768   # Given by DistilBERT base hidden size
N_CLASSES = 2   # Positive and Negative sentiment
N_LAYERS = 2    # Default value from training
N_HEADS = 4     # Default value from training
D_FF = 64       # Default value from training


parser = argparse.ArgumentParser(description="Review Classification Inference")
parser.add_argument("-p", "--model_path", type=str, default='models/sample_250/best_model_l2h4ff64b32.npz', help='Path to the trained model file')
parser.add_argument("-m", "--mode", type=str, default='test', help="Mode: 'test' for test set inference, 'interactive' for live input inference")
parser.add_argument("-l", "--n_layers", type=int, default=N_LAYERS, help="Number of transformer encoder layers (default: 2)")
parser.add_argument("-a", "--n_heads", type=int, default=N_HEADS, help="Number of attention heads per layer (default: 4)")
parser.add_argument("-f", "--d_ff", type=int, default=D_FF, help="Feedforward network dimension (default: 64)")
args = parser.parse_args()

model = TransformerClassifier(d_model=N_MODEL, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff, n_classes=N_CLASSES)
model.load(args.model_path)


def inference_on_test_split():
    # Import here to so that src modules are only loaded when needed
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
    # Import here to so that src modules are only loaded when needed
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    print("\nEnter a review and class divided by '|' (type 'exit' to quit):")
    print("\tPositive example: This movie was fantastic! I loved it. | 1")
    print("\tNegative example: I hated this movie. It was terrible. | 0")

    correct = 0
    count = 0
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            if count > 0:
                accuracy = correct / count * 100
                print(f"Overall Accuracy: {accuracy:.2f}%")
            break

        review, correct_sentiment = text.split('|')

        # Tokenize and get embeddings
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.numpy()
        attention_mask = inputs['attention_mask'].numpy()

        # Get prediction
        logits = model.forward(embeddings, attention_mask)
        pred = np.argmax(logits)
        sentiment = 'Positive' if pred == 1 else 'Negative'
        correct += (pred == int(correct_sentiment.strip()))
        count += 1
        print(f"Predicted Sentiment: {sentiment}")



if __name__ == '__main__':
    if args.mode == 'test':
        inference_on_test_split()
    elif args.mode == 'interactive':
        interactive_inference()

