import numpy as np


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr:float=0.001):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.lr = lr

    def step_sgd(self):
        for param, grad in self.model.get_param_grads():
            param -= self.lr * grad

    def train_epoch(self):
        total_loss = 0.0
        total_acc = 0.0
        n_batches = len(self.train_dataloader)

        for idx, (embeddings, mask, labels) in enumerate(self.train_dataloader):
            self.model.clear_grads()

            logits = self.model.forward(embeddings, mask)

            loss, d_logits = self.cross_entropy_loss_and_grad(logits, labels)
            acc = self.accuracy_from_logits(logits, labels)

            self.model.backward(d_logits)
            self.step_sgd()
            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        return avg_loss, avg_acc

    def evaluate(self):
        total_loss = 0.0
        total_acc = 0.0
        n_batches = len(self.test_dataloader)

        for idx, (embeddings, mask, labels) in enumerate(self.train_dataloader):
            logits = self.model.forward(embeddings, mask)

            loss, _ = self.cross_entropy_loss_and_grad(logits, labels)
            acc = self.accuracy_from_logits(logits, labels)

            total_loss += loss
            total_acc += acc

        return total_loss / n_batches, total_acc / n_batches

    @staticmethod
    def cross_entropy_loss_and_grad(logits: np.ndarray, labels: np.ndarray):
        logits = logits.astype(np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        batch_size = logits.shape[0]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), labels] = 1
        loss = - np.sum(one_hot * np.log(probs + 1e-9)) / batch_size
        grad = (probs - one_hot) / batch_size
        return loss, grad

    @staticmethod
    def accuracy_from_logits(logits, labels):
        preds = np.argmax(logits, axis=1)
        return (preds == labels).mean()