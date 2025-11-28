import numpy as np


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, lr:float=2e-5, weight_decay:float=1e-4, betas:tuple=(0.9, 0.999), eps:float=1e-8):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.lr = lr
        self.optimizer = optimizer

        # AdamW states
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.m = {} # first moment
        self.v = {} # second moment
        self.t = 0  # time step

        for i, (param, _) in enumerate(self.model.get_param_grads()):
            self.m[i] = np.zeros_like(param)
            self.v[i] = np.zeros_like(param)

    def step_sgd(self):
        for param, grad in self.model.get_param_grads():
            param -= self.lr * grad

    def step_adamw(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.model.get_param_grads()):
            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            # Update parameters with weight decay
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)

    def train_epoch(self):
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for idx, (embeddings, mask, labels) in enumerate(self.train_dataloader):
            self.model.clear_grads()

            logits = self.model.forward(embeddings, mask)

            loss, d_logits = self.cross_entropy_loss_and_grad(logits, labels)
            num_samples, correct = self.accuracy_from_logits(logits, labels)

            total_correct += correct
            total_samples += num_samples
            total_loss += loss * num_samples

            self.model.backward(d_logits)

            if self.optimizer == 'adamw':
                self.step_adamw()
            else:
                self.step_sgd()

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def evaluate(self):
        total_samples = 0
        total_loss = 0.0
        total_correct = 0

        for idx, (embeddings, mask, labels) in enumerate(self.test_dataloader):
            logits = self.model.forward(embeddings, mask)

            loss, _ = self.cross_entropy_loss_and_grad(logits, labels)
            num_samples, correct = self.accuracy_from_logits(logits, labels)

            total_correct += correct
            total_samples += num_samples
            total_loss += loss * num_samples

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

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
        correct = (preds == labels).sum()
        num_samples = len(labels)
        return num_samples, correct
