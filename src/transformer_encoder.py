import numpy as np


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads              # number of heads
        self.d_model = d_model              # model dimension
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads       # dimension per head

        self.W_Q = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # query matrix
        self.W_K = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # key matrix
        self.W_V = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # value matrix
        self.W_O = np.random.randn(n_heads * self.d_k, d_model) / np.sqrt(d_model)  # output matrix

        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

        self.cache = {}
        self.X = None

    def forward(self, X, mask=None):
        if X.ndim == 2:
            X = X[None]
        self.X = X
        batch_size, seq_len, d_model = X.shape

        # Linear projections
        Q_lin = X @ self.W_Q
        K_lin = X @ self.W_K
        V_lin = X @ self.W_V

        # Split into heads
        Q = Q_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Apply attention on all heads
        heads = []
        weights = []
        attn_caches = []
        for i in range(self.n_heads):
            out_h, w_h, cache_h = self.scaled_dot_product_attention(Q[:, i], K[:, i], V[:, i], mask)
            heads.append(out_h)
            weights.append(w_h)
            attn_caches.append(cache_h)

        # Concatenate heads
        H = np.stack(heads, axis=1)
        concat = H.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        out = concat @ self.W_O

        self.cache = {
            "Q_lin": Q_lin,
            "K_lin": K_lin,
            "V_lin": V_lin,
            "Q": Q,
            "K": K,
            "V": V,
            "weights": weights,
            "attn_caches": attn_caches,
            "concat": concat,
            "mask": mask,
        }

        return out

    def backward(self, d_out):
        cache = self.cache
        concat = cache["concat"]
        batch_size, seq_len, d_model = d_out.shape

        # Gradient w.r.t. W_O
        self.dW_O += concat.reshape(-1, d_model).T @ d_out.reshape(-1, d_model)

        # Gradient w.r.t. concat
        dconcat = d_out @ self.W_O.T
        dH = dconcat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # accumulate gradients for Q_lin, K_lin, V_lin
        dQ_lin = np.zeros_like(cache["Q_lin"])
        dK_lin = np.zeros_like(cache["K_lin"])
        dV_lin = np.zeros_like(cache["V_lin"])

        for h in range(self.n_heads):
            dout_h = dH[:, h]
            attn_cache = cache["attn_caches"][h]
            dQh, dKh, dVh = self.scaled_dot_product_attention_backward(dout_h, *attn_cache)
            start = h * self.d_k
            end = (h + 1) * self.d_k
            dQ_lin[:, :, start:end] = dQh
            dK_lin[:, :, start:end] = dKh
            dV_lin[:, :, start:end] = dVh

        # Gradients w.r.t. projection weights
        X = self.X
        self.dW_Q += X.reshape(-1, d_model).T @ dQ_lin.reshape(-1, d_model)
        self.dW_K += X.reshape(-1, d_model).T @ dK_lin.reshape(-1, d_model)
        self.dW_V += X.reshape(-1, d_model).T @ dV_lin.reshape(-1, d_model)

        # Gradients w.r.t. W_Q, W_K, W_V
        dX_Q = dQ_lin @ self.W_Q.T
        dX_K = dK_lin @ self.W_K.T
        dX_V = dV_lin @ self.W_V.T

        dx = dX_Q + dX_K + dX_V
        return dx

    def get_param_grads(self):
        # Note: Gradients for multi-head attention parameters are not implemented
        return [
            (self.W_Q, self.dW_Q),
            (self.W_K, self.dW_K),
            (self.W_V, self.dW_V),
            (self.W_O, self.dW_O),
        ]

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        assert Q.ndim in (2, 3) and Q.ndim == K.ndim == V.ndim

        is_batch = Q.ndim == 3
        if not is_batch:
            Q = Q[None]
            K = K[None]
            V = V[None]

        batch, seq_len, d_k = Q.shape
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

        scores += (~mask) * -1e9

        # Softmax
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = exp / exp.sum(axis=-1, keepdims=True)

        output = weights @ V
        if not is_batch:
            output = output[0]
            weights = weights[0]
        cache = (Q, K, V, weights, mask)
        return output, weights, cache

    def scaled_dot_product_attention_backward(self, d_out, Q, K, V, weights, mask):
        is_batch = Q.ndim == 3
        if not is_batch:
            Q = Q[None]
            K = K[None]
            V = V[None]
            d_out = d_out[None]
            weights = weights[None]

        batch_size, seq_len, d_k = Q.shape

        # Gradient w.r.t. V
        dweights = d_out @ V.transpose(0, 2, 1)
        dV = weights.transpose(0, 2, 1) @ d_out

        # Softmax backward
        dscores = (dweights - (weights * dweights).sum(axis=-1, keepdims=True)) * weights
        dscores /= np.sqrt(d_k)

        # Gradient w.r.t. Q and K
        dQ = dscores @ K
        dK = dscores.transpose(0, 2, 1) @ Q

        if not is_batch:
            dQ = dQ[0]
            dK = dK[0]
            dV = dV[0]

        return dQ, dK, dV


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gama = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.dgama = np.zeros_like(self.gama)
        self.dbeta = np.zeros_like(self.beta)
        self.cache = {}

    def forward(self, X):
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        x_norm = (X - mean) / (std + self.eps)

        out = self.gama * x_norm + self.beta

        # Store cache for backward pass
        self.cache = (X, mean, std, x_norm)

        return out

    def backward(self, d_out):
        X, mean, std, x_norm = self.cache
        N = X.shape[-1]

        axes = tuple(range(X.ndim-1))

        self.dgama += np.sum(d_out * x_norm, axis=axes)
        self.dbeta += np.sum(d_out, axis=axes)

        dx_norm = d_out * self.gama
        dstd = np.sum(dx_norm * (X - mean) * (-0.5) * (std + self.eps) ** -3, axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * (-1 / (std + self.eps)), axis=-1, keepdims=True) + dstd * np.mean(-2 * (X - mean), axis=-1, keepdims=True)

        dx = dx_norm / (std + self.eps) + dstd * 2 * (X - mean) / N + dmean / N
        return dx

    def get_param_grads(self):
        return [(self.gama, self.dgama), (self.beta, self.dbeta)]

    def clear_grads(self):
        self.dgama.fill(0)
        self.dbeta.fill(0)


class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):

        self.mha = MultiHeadAttention(d_model, n_heads)

        # Feed-forward network parameters
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # Layer norms
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # caches container
        self.cache = {}

    def forward(self, X, mask=None):
        self.cache.clear()

        # self-attention + residual + layer norm
        self.cache["X_pre_mha"] = X.copy()
        attn_out = self.mha.forward(X, mask)
        self.cache["attn_out"] = attn_out

        X1 = X + attn_out
        out1 = self.ln1.forward(X1)
        self.cache["out1"] = out1

        # feed-forward + residual + layer norm
        pre1 = out1 @ self.W1 + self.b1
        relu = np.maximum(0, pre1)
        self.cache["pre1"] = pre1
        self.cache["relu"] = relu

        pre2 = relu @ self.W2 + self.b2
        X2 = out1 + pre2
        out2 = self.ln2.forward(X2)
        return out2

    def backward(self, d_out):
        # Backprop through second layer norm
        dX2 = self.ln2.backward(d_out)

        # Backprop through feed-forward and residual
        d_pre2 = dX2

        # Gradients for W2 and b2
        relu = self.cache["relu"]
        bat_seq = d_pre2.reshape(-1, d_pre2.shape[-1])
        relu_flat = relu.reshape(-1, relu.shape[-1])
        self.dW2 += relu_flat.T @ bat_seq
        self.db2 += bat_seq.sum(axis=0)

        # Backprop through ReLU
        d_relu = d_pre2 @ self.W2.T
        d_pre1 = d_relu * (self.cache["pre1"] > 0)

        # Gradients for W1 and b1
        ff_in = self.cache["out1"]
        ff_in_flat = ff_in.reshape(-1, ff_in.shape[-1])
        d_pre1_flat = d_pre1.reshape(-1, d_pre1.shape[-1])
        self.dW1 += ff_in_flat.T @ d_pre1_flat
        self.db1 += d_pre1_flat.sum(axis=0)

        # Backprop to out1
        d_ff_in = d_pre1 @ self.W1.T
        d_out1 = dX2 + d_ff_in

        # Backprop through first layer norm
        dX1 = self.ln1.backward(d_out1)

        # split residual connection
        d_attn_out = dX1
        d_X_skip = dX1

        # Backprop through multi-head attention
        dx_from_mha = self.mha.backward(d_attn_out)

        # Combine gradients
        dX = d_X_skip + dx_from_mha

        return dX

    def get_param_grads(self):
        grads = []
        grads += self.mha.get_param_grads()
        grads += [
            (self.W1, self.dW1),
            (self.b1, self.db1),
            (self.W2, self.dW2),
            (self.b2, self.db2),
        ]
        grads += self.ln1.get_param_grads()
        grads += self.ln2.get_param_grads()
        return grads

    def clear_grads(self):
        self.mha.dW_Q.fill(0)
        self.mha.dW_K.fill(0)
        self.mha.dW_V.fill(0)
        self.mha.dW_O.fill(0)
        self.dW1.fill(0)
        self.db1.fill(0)
        self.dW2.fill(0)
        self.db2.fill(0)
        self.ln1.clear_grads()
        self.ln2.clear_grads()


class Encoder:
    def __init__(self, num_layers, d_model, n_heads, d_ff):
        self.layers = [EncoderBlock(d_model, n_heads, d_ff) for i in range(num_layers)]

    def forward(self, X, mask=None):
        if X.ndim == 2:
            X = X[None]

        for layer in self.layers:
            X = layer.forward(X, mask)

        return X.squeeze(0) if X.shape[0] == 1 else X

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def get_param_grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.get_param_grads()
        return grads

    def clear_grads(self):
        for layer in self.layers:
            layer.clear_grads()


class Classifier:
    def __init__(self, d_model, num_classes):
        self.W = np.random.randn(d_model, num_classes) / np.sqrt(d_model)
        self.b = np.zeros(num_classes)
        # Backpropagation variables
        self.enc_out = None
        self.h = None
        self.logits = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, enc_out, mask):
        self.enc_out = enc_out

        # mean pooling
        if enc_out.ndim == 3:
            self.h = enc_out.mean(axis=1)
        else:
            self.h = enc_out.mean(axis=0)

        self.logits = self.h @ self.W + self.b

        # softmax
        exp = np.exp(self.logits - np.max(self.logits))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs

    def backward(self, d_logits):
        # affine layer gradients
        self.dW = self.h.T @ d_logits
        self.db = d_logits.sum(axis=0)

        # gradient w.r.t. encoder output
        dh = d_logits @ self.W.T

        # backward mean pooling
        if self.enc_out.ndim == 3:
            _, seq, _ = self.enc_out.shape
            d_enc_out = np.repeat(dh[:, None, :], seq, axis=1) / seq
        else:
            seq = self.enc_out.shape[0]
            d_enc_out = np.repeat(dh[None, :], seq, axis=0) / seq

        return d_enc_out

    def get_param_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def clear_grads(self):
        self.dW.fill(0)
        self.db.fill(0)


class TransformerClassifier:
    def __init__(self, num_layers, d_model, n_heads, d_ff, num_classes):
        self.encoder = Encoder(num_layers, d_model, n_heads, d_ff)
        self.classifier = Classifier(d_model, num_classes)

    def forward(self, X, mask=None):
        enc_out = self.encoder.forward(X, mask[:, None, :])
        logits = self.classifier.forward(enc_out, mask)
        return logits

    def backward(self, d_logits):
        d_enc_out = self.classifier.backward(d_logits)
        self.encoder.backward(d_enc_out)

    def get_param_grads(self):
        return self.classifier.get_param_grads() + self.encoder.get_param_grads()

    def clear_grads(self):
        self.classifier.clear_grads()
        self.encoder.clear_grads()

    def save(self, path):
        params = []
        for param, _ in self.get_param_grads():
            params.append(param)
        np.savez(path, *params)
        print(f"Model parameters saved to {path}")

    def load(self, path):
        data = np.load(path)
        params = []
        for param, _ in self.get_param_grads():
            params.append(param)
        for i, param in enumerate(params):
            param[:] = data[f'arr_{i}']
        print(f"Model parameters loaded from {path}")