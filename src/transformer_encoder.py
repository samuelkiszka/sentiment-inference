import numpy as np


def softmax(x, eps=1e-9):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / (exp.sum(axis=-1, keepdims=True) + eps)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads              # number of heads
        self.d_model = d_model              # model dimension
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads       # dimension per head

        # Learnable linear projections
        self.W_Q = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # query matrix
        self.W_K = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # key matrix
        self.W_V = np.random.randn(d_model, n_heads * self.d_k) / np.sqrt(d_model)  # value matrix
        self.W_O = np.random.randn(n_heads * self.d_k, d_model) / np.sqrt(d_model)  # output matrix

        # Gradient buffers
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

        # Cache for backward pass
        self.cache = {}
        self.x = None

    def forward(self, x, mask=None):
        # Add batch dimension if missing
        if x.ndim == 2:
            x = x[None]
        self.x = x
        batch_size, seq_len, d_model = x.shape

        # ---- Linear projections ----
        Q_lin = x @ self.W_Q
        K_lin = x @ self.W_K
        V_lin = x @ self.W_V

        # ---- Split into multiple heads ----
        Q = Q_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V_lin.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # ---- Scaled Dot-Product Attention for each head ----
        head_outputs = []
        attn_weights = []
        attn_caches = []

        for h in range(self.n_heads):
            out_h, weight_h, cache_h = self.scaled_dot_product_attention(Q[:, h], K[:, h], V[:, h], mask)
            head_outputs.append(out_h)
            attn_weights.append(weight_h)
            attn_caches.append(cache_h)

        # ---- Concatenate heads ----
        heads = np.stack(head_outputs, axis=1)
        concat = heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # ---- Final linear projection ----
        out = concat @ self.W_O

        # store cache for backward pass
        self.cache = dict(
            Q_lin=Q_lin, K_lin=K_lin, V_lin=V_lin,
            Q=Q, K=K, V=V,
            weights=attn_weights,
            attn_caches=attn_caches,
            concat=concat,
            mask=mask,
        )

        return out

    def backward(self, d_out):
        cache = self.cache
        concat = cache["concat"]
        batch_size, seq_len, d_model = d_out.shape

        # ---- Gradients w.r.t. W_O ----
        self.dW_O += concat.reshape(-1, d_model).T @ d_out.reshape(-1, d_model)

        # ---- Gradient w.r.t. concatenated heads ----
        d_concat = d_out @ self.W_O.T
        d_H = d_concat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # ---- Initialize gradients for Q, K, V linear projections ----
        d_Q_lin = np.zeros_like(cache["Q_lin"])
        d_K_lin = np.zeros_like(cache["K_lin"])
        d_V_lin = np.zeros_like(cache["V_lin"])

        # ---- Backprop through each head ----
        for h in range(self.n_heads):
            d_out_h = d_H[:, h]
            q_cache = cache["attn_caches"][h]
            d_Qh, d_Kh, d_Vh = self.scaled_dot_product_attention_backward(d_out_h, *q_cache)
            start = h * self.d_k
            end = (h + 1) * self.d_k
            d_Q_lin[:, :, start:end] = d_Qh
            d_K_lin[:, :, start:end] = d_Kh
            d_V_lin[:, :, start:end] = d_Vh

        # ---- Gradients w.r.t. projection weights ----
        x = self.x
        self.dW_Q += x.reshape(-1, d_model).T @ d_Q_lin.reshape(-1, d_model)
        self.dW_K += x.reshape(-1, d_model).T @ d_K_lin.reshape(-1, d_model)
        self.dW_V += x.reshape(-1, d_model).T @ d_V_lin.reshape(-1, d_model)

        # ---- Gradients w.r.t. input x ----
        d_X_Q = d_Q_lin @ self.W_Q.T
        d_X_K = d_K_lin @ self.W_K.T
        d_X_V = d_V_lin @ self.W_V.T

        d_x = d_X_Q + d_X_K + d_X_V
        return d_x

    def get_param_grads(self):
        # Note: Gradients for multi-head attention parameters are not implemented
        return [
            (self.W_Q, self.dW_Q),
            (self.W_K, self.dW_K),
            (self.W_V, self.dW_V),
            (self.W_O, self.dW_O),
        ]
    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask=None):
        # Add batch dimension if missing
        is_batch = Q.ndim == 3
        if not is_batch:
            Q = Q[None]
            K = K[None]
            V = V[None]

        batch, seq_len, d_k = Q.shape

        # ---- Scaled dot-product attention ----
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # ---- Apply mask if provided ----
        if mask is not None:
            scores += (~mask) * -1e9

        # ---- Softmax to get attention weights ----
        weights = softmax(scores)

        # ---- Weighted sum of values ----
        output = weights @ V

        # Remove batch dimension if it was added
        if not is_batch:
            output = output[0]
            weights = weights[0]
        cache = (Q, K, V, weights)
        return output, weights, cache

    @staticmethod
    def scaled_dot_product_attention_backward(d_out, Q, K, V, weights):
        # add batch dimension if missing
        is_batch = Q.ndim == 3
        if not is_batch:
            Q = Q[None]
            K = K[None]
            V = V[None]
            d_out = d_out[None]
            weights = weights[None]

        batch_size, seq_len, d_k = Q.shape

        # ---- Gradient w.r.t. V ----
        dweights = d_out @ V.transpose(0, 2, 1)

        # ---- Gradient w.r.t. weights ----
        dV = weights.transpose(0, 2, 1) @ d_out

        # ---- Softmax backwards ----
        dscores = (dweights - (weights * dweights).sum(axis=-1, keepdims=True)) * weights

        # ---- Scale back for the sqrt(d_k) factor ----
        dscores /= np.sqrt(d_k)

        # ---- Gradient w.r.t. Q and K ----
        dQ = dscores @ K
        dK = dscores.transpose(0, 2, 1) @ Q

        # remove batch dimension if it was added
        if not is_batch:
            dQ = dQ[0]
            dK = dK[0]
            dV = dV[0]

        return dQ, dK, dV


class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps

        # affine parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        # gradient buffers
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # cache for backward pass
        self.cache = None

    def forward(self, x):
        # mean and std over last dimension
        mu = x.mean(axis=-1, keepdims=True)
        sigma = x.std(axis=-1, keepdims=True)

        # normalize values
        x_hat = (x - mu) / (sigma + self.eps)

        # affine transformation
        out = self.gamma * x_hat + self.beta

        # store cache for backward pass
        self.cache = dict(x=x, mu=mu, sigma=sigma, x_hat=x_hat)

        return out

    def backward(self, d_out):
        cache = self.cache
        x = cache['x']
        mu = cache['mu']
        sigma = cache['sigma']
        x_hat = cache['x_hat']

        D = x.shape[-1]                         # number of features normalized over
        reduce_axes = tuple(range(x.ndim-1))

        # ---- Gradients w.r.t. gamma and beta ----
        self.dgamma += np.sum(d_out * x_hat, axis=reduce_axes)
        self.dbeta += np.sum(d_out, axis=reduce_axes)

        # ---- Backprop through affine transformation ----
        dx_hat = d_out * self.gamma

        # ---- Backprop through normalization ----
        # derivative w.r.t. std
        d_sigma = np.sum(dx_hat * (x - mu) * (-0.5) * (sigma + self.eps) ** -3, axis=-1, keepdims=True)
        # derivative w.r.t. mean
        d_mu = np.sum(dx_hat * (-1 / (sigma + self.eps)), axis=-1, keepdims=True) + d_sigma * np.mean(-2 * (x - mu), axis=-1, keepdims=True)
        # derivative w.r.t. x
        dx = dx_hat / (sigma + self.eps) + d_sigma * 2 * (x - mu) / D + d_mu / D
        return dx

    def get_param_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def clear_grads(self):
        self.dgamma.fill(0)
        self.dbeta.fill(0)


class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        # Multi-head self-attention model
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Layer norms
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # Feed-forward network parameters
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

        # Gradients
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # Stores intermediate values for backward pass
        self.cache = None

    def forward(self, x, mask=None):
        # ---- Multi-Head Attention ----
        # self-attention
        mha_out = self.mha.forward(x, mask)

        # residual connection 1
        res1 = x + mha_out

        # layer norm 1
        x_norm1 = self.ln1.forward(res1)

        # ---- Feed-Forward Network ----
        # first linear projection
        ffn_pre = x_norm1 @ self.W1 + self.b1

        # ReLU activation
        ffn_act = np.maximum(0, ffn_pre)

        # second linear projection
        ffn_out = ffn_act @ self.W2 + self.b2

        # residual connection 2
        res2 = x_norm1 + ffn_out

        # layer norm 2
        x_norm2 = self.ln2.forward(res2)

        # store cache for backward pass
        self.cache = dict(x_norm1=x_norm1, ffn_pre=ffn_pre, ffn_act=ffn_act)
        return x_norm2

    def backward(self, d_block_out):
        cache = self.cache
        ffn_act = cache["ffn_act"]
        ffn_pre = cache["ffn_pre"]
        x_norm1 = cache["x_norm1"]

        # ---- Backprop through layer norm 2 ----
        d_x_norm2 = self.ln2.backward(d_block_out)

        # ---- Backprop through FFN ----
        # gradients for W2 and b2
        dln2_cpy = d_x_norm2.copy()
        bat_seq = dln2_cpy.reshape(-1, dln2_cpy.shape[-1])
        relu_flat = ffn_act.reshape(-1, ffn_act.shape[-1])
        self.dW2 += relu_flat.T @ bat_seq
        self.db2 += bat_seq.sum(axis=0)
        # backprop to activation
        d_ffn_act = d_x_norm2 @ self.W2.T
        # backprop through ReLU
        d_ffn_pre = d_ffn_act * (ffn_pre > 0)
        # gradients for W1 and b1
        ln1_out_flat = x_norm1.reshape(-1, x_norm1.shape[-1])
        d_pre1_flat = d_ffn_pre.reshape(-1, d_ffn_pre.shape[-1])
        self.dW1 += ln1_out_flat.T @ d_pre1_flat
        self.db1 += d_pre1_flat.sum(axis=0)
        # gradient w.r.t. layer norm 1 output
        d_ffn_input = d_ffn_pre @ self.W1.T
        # combine gradients from residual connection
        d_out1 = d_x_norm2 + d_ffn_input

        # ---- Backprop through layer norm 1 ----
        d_x_norm1 = self.ln1.backward(d_out1)

        # ---- Backprop through MHA ----
        # backprop through multi-head attention
        d_mha_out = d_x_norm1
        d_mha_input = self.mha.backward(d_mha_out)
        # combine gradients from residual connection
        d_x = d_x_norm1 + d_mha_input

        return d_x

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
    def __init__(self, d_model, num_layers, n_heads, d_ff):
        self.layers = [                                     # stack of encoder blocks
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        # Add batch dimension if missing
        if x.ndim == 2:
            x = x[None]

        for layer in self.layers:
            x = layer.forward(x, mask)

        # Remove batch dimension if it was added
        return x.squeeze(0) if x.shape[0] == 1 else x

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
        self.W = np.random.randn(d_model, num_classes) / np.sqrt(d_model)   # weights
        self.b = np.zeros(num_classes)                                      # bias

        # backpropagation variables
        self.enc_out = None                                                 # encoder output
        self.h = None                                                       # pooled representation
        self.logits = None                                                  # logits
        self.dW = np.zeros_like(self.W)                                     # weight gradients
        self.db = np.zeros_like(self.b)                                     # bias gradients

    def forward(self, enc_out):
        self.enc_out = enc_out

        # mean pooling
        if enc_out.ndim == 3:
            self.h = enc_out.mean(axis=1)
        else:
            self.h = enc_out.mean(axis=0)

        # affine transformation
        self.logits = self.h @ self.W + self.b

        # softmax
        probs = softmax(self.logits)

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
            d_cls_out = np.repeat(dh[:, None, :], seq, axis=1) / seq
        else:
            seq = self.enc_out.shape[0]
            d_cls_out = np.repeat(dh[None, :], seq, axis=0) / seq

        return d_cls_out

    def get_param_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def clear_grads(self):
        self.dW.fill(0)
        self.db.fill(0)


class TransformerClassifier:
    def __init__(self, d_model, n_layers, n_heads, d_ff, n_classes):
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff)
        self.classifier = Classifier(d_model, n_classes)

    def forward(self, x, mask=None):
        enc_out = self.encoder.forward(x, mask[:, None, :] if mask is not None else None)
        logits = self.classifier.forward(enc_out)
        return logits

    def backward(self, d_logits):
        d_cls_out = self.classifier.backward(d_logits)
        self.encoder.backward(d_cls_out)

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
