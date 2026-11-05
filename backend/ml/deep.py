import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class _ANN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)


class _CNN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class _RNN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        _, h = self.gru(x.unsqueeze(2))
        return self.head(h[-1])


class _CNN2D(nn.Module):
    def __init__(self, n_channels, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, n_out))

    def forward(self, x):
        return self.head(self.conv(x))


_ARCHS = {"ann": _ANN, "cnn": _CNN, "rnn": _RNN, "cnn2d": _CNN2D}


class _TorchBase(BaseEstimator):
    def __init__(self, arch="ann", lr=0.003, epochs=40, batch_size=64, patience=6):
        self.arch = arch
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

    def _train(self, X, y, n_out, loss_fn):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        X = X if torch.is_tensor(X) else torch.tensor(np.asarray(X, dtype=np.float32))
        y = torch.tensor(y)
        idx = np.arange(len(X))
        if len(X) >= 50:
            tr, va = train_test_split(idx, test_size=0.1, random_state=42)
        else:
            tr, va = idx, idx
        net = self._build_net(X, n_out)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(self.epochs):
            net.train()
            perm = np.random.RandomState(0).permutation(tr)
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(net(X[b]), y[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(X[va]), y[va]).item()
            if val < best - 1e-4:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net

    def _build_net(self, X, n_out):
        return _ARCHS[self.arch](X.shape[1], n_out)

    def _forward(self, X):
        X = X if torch.is_tensor(X) else torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            return self.net_(X)


class TorchClassifier(_TorchBase, ClassifierMixin):
    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self._train(X, self.le_.transform(y).astype(np.int64), len(self.classes_), nn.CrossEntropyLoss())
        return self

    def predict_proba(self, X):
        return torch.softmax(self._forward(X), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchRegressor(_TorchBase, RegressorMixin):
    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = float(y.mean()), float(y.std()) or 1.0
        t = ((y - self.mu_) / self.sd_).reshape(-1, 1)
        self._train(X, t, 1, nn.MSELoss())
        return self

    def predict(self, X):
        return self._forward(X).numpy().ravel() * self.sd_ + self.mu_


class TorchImageClassifier(TorchClassifier):
    """CNN for images. Input X: uint8 array (N, H, W, 3)."""

    def __init__(self, arch="cnn2d", lr=0.003, epochs=15, batch_size=32, patience=4):
        super().__init__(arch=arch, lr=lr, epochs=epochs, batch_size=batch_size, patience=patience)

    @staticmethod
    def _prep(X):
        return np.asarray(X, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0

    def fit(self, X, y):
        return super().fit(self._prep(X), y)

    def predict_proba(self, X):
        return super().predict_proba(self._prep(X))


class _GRUText(nn.Module):
    def __init__(self, vocab_size, n_out):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, n_out))

    def forward(self, x):
        _, h = self.gru(self.emb(x))
        return self.head(torch.cat([h[0], h[1]], dim=1))


class _TextCNN(nn.Module):
    def __init__(self, vocab_size, n_out):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(64, 32, k, padding=k // 2) for k in (3, 4, 5)])
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(96, n_out))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.head(torch.cat(pooled, dim=1))


_TEXT_ARCHS = {"gru": _GRUText, "textcnn": _TextCNN}
_WORD = re.compile(r"[a-z0-9']+")


class TorchTextClassifier(TorchClassifier):
    """RNN (GRU) or TextCNN for text. Input X: list/array of strings."""

    def __init__(self, arch="gru", lr=0.003, epochs=15, batch_size=32, patience=4,
                 max_words=8000, max_len=100):
        super().__init__(arch=arch, lr=lr, epochs=epochs, batch_size=batch_size, patience=patience)
        self.max_words = max_words
        self.max_len = max_len

    def _encode(self, X):
        out = np.zeros((len(X), self.max_len), dtype=np.int64)
        for i, text in enumerate(X):
            ids = [self.vocab_.get(w, 1) for w in _WORD.findall(str(text).lower())][: self.max_len]
            out[i, : len(ids)] = ids
        return torch.tensor(out)

    def _build_net(self, X, n_out):
        return _TEXT_ARCHS[self.arch](len(self.vocab_) + 2, n_out)

    def fit(self, X, y):
        counts = Counter(w for t in X for w in _WORD.findall(str(t).lower()))
        self.vocab_ = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(self.max_words))}
        return super().fit(self._encode(X), y)

    def predict_proba(self, X):
        return super().predict_proba(self._encode(X))