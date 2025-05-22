import torch
import torch.nn.functional as F
from torch.optim import Adam
import time


class DynamicTSNE:
    def __init__(
            self,
            output_dims=2,
            verbose=True,
    ):
        self.output_dims = output_dims
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_affinities(self, Xs, perplexity=30.0, k_neighbors=90):
        def Hbeta(D, beta):
            P = torch.exp(-D * beta)
            sumP = torch.sum(P)
            sumP = torch.clamp(sumP, min=1e-8)
            H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
            P = torch.clamp(P / sumP, min=1e-8)
            return H, P

        def compute_P(X, init_beta=None):
            t0 = time.time()

            n = X.shape[0]
            D = torch.cdist(X, X, p=2).pow(2)
            P = torch.zeros((n, n), device=X.device)
            beta = init_beta.clone() if init_beta is not None else torch.ones(n, device=X.device)
            logU = torch.log(torch.tensor(perplexity, device=X.device))
            all_tries = 0

            for i in range(n):
                distances = D[i]
                topk = torch.topk(distances, k=k_neighbors + 1, largest=False)
                idx = topk.indices[topk.indices != i][:k_neighbors]
                Di = torch.clamp(distances[idx], max=1e3)

                betamin, betamax = None, None
                H, thisP = Hbeta(Di, beta[i])
                tries = 0
                while torch.abs(H - logU) > 1e-5 and tries < 50:
                    if H > logU:
                        betamin = beta[i].clone()
                        beta[i] = beta[i] * 2 if betamax is None else (beta[i] + betamax) / 2
                    else:
                        betamax = beta[i].clone()
                        beta[i] = beta[i] / 2 if betamin is None else (beta[i] + betamin) / 2
                    H, thisP = Hbeta(Di, beta[i])
                    tries += 1
                all_tries += tries
                P[i, idx] = thisP

            if self.verbose:
                print(f"Total affinity computation time: {time.time() - t0:.2f}s, {all_tries / n} Tries")

            P = (P + P.T) / (2 * n)
            return P, beta

        X_tensor = [torch.tensor(X, device=self.device) for X in Xs]
        self.Xs = X_tensor

        Ps = []
        prev_beta = None
        for X in X_tensor:
            P, prev_beta = compute_P(X, prev_beta)
            Ps.append(P)

        self.Ps = torch.stack(Ps)
        assert not torch.isnan(self.Ps).any(), "Affinity matrix has NaN"

    def fit(self, n_epochs=1000, exaggeration=12.0, exaggeration_epochs=250, lr=200.0, lambd=0.1):
        T = len(self.Xs)
        n = self.Xs[0].shape[0]

        Y_init = []
        for X in self.Xs:
            X_cpu = X.detach().cpu().numpy()
            pca = PCA(n_components=self.output_dims)
            Y_pca = pca.fit_transform(X_cpu)
            Y_init.append(torch.tensor(Y_pca, device=self.device, dtype=torch.float32))

        Y = torch.stack(Y_init)
        Y.requires_grad_()

        optimizer = Adam([Y], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            total_loss = 0

            if epoch < exaggeration_epochs:
                P_use = self.Ps * exaggeration
            else:
                P_use = self.Ps

            for t in range(T):
                Qt, _ = self._compute_lowdim_affinities(Y[t])
                loss = self._kl_divergence(P_use[t], Qt)
                if t > 0:
                    loss += lambd * F.mse_loss(Y[t], Y[t - 1])
                total_loss += loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([Y], max_norm=10.0)
            optimizer.step()
            scheduler.step()

            if self.verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

        return [Y[t].detach().cpu().numpy() for t in range(T)]

    def _compute_lowdim_affinities(self, Y):
        num = 1 / (1 + torch.cdist(Y, Y, p=2).pow(2))
        num.fill_diagonal_(0.0)
        Q = torch.clamp(num / num.sum(), min=1e-5)
        return Q, num

    def _kl_divergence(self, P, Q):
        return torch.sum(P * torch.log((P + 1e-8) / (Q + 1e-8)))


import numpy as np
from openTSNE import TSNE
from sklearn.decomposition import PCA


class DynamicTSNE_2:
    def __init__(self, perplexity=30, n_iter=1000, init='pca', random_state=None):
        """
        init: 'pca', 'random', or 'previous'
        """
        assert init in ['pca', 'random', 'previous']
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.init = init
        self.random_state = random_state

    def fit_transform(self, Xs):
        """
        Xs: List of np.ndarray (each shape: [n_samples, n_features])
        Returns: List of np.ndarray (each shape: [n_samples, 2])
        """
        embeddings = []
        previous_embedding = None

        for i, X in enumerate(Xs):
            print(i)
            if i == 0 or self.init == 'pca':
                init_embedding = PCA(n_components=2).fit_transform(X) if self.init != 'random' else 'random'
            else:
                init_embedding = previous_embedding

            tsne = TSNE(
                n_jobs=-1,
                perplexity=self.perplexity,
                n_iter=self.n_iter,
                initialization=init_embedding,
                random_state=self.random_state,
                verbose=True
            )
            embedding = tsne.fit(X)
            embeddings.append(embedding)
            previous_embedding = embedding

        return embeddings