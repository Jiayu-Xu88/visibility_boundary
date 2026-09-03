from __future__ import annotations

import math

import torch
from torch import nn


class MeanPoolMIL(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=0)
        return self.classifier(pooled)


class ABMIL(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, attention_dim: int = 128) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = attention @ embedded
        return self.classifier(pooled.squeeze(0))

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        attention = self.attention_w(torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded)))
        attention = torch.softmax(attention.transpose(0, 1), dim=1)
        return attention, embedded


class CLAMLite(ABMIL):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512, attention_dim: int = 256) -> None:
        super().__init__(feature_dim, num_classes, hidden_dim, attention_dim)


class PrototypeGatedCLAMLite(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        gate_alpha: float = 0.5,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.gate_alpha = gate_alpha

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = attention @ embedded
        return self.classifier(pooled.squeeze(0))

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).squeeze(1)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        prototype_score = (feature_norm @ prototype_norm.T).max(dim=1).values
        gated_logits = base_logits + self.gate_alpha * prototype_score
        attention = torch.softmax(gated_logits.unsqueeze(0), dim=1)
        return attention, embedded


class MorphologyMixtureOfExpertsMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        prototype_alpha: float = 1.0,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        gate_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        self.num_prototypes = int(prototype_bank.shape[0])
        self.num_experts = self.num_prototypes + 1
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, self.num_experts)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.num_prototypes),
            nn.Linear(self.num_prototypes, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, self.num_experts),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.prototype_alpha = prototype_alpha

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded, expert_weights = self.expert_attention(features)
        expert_embeddings = attention @ embedded
        pooled = torch.sum(expert_weights.unsqueeze(1) * expert_embeddings, dim=0)
        return self.classifier(pooled)

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention, embedded, expert_weights = self.expert_attention(features)
        combined_attention = torch.sum(expert_weights.unsqueeze(1) * attention, dim=0, keepdim=True)
        return combined_attention, embedded

    def expert_attention(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).transpose(0, 1)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        similarity = feature_norm @ prototype_norm.T

        prototype_logits = torch.zeros_like(base_logits)
        prototype_logits[1:, :] = similarity.T
        attention = torch.softmax(base_logits + self.prototype_alpha * prototype_logits, dim=1)

        morphology_fraction = torch.softmax(similarity, dim=1).mean(dim=0)
        expert_weights = torch.softmax(self.gate(morphology_fraction), dim=0)
        return attention, embedded, expert_weights


class PrototypeCalibratedCLAMLite(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        class_prototype_indices: list[list[int]],
        gate_alpha: float = 0.0,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        if len(class_prototype_indices) != num_classes:
            raise ValueError("class_prototype_indices must provide one prototype-index list per class.")
        class_centers = []
        for indices in class_prototype_indices:
            if not indices:
                raise ValueError("Each class needs at least one prototype index for prototype-calibrated head.")
            class_centers.append(prototype_bank[indices].float().mean(dim=0))
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.prototype_projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.register_buffer("class_centers", torch.stack(class_centers, dim=0))
        self.gate_alpha = gate_alpha

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = (attention @ embedded).squeeze(0)
        pooled_norm = torch.nn.functional.normalize(pooled, dim=0)
        class_centers = self.prototype_projector(self.class_centers)
        class_centers_norm = torch.nn.functional.normalize(class_centers, dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (class_centers_norm @ pooled_norm) + self.class_bias

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).squeeze(1)
        if self.gate_alpha:
            feature_norm = torch.nn.functional.normalize(features, dim=1)
            prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
            prototype_score = (feature_norm @ prototype_norm.T).max(dim=1).values
            base_logits = base_logits + self.gate_alpha * prototype_score
        attention = torch.softmax(base_logits.unsqueeze(0), dim=1)
        return attention, embedded


class MorphologyPrototypeTokenMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        num_free_tokens: int = 2,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        self.num_prototype_tokens = int(prototype_bank.shape[0])
        self.num_free_tokens = int(num_free_tokens)
        self.num_tokens = self.num_prototype_tokens + self.num_free_tokens
        self.patch_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.token_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        if self.num_free_tokens:
            free_tokens = torch.randn(self.num_free_tokens, feature_dim) / math.sqrt(feature_dim)
            initial_tokens = torch.cat([prototype_bank.float(), free_tokens], dim=0)
        else:
            initial_tokens = prototype_bank.float()
        self.evidence_tokens = nn.Parameter(initial_tokens)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.num_tokens),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_tokens, num_classes),
        )
        self._last_attention: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embeddings(features)
        return self.classifier(token_embeddings.flatten())

    def token_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_proj(features).unsqueeze(0)
        query_tokens = self.token_proj(self.evidence_tokens).unsqueeze(0)
        attended, attention = self.cross_attention(
            query_tokens,
            patch_embeddings,
            patch_embeddings,
            need_weights=True,
            average_attn_weights=False,
        )
        token_embeddings = self.token_norm(query_tokens + attended)
        token_embeddings = self.token_norm(token_embeddings + self.token_ffn(token_embeddings))
        self._last_attention = attention.detach()
        return token_embeddings.squeeze(0)

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = self.token_embeddings(features)
        if self._last_attention is None:
            raise RuntimeError("Missing attention weights after token embedding computation.")
        patch_attention = self._last_attention.mean(dim=(1, 2))
        patch_attention = patch_attention / patch_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return patch_attention, token_embeddings

    def token_diversity_loss(self) -> torch.Tensor:
        token_norm = torch.nn.functional.normalize(self.token_proj(self.evidence_tokens), dim=1)
        similarity = token_norm @ token_norm.T
        off_diagonal = similarity - torch.eye(self.num_tokens, device=similarity.device, dtype=similarity.dtype)
        return off_diagonal.pow(2).sum() / (self.num_tokens * (self.num_tokens - 1))


class SubtypePrototypeClassAttentionMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        class_prototype_indices: list[list[int]],
        prototype_alpha: float = 1.0,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        if len(class_prototype_indices) != num_classes:
            raise ValueError("class_prototype_indices must provide one prototype-index list per class.")
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, num_classes)
        self.classifier_weight = nn.Parameter(torch.randn(num_classes, hidden_dim) / math.sqrt(hidden_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.class_prototype_indices = [list(indices) for indices in class_prototype_indices]
        self.prototype_alpha = prototype_alpha

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        class_embeddings = attention @ embedded
        return (class_embeddings * self.classifier_weight).sum(dim=1) + self.classifier_bias

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).transpose(0, 1)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        similarity = feature_norm @ prototype_norm.T
        prototype_logits = []
        for indices in self.class_prototype_indices:
            if not indices:
                prototype_logits.append(torch.zeros(features.shape[0], device=features.device, dtype=features.dtype))
            else:
                prototype_logits.append(similarity[:, indices].max(dim=1).values)
        prototype_logits_tensor = torch.stack(prototype_logits, dim=0)
        attention = torch.softmax(base_logits + self.prototype_alpha * prototype_logits_tensor, dim=1)
        return attention, embedded


class ProtoABMIL(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_prototypes: int = 12,
        prototype_temperature: float = 0.2,
        prototype_bank: torch.Tensor | None = None,
        freeze_prototypes: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        if prototype_bank is None:
            prototype_bank = torch.randn(num_prototypes, feature_dim) / math.sqrt(feature_dim)
        else:
            if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
                raise ValueError(
                    f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}"
                )
            num_prototypes = int(prototype_bank.shape[0])
        if freeze_prototypes:
            self.register_buffer("prototype_bank", prototype_bank.float())
        else:
            self.prototype_bank = nn.Parameter(prototype_bank.float())
        self.prototype_temperature = prototype_temperature
        self.prototype_norm = nn.LayerNorm(num_prototypes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + num_prototypes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = attention @ embedded
        histogram = self.prototype_histogram(features)
        representation = torch.cat([pooled.squeeze(0), histogram], dim=0)
        return self.classifier(representation)

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        attention = self.attention_w(torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded)))
        attention = torch.softmax(attention.transpose(0, 1), dim=1)
        return attention, embedded

    def prototype_histogram(self, features: torch.Tensor) -> torch.Tensor:
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        assignment = torch.softmax((feature_norm @ prototype_norm.T) / self.prototype_temperature, dim=1)
        return self.prototype_norm(assignment.mean(dim=0))

    def prototype_attention_distribution(self, features: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention_scores(features)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        assignment = torch.softmax((feature_norm @ prototype_norm.T) / self.prototype_temperature, dim=1)
        return (attention @ assignment).squeeze(0)


class TransMILLite(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.normal_(self.cls_token, std=1.0 / math.sqrt(hidden_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.input(features).unsqueeze(0)
        cls_token = self.cls_token.expand(embedded.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls_token, embedded], dim=1))
        return self.classifier(encoded[:, 0, :].squeeze(0))


class PrototypeGatedCLAMLitePerAtomTau(nn.Module):
    """Per-atom learnable temperature variant of PrototypeGatedCLAMLite.

    Added under MODEL_LOCK_CHANGELOG.md entry 001 (2026-05-06).
    The original PrototypeGatedCLAMLite class above is byte-for-byte
    unchanged. This new class scales the per-atom cosine similarity by
    a learnable per-atom temperature tau_k = exp(log_tau_k) before
    taking the max over k:

        prototype_score_i = max_k (cos(x_i, p_k) / tau_k)

    log_tau_k is initialised to 0 so tau_k = 1 at init, recovering
    the original head numerically. This is required for the
    regression test in test_per_atom_tau_equivalence.py.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        gate_alpha: float = 1.0,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.gate_alpha = gate_alpha
        self.log_tau = nn.Parameter(torch.zeros(int(prototype_bank.shape[0])))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = attention @ embedded
        return self.classifier(pooled.squeeze(0))

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).squeeze(1)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        tau = torch.exp(self.log_tau)
        prototype_score = ((feature_norm @ prototype_norm.T) / tau.unsqueeze(0)).max(dim=1).values
        gated_logits = base_logits + self.gate_alpha * prototype_score
        attention = torch.softmax(gated_logits.unsqueeze(0), dim=1)
        return attention, embedded


class PrototypeGatedCLAMLiteSoftTopR(nn.Module):
    """Soft sparse-prototype gated variant of PrototypeGatedCLAMLite.

    Pre-registered in PRE_REG_SOFT_SPARSE_PROTOTYPE_GATE_V1.md (2026-05-17).
    The original PrototypeGatedCLAMLite (line 51) and
    PrototypeGatedCLAMLitePerAtomTau (line 433) are byte-for-byte
    unchanged.

    The hard-max prototype gate

        prototype_score_i = max_k cos(x_i, p_k)

    masks the gradient on K-1 prototypes per patch. This class replaces
    it with a top-r softmax-weighted sum over the r most-similar
    prototypes per patch:

        s_{ik}   = cos(x_i, p_k)
        Top-r(i) = argtopk_k s_{ik}                    # size r
        w_{ik}   = softmax_{k in Top-r(i)} (beta * s_{ik})
        g_i      = sum_{k in Top-r(i)} w_{ik} * s_{ik}

    with `gated_attn_logit_i = base_attn_logit_i + alpha * g_i`.

    Equivalence: at top_r=1 (and any beta), softmax over a single
    element is 1.0, so g_i collapses exactly to max_k s_{ik}, i.e. the
    original PrototypeGatedCLAMLite. This is verified bit-for-bit in
    test_soft_topr_equivalence.py.

    Hyperparameters:
        top_r: number of prototypes selected per patch (default 4,
            matching the SAE encoder's k_active=4).
        beta:  inverse temperature for the softmax over the top-r
            cosines. Default 5.0. If `learn_beta=True`, the
            log-parameterised beta is trainable (initialised to log(5)).
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        gate_alpha: float = 1.0,
        top_r: int = 4,
        beta: float = 5.0,
        learn_beta: bool = False,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        K = int(prototype_bank.shape[0])
        if not (1 <= int(top_r) <= K):
            raise ValueError(f"top_r must be in [1, K={K}], got {top_r}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.gate_alpha = float(gate_alpha)
        self.top_r = int(top_r)
        if learn_beta:
            self.log_beta = nn.Parameter(torch.log(torch.tensor(float(beta))))
            self._beta_fixed = None
        else:
            self.log_beta = None
            self._beta_fixed = float(beta)

    @property
    def beta(self) -> torch.Tensor:
        if self.log_beta is not None:
            return torch.exp(self.log_beta)
        return torch.tensor(self._beta_fixed, dtype=self.prototype_bank.dtype, device=self.prototype_bank.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention, embedded = self.attention_scores(features)
        pooled = attention @ embedded
        return self.classifier(pooled.squeeze(0))

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).squeeze(1)
        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        similarity = feature_norm @ prototype_norm.T  # [N, K]
        topr_vals, _ = similarity.topk(self.top_r, dim=1)  # [N, r]
        weights = torch.softmax(self.beta * topr_vals, dim=1)
        prototype_score = (weights * topr_vals).sum(dim=1)
        gated_logits = base_logits + self.gate_alpha * prototype_score
        attention = torch.softmax(gated_logits.unsqueeze(0), dim=1)
        return attention, embedded


class ResidualClassSparseConceptAttentionMIL(nn.Module):
    """Residual class-conditioned sparse concept attention MIL.

    This model keeps the stable shared CLAM-lite attention path and adds a
    small class-conditioned concept-attention residual. The residual mixing
    coefficient is initialised near zero, so the model starts close to a shared
    attention baseline and only uses class-specific concept evidence when it is
    useful during training.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        prototype_bank: torch.Tensor,
        concept_alpha: float = 1.0,
        residual_init: float = 0.05,
        concept_gate_init: float = -3.0,
        hidden_dim: int = 512,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[1] != feature_dim:
            raise ValueError(f"Expected prototype bank [K, {feature_dim}], got {tuple(prototype_bank.shape)}")
        if not 0.0 < residual_init < 1.0:
            raise ValueError(f"residual_init must be in (0, 1), got {residual_init}")
        self.embed = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.attention_v = nn.Linear(hidden_dim, attention_dim)
        self.attention_u = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier_weight = nn.Parameter(torch.randn(num_classes, hidden_dim) / math.sqrt(hidden_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        self.class_concept_logits = nn.Parameter(
            torch.full((num_classes, int(prototype_bank.shape[0])), float(concept_gate_init))
        )
        residual_logit = math.log(residual_init / (1.0 - residual_init))
        self.residual_logit = nn.Parameter(torch.full((num_classes,), float(residual_logit)))
        self.register_buffer("prototype_bank", prototype_bank.float())
        self.concept_alpha = float(concept_alpha)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared_attention, class_attention, embedded = self._attentions(features)
        shared_embedding = (shared_attention @ embedded).squeeze(0)
        class_embeddings = class_attention @ embedded
        rho = torch.sigmoid(self.residual_logit).unsqueeze(1)
        mixed_embeddings = (1.0 - rho) * shared_embedding.unsqueeze(0) + rho * class_embeddings
        return (mixed_embeddings * self.classifier_weight).sum(dim=1) + self.classifier_bias

    def attention_scores(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_attention, _, embedded = self._attentions(features)
        return shared_attention, embedded

    def _attentions(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embed(features)
        base_logits = self.attention_w(
            torch.tanh(self.attention_v(embedded)) * torch.sigmoid(self.attention_u(embedded))
        ).squeeze(1)
        shared_attention = torch.softmax(base_logits.unsqueeze(0), dim=1)

        feature_norm = torch.nn.functional.normalize(features, dim=1)
        prototype_norm = torch.nn.functional.normalize(self.prototype_bank, dim=1)
        similarity = feature_norm @ prototype_norm.T
        concept_gates = torch.sigmoid(self.class_concept_logits)
        gate_mass = concept_gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        class_evidence = similarity @ (concept_gates / gate_mass).T
        class_logits = base_logits.unsqueeze(0) + self.concept_alpha * class_evidence.T
        class_attention = torch.softmax(class_logits, dim=1)
        return shared_attention, class_attention, embedded

    def concept_sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.class_concept_logits).mean()

    def residual_sparsity_loss(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit).mean()


def build_model(model_name: str, feature_dim: int, num_classes: int, config: dict | None = None) -> nn.Module:
    if model_name == "mean":
        return MeanPoolMIL(feature_dim, num_classes)
    if model_name == "abmil":
        return ABMIL(feature_dim, num_classes)
    if model_name == "clam_lite":
        return CLAMLite(feature_dim, num_classes)
    if model_name == "proto_gated_clam_lite":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("proto_gated_clam_lite requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return PrototypeGatedCLAMLite(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            gate_alpha=float(config.get("prototype_gate_alpha", 0.5)),
        )
    if model_name == "morph_moe_mil":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("morph_moe_mil requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return MorphologyMixtureOfExpertsMIL(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            prototype_alpha=float(config.get("prototype_alpha", 1.0)),
            gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
        )
    if model_name == "proto_calibrated_clam_lite":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("proto_calibrated_clam_lite requires config['prototype_bank_path']")
        if "class_prototype_indices" not in config:
            raise ValueError("proto_calibrated_clam_lite requires config['class_prototype_indices']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return PrototypeCalibratedCLAMLite(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            class_prototype_indices=config["class_prototype_indices"],
            gate_alpha=float(config.get("prototype_gate_alpha", 0.0)),
        )
    if model_name == "mpt_mil":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("mpt_mil requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return MorphologyPrototypeTokenMIL(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            num_free_tokens=int(config.get("num_free_tokens", 2)),
            hidden_dim=int(config.get("hidden_dim", 512)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.25)),
        )
    if model_name == "spca_mil":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("spca_mil requires config['prototype_bank_path']")
        if "class_prototype_indices" not in config:
            raise ValueError("spca_mil requires config['class_prototype_indices']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return SubtypePrototypeClassAttentionMIL(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            class_prototype_indices=config["class_prototype_indices"],
            prototype_alpha=float(config.get("prototype_alpha", 1.0)),
        )
    if model_name == "proto_abmil":
        return ProtoABMIL(feature_dim, num_classes)
    if model_name == "fixed_proto_abmil":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("fixed_proto_abmil requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return ProtoABMIL(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            freeze_prototypes=bool(config.get("freeze_prototypes", True)),
            prototype_temperature=float(config.get("prototype_temperature", 0.2)),
        )
    if model_name == "transmil_lite":
        return TransMILLite(feature_dim, num_classes)
    if model_name == "proto_gated_clam_lite_tau":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("proto_gated_clam_lite_tau requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return PrototypeGatedCLAMLitePerAtomTau(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            gate_alpha=float(config.get("prototype_gate_alpha", 1.0)),
        )
    if model_name == "proto_gated_clam_lite_soft_topr":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("proto_gated_clam_lite_soft_topr requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return PrototypeGatedCLAMLiteSoftTopR(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            gate_alpha=float(config.get("prototype_gate_alpha", 1.0)),
            top_r=int(config.get("top_r", 4)),
            beta=float(config.get("beta", 5.0)),
            learn_beta=bool(config.get("learn_beta", False)),
        )
    if model_name == "r_csca_mil":
        if not config or not config.get("prototype_bank_path"):
            raise ValueError("r_csca_mil requires config['prototype_bank_path']")
        payload = torch.load(config["prototype_bank_path"], map_location="cpu")
        centers = payload["centers"] if isinstance(payload, dict) and "centers" in payload else payload
        return ResidualClassSparseConceptAttentionMIL(
            feature_dim,
            num_classes,
            prototype_bank=centers,
            concept_alpha=float(config.get("concept_alpha", 1.0)),
            residual_init=float(config.get("residual_init", 0.05)),
            concept_gate_init=float(config.get("concept_gate_init", -3.0)),
            hidden_dim=int(config.get("hidden_dim", 512)),
            attention_dim=int(config.get("attention_dim", 256)),
        )
    raise ValueError(f"Unknown model: {model_name}")
