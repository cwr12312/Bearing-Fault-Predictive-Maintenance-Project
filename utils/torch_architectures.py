"""
utils/torch_architectures.py
=============================
Exact PyTorch `nn.Module` definitions copied from the original training
notebooks (meta_sgd_model.py, maml.py, fbcl_model.py) so the saved
`state_dict()` checkpoints can be loaded and used for inference without
any retraining. Do not change layer names/shapes here — they must match
the keys stored in the .pth files.
"""

from __future__ import annotations         # allows modern type-hint syntax (e.g. X | None) on older Python versions

import torch                                 # core PyTorch tensor library
import torch.nn as nn                        # neural network layer building blocks
import torch.nn.functional as F              # imported for availability (functional ops), not directly used below


# ---------------------------------------------------------------------
# Meta-SGD  (meta_sgd_model.py -> FaultTransformer)
# ---------------------------------------------------------------------
class FaultTransformer(nn.Module):
    """Each input feature becomes a token; self-attention learns feature interactions."""

    def __init__(self, n_feat, n_classes, d_model=64, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()
        self.embed = nn.Linear(1, d_model)                              # projects each scalar feature value into a d_model-dim token embedding
        self.pos = nn.Parameter(torch.randn(1, n_feat, d_model) * 0.01)  # learnable positional embedding, one per feature/token
        self.drop_in = nn.Dropout(dropout)                                # dropout applied right after embedding + positional encoding

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )                                                                  # a single pre-norm transformer encoder layer
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)     # stack of n_layers transformer encoder layers
        self.norm_out = nn.LayerNorm(d_model)                                # final layer norm after the encoder stack

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )                                                                     # classification head mapping pooled representation -> class logits

    def forward(self, x):
        x = x.unsqueeze(-1)                    # (batch, n_feat) -> (batch, n_feat, 1): each feature becomes a scalar "token"
        x = self.embed(x) + self.pos             # embed each token and add its learned positional encoding
        x = self.drop_in(x)                        # regularize the input embeddings
        x = self.encoder(x)                          # run self-attention across all feature-tokens
        x = self.norm_out(x)                           # normalize the encoder output
        x = x.mean(dim=1)                                # mean-pool across tokens/features -> single vector per sample
        return self.head(x)                                # classify the pooled representation


# ---------------------------------------------------------------------
# MAML  (maml.py -> OptimizedmamlTransformer)
# ---------------------------------------------------------------------
class OptimizedmamlTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=128, n_heads=4, n_layers=4, dropout=0.3):
        super().__init__()
        self.n_features = n_features                                        # number of input features (tokens, excluding CLS)
        self.d_model = d_model                                                # embedding dimension used throughout the model
        self.n_classes = n_classes                                             # number of output fault classes

        self.feature_embedding = nn.Linear(1, d_model)                          # projects each scalar feature into a d_model-dim token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)        # learnable token substituted in for masked-out features (used in pretraining)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)          # learnable [CLS]-style token prepended for classification
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features + 1, d_model) * 0.02)  # positional embedding covering CLS token + all features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 8,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )                                                                          # single pre-norm transformer encoder layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)  # stack of n_layers transformer encoder layers
        self.layer_norm = nn.LayerNorm(d_model)                                      # final layer norm after the encoder stack
        self.dropout = nn.Dropout(dropout)                                            # dropout applied after adding positional embeddings

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1),
        )                                                                              # self-supervised pretraining head: reconstructs masked feature values

        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.BatchNorm1d(d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2), nn.BatchNorm1d(d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )                                                                              # fine-tuning head: maps CLS representation -> class logits

        self.pretraining_mode = True  # switched to False for inference below

    def forward(self, x, mask=None):
        batch_size, n_feat = x.shape                                    # x shape: (batch, n_features)
        x = x.unsqueeze(-1)                                                # (batch, n_feat) -> (batch, n_feat, 1)
        x = self.feature_embedding(x)                                       # embed each scalar feature into a d_model-dim token

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model)              # broadcast the boolean mask across the embedding dimension
            mask_token_expanded = self.mask_token.expand(batch_size, n_feat, self.d_model)  # broadcast the learnable mask token to match batch/feature shape
            x = torch.where(mask_expanded, mask_token_expanded, x)                          # replace masked positions with the mask token (pretraining only)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)             # broadcast the CLS token to every sample in the batch
        x = torch.cat([cls_tokens, x], dim=1)                                # prepend CLS token to the sequence of feature tokens
        x = x + self.pos_embedding[:, : x.size(1), :]                          # add positional embeddings (sliced to current sequence length)
        x = self.dropout(x)                                                      # regularize the embedded sequence
        x = self.transformer(x)                                                    # run self-attention across CLS + feature tokens
        x = self.layer_norm(x)                                                       # normalize the encoder output

        if self.pretraining_mode:
            reconstructions = self.reconstruction_head(x[:, 1:, :])                    # pretraining: reconstruct each (non-CLS) feature token
            return reconstructions.squeeze(-1)                                           # drop the trailing singleton dim -> (batch, n_features)

        cls_representation = x[:, 0]                                                       # fine-tuning/inference: use only the CLS token's output
        return self.classification_head(cls_representation)                                  # classify based on the CLS representation

    def switch_to_finetune(self):
        self.pretraining_mode = False                                                          # flips the model from self-supervised pretraining to classification mode


# ---------------------------------------------------------------------
# FBCL  (fbcl_model.py -> OptimizedFBCLTransformer)
# ---------------------------------------------------------------------
class OptimizedFBCLTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=256, n_heads=16, n_layers=6, dropout=0.15):
        super().__init__()
        self.n_features = n_features                                     # number of input features (tokens)
        self.d_model = d_model                                             # embedding dimension used throughout the model
        self.n_classes = n_classes                                          # number of output fault classes

        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model)
        )                                                                     # two-layer MLP embedding for each scalar feature (richer than a single Linear)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)  # learnable positional embedding, one per feature/token
        self.pos_scale = nn.Parameter(torch.ones(1))                             # learnable scalar controlling how strongly positional info is applied

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )                                                                            # single pre-norm transformer encoder layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)   # stack of n_layers transformer encoder layers
        self.layer_norm = nn.LayerNorm(d_model)                                          # final layer norm after the encoder stack
        self.dropout = nn.Dropout(dropout)                                                # dropout applied after adding positional embeddings

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.BatchNorm1d(d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2), nn.BatchNorm1d(d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )                                                                                    # primary classification head -> class logits

        self.boosting_heads = nn.ModuleList()                                                  # additional heads added later via add_boosting_head() (boosting ensemble)

    def add_boosting_head(self):
        head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2), nn.BatchNorm1d(self.d_model * 2), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(self.d_model * 2, self.d_model), nn.BatchNorm1d(self.d_model), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(self.d_model, self.n_classes),
        )                                                                                        # a new auxiliary head, same output shape as the classifier
        self.boosting_heads.append(head)                                                            # register it as a submodule so its weights load/save correctly
        return head                                                                                   # return the new head (e.g. for external training)

    def forward(self, x, use_boosting=True):
        batch_size, n_feat = x.shape                                    # x shape: (batch, n_features)
        x = x.unsqueeze(-1)                                                # (batch, n_feat) -> (batch, n_feat, 1)
        x = self.feature_embedding(x)                                       # embed each scalar feature via the 2-layer MLP
        x = x + self.pos_scale * self.pos_embedding[:, :n_feat, :]            # add scaled positional embeddings
        x = self.dropout(x)                                                     # regularize the embedded sequence
        x = self.transformer(x)                                                   # run self-attention across feature tokens
        x = self.layer_norm(x)                                                      # normalize the encoder output
        x = x.mean(dim=1)                                                             # mean-pool across tokens/features -> single vector per sample

        logits = self.classifier(x)                                                     # base classification logits
        if use_boosting and len(self.boosting_heads) > 0:
            for head in self.boosting_heads:
                logits = logits + head(x)                                                  # boosting: add each auxiliary head's contribution to the logits
        return logits                                                                         # final (possibly boosted) class logits


def count_boosting_heads(state_dict: dict) -> int:
    """Infer how many boosting heads were saved in an FBCL checkpoint."""
    indices = set()                                              # collects unique boosting-head indices found in the checkpoint's keys
    for key in state_dict:
        if key.startswith("boosting_heads."):
            indices.add(int(key.split(".")[1]))                    # extract the head index from a key like "boosting_heads.<i>.0.weight"
    return len(indices)                                             # total number of distinct boosting heads present in the checkpoint
