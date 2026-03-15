import torch
from torch import nn
import torch.nn.functional as F


class MultiViewReID(nn.Module):
    """Independent multi-view ReID branch for clustering-oriented feature learning."""

    def __init__(self, in_channels=3, hidden_dim=128, emb_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def encode_view_batch(self, view_samples):
        """view_samples: NestedTensor with tensors [B, T, C, H, W]"""
        x = view_samples.tensors
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.encoder(x).flatten(1)
        feat = self.proj(feat)
        feat = feat.reshape(b, t, -1).mean(dim=1)
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def forward(self, multi_view_samples):
        """multi_view_samples: list[NestedTensor], len=3"""
        per_view = [self.encode_view_batch(vs) for vs in multi_view_samples]
        # stack as [B, V, D]
        return torch.stack(per_view, dim=1)


def reid_cluster_loss(mv_embeddings, labels, margin=0.3):
    """Clustering-oriented loss over multi-view embeddings.

    mv_embeddings: [B, V, D], labels: [B]
    """
    if mv_embeddings is None or labels is None or mv_embeddings.size(0) < 2:
        return mv_embeddings.new_tensor(0.0) if isinstance(mv_embeddings, torch.Tensor) else torch.tensor(0.0)

    emb = mv_embeddings.mean(dim=1)
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()

    label_eq = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    diag = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    pos_mask = label_eq & (~diag)
    neg_mask = (~label_eq) & (~diag)

    losses = []
    if pos_mask.any():
        losses.append((1.0 - sim[pos_mask]).mean())
    if neg_mask.any():
        losses.append(F.relu(sim[neg_mask] - margin).mean())
    if len(losses) == 0:
        return sim.new_tensor(0.0)
    return sum(losses)


def build_reid_labels_from_targets(view_targets):
    """Build per-sample ReID labels from one-view target list (batch dimension)."""
    labels = []
    for i, t in enumerate(view_targets):
        labels.append(int(t.get("exp_id", i)))
    return torch.tensor(labels, dtype=torch.long)
