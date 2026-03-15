import torch
from torch import nn
import torch.nn.functional as F


class SpatialTemporalReIDHead(nn.Module):
    """ReID head inspired by visual + spatial-temporal cue fusion.

    Inputs:
        obj_ptr: [B, C] visual pointer feature from decoder
        view_ids: [B] camera/view ids
        frame_indices: [B] frame indices in clip/video
        num_frames: scalar frame count for normalization
    """

    def __init__(self, in_dim: int, emb_dim: int = 256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )
        self.st_proj = nn.Sequential(
            nn.Linear(2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, obj_ptr, view_ids, frame_indices, num_frames: int):
        visual_feat = self.visual_proj(obj_ptr)

        denom = max(num_frames - 1, 1)
        norm_frame = frame_indices.float() / float(denom)
        norm_view = view_ids.float() / 2.0
        st_feat = torch.stack([norm_view, norm_frame], dim=1)
        st_feat = self.st_proj(st_feat)

        fused = self.fuse(torch.cat([visual_feat, st_feat], dim=1))
        fused = F.normalize(fused, p=2, dim=1)
        return fused
