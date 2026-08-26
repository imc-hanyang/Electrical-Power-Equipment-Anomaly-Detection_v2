import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import *
from utils import *
import timm
import open_clip

def build_FR(config):
    CLAdapter_Module = CLAdapter(check_point = config.MODEL.CLAdapter.checkpoint, 
                    width = config.MODEL.backbone.out_dim, 
                    len_token = config.MODEL.backbone.num_patch, 
                    centers = config.MODEL.CLAdapter.centers, 
                    dt_layers = config.MODEL.CLAdapter.layers,
                    mlp_ratio = config.MODEL.CLAdapter.mlp_ratio)
    return CLAdapter_Module

class Post_vit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.MODEL.backbone.out_dim)
        self.norm2 = nn.LayerNorm(config.MODEL.backbone.out_dim)
        self.CLAdapter_layers = build_FR(config)

    def forward(self, x):
        x = self.norm1(x)
        x = self.CLAdapter_layers(x)
        x = self.norm2(x)
        return x

class CLAdapter_CLIP_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.m_mode = config.MODEL.m_mode
        self.f_mode = config.MODEL.f_mode
        self.pooling_mode = getattr(config.MODEL, "pooling_mode", "mean")
        self.topk_ratio = float(getattr(config.MODEL, "topk_ratio", 0.10))
        if self.pooling_mode not in {"mean", "topk", "attention", "gated_mil", "rank_mil"}:
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio must be in (0, 1], got {self.topk_ratio}")
        pretrained = getattr(config.MODEL.backbone, "pretrained", True)
        if ' ' not in config.MODEL.backbone.model_name:
            self.backbone = timm.create_model(config.MODEL.backbone.model_name, pretrained=pretrained)
        else:
            names = config.MODEL.backbone.model_name.split()
            self.backbone = open_clip.create_model(names[0], names[1]).visual.trunk

        if self.f_mode != 'full' and config.MODEL.finetune is None:
            for param in self.backbone.parameters():
                param.requires_grad=False
        if self.f_mode == 'cla':
            self.post = Post_vit(config)
        self.head = nn.Linear(config.MODEL.backbone.out_dim, config.MODEL.num_classes)
        if self.pooling_mode == "attention":
            hidden_dim = int(getattr(config.MODEL, "attention_hidden_dim", 192))
            self.attention_pool = nn.Sequential(
                nn.LayerNorm(config.MODEL.backbone.out_dim),
                nn.Linear(config.MODEL.backbone.out_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.attention_pool[-1].weight)
            nn.init.zeros_(self.attention_pool[-1].bias)
        if self.pooling_mode == "gated_mil":
            dim = config.MODEL.backbone.out_dim
            hidden_dim = int(getattr(config.MODEL, "attention_hidden_dim", 192))
            self.mil_roi_top = float(getattr(config.MODEL, "mil_roi_top", 0.28))
            self.mil_roi_bottom = float(getattr(config.MODEL, "mil_roi_bottom", 0.72))
            gate_init = float(getattr(config.MODEL, "mil_gate_init", 0.10))
            if not 0.0 <= self.mil_roi_top < self.mil_roi_bottom <= 1.0:
                raise ValueError("MIL ROI must satisfy 0 <= top < bottom <= 1")
            if not 0.0 < gate_init < 1.0:
                raise ValueError("mil_gate_init must be in (0, 1)")
            self.mil_norm = nn.LayerNorm(dim)
            self.mil_attention_v = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
            self.mil_attention_u = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
            self.mil_attention_w = nn.Linear(hidden_dim, 1)
            self.mil_head = nn.Linear(dim, config.MODEL.num_classes)
            self.mil_gate_logit = nn.Parameter(
                torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32)
            )
        if self.pooling_mode == "rank_mil":
            dim = config.MODEL.backbone.out_dim
            self.rank_mil_roi_top = float(getattr(config.MODEL, "mil_roi_top", 0.28))
            self.rank_mil_roi_bottom = float(getattr(config.MODEL, "mil_roi_bottom", 0.72))
            self.rank_mil_topk_ratio = float(getattr(config.MODEL, "topk_ratio", 0.05))
            if not 0.0 <= self.rank_mil_roi_top < self.rank_mil_roi_bottom <= 1.0:
                raise ValueError("rank MIL ROI must satisfy 0 <= top < bottom <= 1")
            if not 0.0 < self.rank_mil_topk_ratio <= 1.0:
                raise ValueError("rank MIL top-k ratio must be in (0, 1]")
            self.rank_mil_norm = nn.LayerNorm(dim)
            self.rank_mil_head = nn.Linear(dim, config.MODEL.num_classes)

        if config.MODEL.backbone.checkpoint:
            self.backbone.set_grad_checkpointing()

    def gated_mil_pool(self, tokens, return_attention=False):
        """Fuse global mean evidence with learned central-ROI MIL evidence."""
        batch, num_tokens, _ = tokens.shape
        grid = math.isqrt(num_tokens)
        if grid * grid != num_tokens:
            raise ValueError(f"gated_mil requires a square patch grid, got {num_tokens} tokens")

        normalized = self.mil_norm(tokens)
        attention_logits = self.mil_attention_w(
            self.mil_attention_v(normalized) * self.mil_attention_u(normalized)
        ).squeeze(-1)
        row_index = torch.arange(num_tokens, device=tokens.device) // grid
        first_row = max(0, min(grid - 1, int(math.floor(self.mil_roi_top * grid))))
        last_row = max(first_row + 1, min(grid, int(math.ceil(self.mil_roi_bottom * grid))))
        roi_mask = (row_index >= first_row) & (row_index < last_row)
        attention_logits = attention_logits.masked_fill(~roi_mask.unsqueeze(0), float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)

        global_logits = self.head(tokens.mean(1))
        local_embedding = torch.sum(tokens * attention_weights.unsqueeze(-1), dim=1)
        local_logits = self.mil_head(local_embedding)
        gate = torch.sigmoid(self.mil_gate_logit)
        logits = global_logits + gate * local_logits
        if return_attention:
            return logits, {
                "attention": attention_weights,
                "global_logits": global_logits,
                "local_logits": local_logits,
                "local_gate": gate.expand(batch),
                "roi_rows": (first_row, last_row),
            }
        return logits

    def rank_mil_pool(self, tokens, return_aux=False):
        """Pool hard local instances while retaining the global classifier.

        The local branch is restricted to the central wire ROI.  Its hardest
        patches form an image-level positive/negative bag.  Training adds an
        explicit ranking objective between anomalous and normal bags.
        """
        batch, num_tokens, _ = tokens.shape
        grid = math.isqrt(num_tokens)
        if grid * grid != num_tokens:
            raise ValueError(f"rank_mil requires a square patch grid, got {num_tokens} tokens")

        global_logits = self.head(tokens.mean(1))
        patch_logits = self.rank_mil_head(self.rank_mil_norm(tokens))
        patch_evidence = patch_logits[..., 1] - patch_logits[..., 0]

        row_index = torch.arange(num_tokens, device=tokens.device) // grid
        first_row = max(0, min(grid - 1, int(math.floor(self.rank_mil_roi_top * grid))))
        last_row = max(first_row + 1, min(grid, int(math.ceil(self.rank_mil_roi_bottom * grid))))
        roi_mask = (row_index >= first_row) & (row_index < last_row)
        roi_logits = patch_logits[:, roi_mask, :]
        roi_evidence = patch_evidence[:, roi_mask]
        k = max(1, int(math.ceil(roi_evidence.shape[1] * self.rank_mil_topk_ratio)))
        topk_indices = roi_evidence.topk(k, dim=1, largest=True, sorted=False).indices
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, roi_logits.shape[-1])
        bag_logits = roi_logits.gather(1, gather_indices).mean(1)
        bag_evidence = roi_evidence.gather(1, topk_indices).mean(1)

        # Fixed equal fusion: no validation search over a mixing coefficient.
        logits = 0.5 * global_logits + 0.5 * bag_logits
        if return_aux:
            return logits, {
                "global_logits": global_logits,
                "bag_logits": bag_logits,
                "bag_evidence": bag_evidence,
                "patch_logits": patch_logits,
                "patch_evidence": patch_evidence,
                "roi_patch_evidence": roi_evidence,
                "roi_mask": roi_mask,
                "roi_rows": (first_row, last_row),
                "topk_patch_count": k,
            }
        return logits

    def pool_patch_tokens(self, tokens, return_attention=False):
        if self.pooling_mode == "mean":
            return self.head(tokens.mean(1))
        if self.pooling_mode == "topk":
            patch_logits = self.head(tokens)
            anomaly_evidence = patch_logits[..., 1] - patch_logits[..., 0]
            k = max(1, int(math.ceil(tokens.shape[1] * self.topk_ratio)))
            topk_indices = anomaly_evidence.topk(k, dim=1, largest=True, sorted=False).indices
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, patch_logits.shape[-1])
            return patch_logits.gather(1, gather_indices).mean(1)
        if self.pooling_mode == "gated_mil":
            return self.gated_mil_pool(tokens, return_attention=return_attention)
        if self.pooling_mode == "rank_mil":
            return self.rank_mil_pool(tokens, return_aux=return_attention)
        attention_logits = self.attention_pool(tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        return self.head((tokens * attention_weights).sum(1))

    def forward(self, x, test=False, return_attention=False):
        x = self.backbone.forward_features(x)
        if self.m_mode == 'conv' or self.m_mode == 'res_xcep':
            x = x.flatten(2).permute(0, 2, 1)
            if self.f_mode == 'cla':
                x = self.post(x)
            x = x.mean(1)
            return self.head(x)
        else:
            if self.f_mode == 'cla':
                x = self.post(x[:, 1:, :])
                return self.pool_patch_tokens(x, return_attention=return_attention)
            else:
                x = x[:, 0, :]
                return self.head(x)
