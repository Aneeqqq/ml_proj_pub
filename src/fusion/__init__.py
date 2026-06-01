"""Late (decision-level) fusion of per-modality blockage probabilities.

Paper §III-F: P_fused = sum_i w_i * P_i, with w = softmax over per-modality validation F1.
See ML_Proj_Vault/modalities/fusion.md.
"""
from .late_fusion import softmax_f1_weights, fuse_probs, predict_probs

__all__ = ["softmax_f1_weights", "fuse_probs", "predict_probs"]
