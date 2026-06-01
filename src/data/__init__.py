"""Sequence-correct data pipeline for DeepSense Scenario 31 blockage prediction.

Modules:
  splits         -- stratified, sequence-level train/val/test assignment (no leakage)
  radar_features -- complex radar (4,256,250) -> 8-channel (8,256,64) feature tensor
  dataset        -- BlockageWindowDataset (camera+radar windows, K-step horizon labels)
"""
