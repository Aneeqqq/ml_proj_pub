---
title: Open Questions (hub)
tags: [open-questions, hub]
updated: 2026-06-01
status: verified
---

# Open Questions — hub

Unresolved decisions. Resolve top-down; log each resolution in [[log]] and update the source page.

## 🔴 Must resolve before training
1. ✅ **RESOLVED — blockage label.** `scenario31_dev_labelled.csv`, `label`∈{blocked,not_blocked},
   200 pos (2.85%). → [[blockage-label]]
1b. **Stratified split for scarce positives.** Only 15/52 seqs (23 episodes) have blockage —
   distribute them across train/val/test (keep seqs whole); verify positive windows per horizon. → [[sequences-and-batching]]
2. **Sub-sampling step `s`?** (native ~100 ms → paper 300 ms ⇒ likely s=3.) Confirm + apply within
   sequences. → [[problem-formulation]]
3. **Window/horizon convention?** Adopt W=5 input, K=5 horizon, t+5 = 1.5 s. Confirm window=1.2 s
   reading. → [[problem-formulation]]
4. **Split assignment?** Which `seq_index` → train/val/test (sequence-level, **stratified** on
   blockage presence per 1b). → [[sequences-and-batching]]

## 🟠 Affect fidelity
5. **Camera FC head:** hidden width? output 1 prob or 5 horizon logits? (we assume 5.) → [[camera]]
6. **Image normalization stats:** ImageNet assumed — confirm. → [[camera]]
7. **Radar exact 8 features & conv channels/kernels; Doppler trim-vs-pad rule.** → [[radar]]
8. **Fusion: F1 scaled before softmax?** (weights are near-uniform otherwise.) Per-horizon or global
   weights? → [[fusion]]
9. **Optimizer/LR/epochs/batch size** — not stated anywhere in the paper. Choose sane defaults and
   record. (Adam, cosine/step LR, early-stop on val F1.)

## 🟡 Nice to confirm
10. **α=1.1 sensitivity** for `w_pos`. → [[class-imbalance]]
11. **Augmentation magnitudes** (rotation angle, blur σ, radar noise σ). → [[camera]] [[radar]]
12. **Do Scenarios 32–34 share schema with 31?** (assume yes; verify on download.) → [[replication-plan]]

Related: [[abnormalities]] · [[replication-plan]] · [[lessons-learned]]
