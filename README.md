# TML Assignment 2 — Stolen Model Detection

**Course**: Trustworthy Machine Learning, Saarland University (Summer 2026)  
**Task**: Assign a continuous "stealing confidence" score to 360 suspect ResNet-18 models  
**Metric**: TPR @ 5% FPR (true positive rate at 5% false positive rate)  
**Final result**: **1st place — score 0.759259**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [Detection Approach](#detection-approach)
   - [Signal 1 — Weight Cosine Similarity](#signal-1--weight-cosine-similarity)
   - [Signal 2 — Soft Logit Agreement](#signal-2--soft-logit-agreement)
   - [Signal 3 — FGSM Adversarial Transfer](#signal-3--fgsm-adversarial-transfer)
   - [Fusion: Why max?](#fusion-why-max)
4. [Results Analysis](#results-analysis)
   - [Score Distribution](#score-distribution)
   - [Signal Breakdown by Theft Type](#signal-breakdown-by-theft-type)
   - [Run History](#run-history)
   - [Key Insight: Run 5 → Run 6](#key-insight-run-5--run-6)
5. [Files](#files)
6. [How to Reproduce](#how-to-reproduce)
7. [Dependencies](#dependencies)
8. [References](#references)

---

## Problem Statement

Given a **target ResNet-18** trained on CIFAR-100, determine which of **360 suspect models** were stolen from it. Models are evaluated by ranking a continuous confidence score using **TPR@5%FPR** — what fraction of actual stolen models are caught while keeping false positives to just 5%.

Theft strategies in the wild include:

| Strategy | Description |
|---|---|
| **Direct copy** | Weights cloned directly from the target |
| **Fine-tuning** | Weights stolen then slightly adjusted |
| **Label-permuted** | Same decision boundaries, output classes relabeled |
| **Knowledge distillation** | Trained from scratch to mimic target outputs |

A good detector must handle all of these without being tuned to a specific subset.

The leaderboard evaluates on 30% of ground truth (public) + 70% (private). Overfitting to the public split can hurt the final score.

---

## Architecture

All models — target and all 360 suspects — use the same modified ResNet-18 for CIFAR-100:

```python
model = resnet18(weights=None)
model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()   # remove downsampling — CIFAR images are 32×32
model.fc     = nn.Linear(512, 100)
```

Input normalization:
- Mean: `(0.5071, 0.4867, 0.4408)`
- Std:  `(0.2675, 0.2565, 0.2761)`

Total trainable parameters: **11,220,132**

---

## Detection Approach

The final score is computed as:

```
score = max(w_norm, soft_agree, adv_transfer)
```

Each signal is a standalone detector for a different theft strategy. Taking `max` ensures every stolen model is caught by whichever signal is most sensitive for its specific theft type.

---

### Signal 1 — Weight Cosine Similarity

**Paper**: standard model fingerprinting / weight-space proximity

Flatten all parameters of both models into single vectors and compute cosine similarity:

```python
def flat_weights(model):
    return torch.cat([p.data.cpu().float().flatten() for p in model.parameters()])

w_cos  = F.cosine_similarity(flat_weights(target).unsqueeze(0),
                              flat_weights(suspect).unsqueeze(0)).item()
w_norm = (w_cos + 1.0) / 2.0   # map cosine [-1, 1] → [0, 1]
```

**Catches**: direct copies and minimally fine-tuned models.  
**Fails on**: label-permuted models (same structure, different output mapping), knowledge-distilled models (functionally similar but independently trained weights).

**Observed range in data**: `w_cos ∈ [-0.003, 1.001]`, `w_norm ∈ [0.499, 1.001]`

---

### Signal 2 — Soft Logit Agreement

**Paper**: Orekondy et al., *Knockoff Nets: Stealing Functionality of Black-Box Models*, CVPR 2019

Run both models on the full CIFAR-100 test set and compute the average cosine similarity of their raw 100-dimensional logit vectors:

```python
def soft_agree(target, suspect, loader, device):
    s = 0.0; total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            s     += F.cosine_similarity(target(x), suspect(x), dim=1).sum().item()
            total += x.size(0)
    return s / total
```

This measures whether the models produce the **same output distribution** on every input, not just the same top-1 label. Fine-tuned stolen models retain this similarity even after weight divergence.

**Catches**: fine-tuned stolen models, knowledge-distilled models with matched output distributions.  
**Fails on**: label-permuted models (output vectors are permuted → low cosine similarity even though decision boundaries are identical).

**Observed range in data**: `soft_agree ∈ [0.065, 1.000]`

---

### Signal 3 — FGSM Adversarial Transfer

**Paper**: Jagielski et al., *High Accuracy and High Fidelity Extraction of Neural Networks*, USENIX Security 2020 / ModelDiff framing

Pre-compute FGSM adversarial examples from the target **once** and measure what fraction of them also fool each suspect:

```python
# Step 1 — build adversarial set from target (run once, ~2000 images)
x_adv = x + eps * sign(∇_x CrossEntropy(target(x), target(x).argmax()))
# Keep only examples where target prediction actually changed

# Step 2 — score each suspect
adv_transfer = fraction where suspect(x_adv).argmax() ≠ original_target_prediction
```

Two models that share **decision boundaries** will be fooled by the same adversarial examples. This signal is invariant to output relabeling — a permuted model has the same internal boundaries even though its class indices differ.

**Catches**: all stolen variants including label-permuted (key insight), fine-tuned, and direct copies.  
**Fails on**: nothing — this is the most general signal. Even random unrelated models share some adversarial vulnerability (~60–70%), but stolen models sit at 85%+.

**Observed range in data**: `adv_transfer ∈ [0.613, 1.000]`

**FGSM hyperparameters used**: `eps = 0.1` (in normalized space), `2000` images for building the adversarial set.

---

### Fusion: Why max?

Each signal captures a different theft strategy:

| Theft type | w_norm | soft_agree | adv_transfer | Dominant signal |
|---|---|---|---|---|
| Direct weight copy | **~1.00** | ~0.95 | ~0.95 | `w_norm` |
| Fine-tuned stolen | ~0.60 | **~0.87** | ~0.88 | `soft_agree` or `adv_transfer` |
| Label-permuted | ~0.60 | ~0.10 | **~0.96** | `adv_transfer` |
| Knowledge distillation | ~0.60 | **~1.00** | ~1.00 | `soft_agree` |
| Unrelated (not stolen) | ~0.60 | ~0.43 | ~0.70 | — (all low) |

A **weighted average** would average out the strong `adv_transfer` signal for label-permuted models (0.96 averaged with soft=0.10 → ~0.53, too low). Taking `max` lets each signal contribute at its maximum strength independently.

Concretely, in Run 4 we used `max(w_norm, 0.5*soft + 0.5*adv)`. For id=1 (label-permuted: soft=0.145, adv=0.965):
- Averaged: `0.5*0.145 + 0.5*0.965 = 0.555` → underscore
- Max: `max(0.597, 0.145, 0.965) = 0.965` → correctly flagged

---

## Results Analysis

### Score Distribution

Score distribution across all 360 suspect models (Run 6, final submission):

```
Score range    Count   Bar
─────────────────────────────────────────────────────
< 0.65          4    ████
[0.65, 0.70)    6    ██████
[0.70, 0.75)   10    ██████████
[0.75, 0.80)    8    ████████
[0.80, 0.85)   22    ██████████████████████
[0.85, 0.90)  176    ████████████████████████████████████████████████████████████████████████████████████ (bulk of suspects)
[0.90, 0.96)   67    ███████████████████████████████████████████████████████
[0.96, 0.99)   20    ████████████████████
[0.99, 1.0+)   47    ███████████████████████████████████████████████
```

The large cluster at `[0.85, 0.90)` represents suspects with moderate behavioral similarity — likely fine-tuned or distilled variants. The `[0.99, 1.0+)` group are near-exact or exact copies.

**Summary statistics (score column):**

| Statistic | Value |
|---|---|
| Min | 0.626 |
| 25th percentile | 0.869 |
| Median | 0.889 |
| Mean | 0.889 |
| 75th percentile | 0.924 |
| Max | 1.0006 |

---

### Signal Breakdown by Theft Type

Selected models illustrating each category. The **bold** value is the dominant signal that determines the final score:

#### Category A — Direct Weight Copy (w_norm ≈ 1.0)

These models were copied with minimal or no modification:

| id | weight_cos | w_norm | soft_agree | adv_transfer | score |
|---|---|---|---|---|---|
| 5   | 0.942 | **0.971** | 0.868 | 0.951 | 0.971 |
| 9   | 0.811 | **0.906** | 0.854 | 0.932 | 0.932 |
| 34  | 0.991 | **0.996** | 0.941 | 0.970 | 0.996 |
| 203 | 0.978 | **0.989** | 0.716 | 0.986 | 0.989 |
| 262 | 0.766 | **0.883** | 0.770 | 0.939 | 0.939 |

Weight cosine > 0.75 is a clear indicator of weight-level theft.

#### Category B — Functional Clone / Knowledge Distillation (soft ≈ 1.0, adv ≈ 1.0, low w_cos)

Trained from scratch but perfectly mimicking the target's outputs:

| id | weight_cos | w_norm | soft_agree | adv_transfer | score |
|---|---|---|---|---|---|
| 4   | 0.264 | 0.632 | **1.000** | **1.000** | 1.000 |
| 40  | 0.120 | 0.560 | **1.000** | **1.000** | 1.000 |
| 46  | 0.263 | 0.632 | **1.000** | **1.000** | 1.000 |
| 64  | 0.135 | 0.567 | **1.000** | **1.000** | 1.000 |
| 72  | 0.136 | 0.568 | **1.000** | **1.000** | 1.000 |
| 81  | 0.257 | 0.628 | **1.000** | **1.000** | 1.000 |
| 109 | 0.256 | 0.628 | **1.000** | **1.000** | 1.000 |
| 234 | 0.258 | 0.629 | **1.000** | **1.000** | 1.000 |

These would be missed by weight-cosine alone (`w_norm ≈ 0.56–0.63`), but are perfectly caught by both behavioral signals.

#### Category C — Label-Permuted Stolen (high adv, very low soft)

Same internal structure as the target but output class indices are relabeled. The key category that drove the design of this approach:

| id | weight_cos | w_norm | soft_agree | adv_transfer | score |
|---|---|---|---|---|---|
| 1   | 0.193 | 0.597 | 0.145 | **0.965** | 0.965 |
| 10  | 0.192 | 0.596 | 0.179 | **0.926** | 0.926 |
| 69  | 0.196 | 0.598 | 0.251 | **0.921** | 0.921 |
| 103 | 0.193 | 0.596 | 0.211 | **0.917** | 0.917 |
| 120 | 0.193 | 0.597 | 0.140 | **0.948** | 0.948 |
| 228 | 0.195 | 0.597 | **0.065** | **0.960** | 0.960 |
| 280 | 0.195 | 0.598 | 0.097 | **0.955** | 0.955 |
| 330 | 0.195 | 0.598 | 0.081 | **0.964** | 0.964 |

`soft_agree` is near-zero because the output vectors point in completely different directions after permutation. Without `adv_transfer` as a standalone signal, all of these would have been severely underscored.

#### Category D — Fine-Tuned Stolen (soft > adv, neither dominant alone)

These models have moderately high similarity across all signals. `soft_agree` is the dominant signal for this group — the insight that moved Run 5 → Run 6:

| id | weight_cos | w_norm | soft_agree | adv_transfer | score |
|---|---|---|---|---|---|
| 19  | 0.202 | 0.601 | **0.909** | 0.893 | 0.909 |
| 79  | 0.201 | 0.600 | **0.875** | 0.850 | 0.875 |
| 247 | 0.202 | 0.601 | **0.873** | 0.856 | 0.873 |
| 335 | 0.200 | 0.600 | **0.901** | 0.869 | 0.901 |
| 349 | 0.200 | 0.600 | **0.838** | 0.814 | 0.838 |

For these, `adv_transfer < soft_agree`. In Run 5 where `behavioral = 0.5*soft + 0.5*adv`, the averaging diluted the soft signal. Making `soft` a standalone signal in Run 6 raised these scores.

---

### Run History

| Run | Scoring Formula | TPR@5%FPR | Leaderboard Position | Key change |
|---|---|---|---|---|
| 1 | `max(w_norm, soft)` | 0.592593 | 21st | Tutorial baseline |
| 2 | `max(w_norm, soft, error_agree)` | Worse | — | Added agreement on target errors — backfired: punishes models that corrected errors |
| 3 | `max(w_norm, soft, CKA)` | Same as Run 1 | — | CKA dominated by soft via max — no new information |
| 4 | `max(w_norm, 0.5*soft + 0.5*adv)` | 0.648148 | 3rd | Added adversarial transfer signal |
| 5 | `max(w_norm, adv, 0.5*soft + 0.5*adv)` | 0.740741 | 2nd | `adv` as standalone catches label-permuted models |
| **6** | **`max(w_norm, soft, adv)`** | **0.759259** | **1st** | **`soft` also as standalone catches fine-tuned where soft > adv** |

---

### Key Insight: Run 5 → Run 6

**Run 5** formula: `score = max(w_norm, adv, 0.5*soft + 0.5*adv)`

The `behavioral = 0.5*soft + 0.5*adv` average still diluted the `soft` signal for models where `soft > adv`. For models like id=335 (`soft=0.901, adv=0.869`):

```
Run 5 score = max(0.600, 0.869, 0.5*0.901 + 0.5*0.869)
            = max(0.600, 0.869, 0.885)
            = 0.885
```

**Run 6** formula: `score = max(w_norm, soft, adv)`

```
Run 6 score = max(0.600, 0.901, 0.869)
            = 0.901    ← higher, correctly captures fine-tuned signal
```

**10 models were boosted** by switching to Run 6:

| id | soft_agree | adv_transfer | Run 5 score | Run 6 score | Δ |
|---|---|---|---|---|---|
| 19  | 0.909 | 0.893 | 0.901 | 0.909 | +0.008 |
| 33  | 0.731 | 0.685 | 0.708 | 0.731 | +0.023 |
| 79  | 0.875 | 0.850 | 0.863 | 0.875 | +0.012 |
| 179 | 0.803 | 0.720 | 0.762 | 0.803 | +0.041 |
| 241 | 0.861 | 0.788 | 0.825 | 0.861 | +0.036 |
| 247 | 0.873 | 0.856 | 0.865 | 0.873 | +0.008 |
| 264 | 0.783 | 0.741 | 0.762 | 0.783 | +0.021 |
| 291 | 0.832 | 0.783 | 0.808 | 0.832 | +0.024 |
| 335 | 0.901 | 0.869 | 0.885 | 0.901 | +0.016 |
| 349 | 0.838 | 0.814 | 0.826 | 0.838 | +0.012 |

---

## Files

| File | Description |
|---|---|
| `stolen_model_detection - 1st.ipynb` | Main Google Colab notebook — **the winning solution, run this** |
| `task_template.py` | Local CPU-compatible Python script (uses smaller subsets for speed) |
| `submission.py` | Submits `submission.csv` to the leaderboard API |
| `submission (4).csv` | Winning submission scores (360 rows, `id` + `score`) |
| `detailed_results (4).csv` | Per-model signal breakdown: `id, score, weight_cos, w_norm, soft_agree, adv_transfer` |

---

## How to Reproduce

### Colab (recommended — GPU T4 or better)

1. Open `stolen_model_detection - 1st.ipynb` in Google Colab with a GPU runtime.
2. **Cell 1**: installs `huggingface_hub safetensors pandas`.
3. **Cell 2**: runs the full pipeline:
   - Downloads target model + all 360 suspects from `SprintML/tml26_task2` on HuggingFace
   - Downloads CIFAR-100 test set
   - Builds FGSM adversarial set (2 000 images, eps=0.1)
   - Scores all 360 suspects — prints per-model signal values
   - Saves `submission.csv` and `detailed_results.csv` to `/content/`
4. **Cell 3**: submits to the leaderboard (replace `API_KEY` with your own).
5. **Cell 4**: downloads result files to local disk.

Expected runtime: **20–30 minutes** on a T4 GPU.

### Local CPU

Install dependencies:
```bash
pip install torch torchvision huggingface_hub safetensors pandas
```

Edit `task_template.py` — set `BASE_DIR` to your local path — then run:
```bash
python task_template.py
```

CPU mode uses smaller subsets (`EVAL_IMAGES=1000`, `ADV_IMAGES=500`) for speed. For full accuracy matching the Colab results, use the notebook with GPU (`EVAL_IMAGES=10000`, `ADV_IMAGES=2000`).

---

## Dependencies

```
torch >= 2.0
torchvision >= 0.15
huggingface_hub >= 0.20
safetensors >= 0.4
pandas >= 1.5
```

All standard PyTorch and HuggingFace packages — no custom installs required.

---

## References

1. Orekondy, T., Schiele, B., Fritz, M. (2019). **Knockoff Nets: Stealing Functionality of Black-Box Models.** *CVPR 2019.*  
   → Motivates soft logit agreement as a behavioral similarity metric.

2. Jagielski, M., Carlini, N., Berthelot, D., Kurakin, A., Papernot, N. (2020). **High Accuracy and High Fidelity Extraction of Neural Networks.** *USENIX Security 2020.*  
   → Motivates adversarial transferability as evidence of shared decision boundaries.

3. Guo, W. et al. (2021). **ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection.** *ISSTA 2021.*  
   → Frames adversarial transfer as a test for model reuse / theft.
