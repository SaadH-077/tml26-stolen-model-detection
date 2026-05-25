# TML Assignment 2 — Stolen Model Detection

How to recreate the leaderboard submission for Team V

## Files

- `stolen_model_detection - 1st.ipynb` — main notebook. Running it end-to-end produces `submission.csv`.
- `submission.py` — uploads `submission.csv` to the leaderboard endpoint.
- `submission (4).csv` — the leaderboard-winning submission.
- `detailed_results (4).csv` — per-suspect signal breakdown written alongside the submission. Not required for grading.
- `task_template.py` — the assignment-provided starter, kept for reference. Uses smaller evaluation subsets and does **not** reproduce the leaderboard score; use the notebook for the final result.

## Recreate the result

1. **Open `stolen_model_detection - 1st.ipynb` in Google Colab** with a GPU runtime (T4 or better).

2. **Run the cells top to bottom:**
   - **Cell 1** installs `huggingface_hub`, `safetensors`, `pandas`.
   - **Cell 2** downloads the target model and all 360 suspects from `SprintML/tml26_task2` on Hugging Face, downloads the CIFAR-100 test set, builds the FGSM adversarial set (2 000 images, `eps = 0.1`), scores every suspect, and writes `/content/submission.csv` and `/content/detailed_results.csv`.
   - **Cell 3** submits `submission.csv` to the leaderboard. Replace `API_KEY = 'YOUR_API_KEY_HERE'` with your own key before running.
   - **Cell 4** downloads both CSVs to your local machine.

**Expected runtime:** 20–30 minutes on a T4 GPU.

## Dependencies

```
torch         >= 2.0
torchvision   >= 0.15
huggingface_hub >= 0.20
safetensors   >= 0.4
pandas        >= 1.5
```

Colab has `torch`, `torchvision`, and `pandas` preinstalled; Cell 1 installs `huggingface_hub` and `safetensors`.
