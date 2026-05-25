import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID     = "SprintML/tml26_task2"
API_KEY     = "YOUR_API_KEY_HERE"
BASE_DIR    = r"/path/to/your/project"
HF_CACHE    = os.path.join(BASE_DIR, "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE

EVAL_IMAGES = 1000    # for soft_agree (CPU: keep small)
ADV_IMAGES  = 500     # for adversarial transfer
ADV_EPS     = 0.1     # FGSM step size in normalized space

# ── Architecture ──────────────────────────────────────────────────────────────
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def load_model(path, device):
    m = make_model()
    m.load_state_dict(load_file(path, device="cpu"))
    return m.to(device).eval()

# ── Signal 1: Weight cosine ───────────────────────────────────────────────────
def flat_weights(model):
    return torch.cat([p.data.cpu().float().flatten() for p in model.parameters()])

def weight_cosine(w1, w2):
    return F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()

# ── Signal 2: Soft logit agreement (Paper 2 — Knockoff Nets) ─────────────────
def soft_agree(target, suspect, loader, device):
    s = 0.0; total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            s     += F.cosine_similarity(target(x), suspect(x), dim=1).sum().item()
            total += x.size(0)
    return s / total

# ── Signal 3: FGSM adversarial transfer (Paper 3 — ModelDiff) ────────────────
def build_adversarial_set(target, loader, device, eps):
    """
    Pre-compute FGSM adversarial examples for the target.
    Keep only examples where the attack succeeded (target prediction changed).
    Computed ONCE, reused for all 360 suspects.
    """
    x_adv_list = []
    orig_pred_list = []
    target.eval()

    for x, _ in loader:
        x = x.to(device)
        x_in = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            pred = target(x_in).argmax(1)
            loss = F.cross_entropy(target(x_in), pred)
            target.zero_grad()
            loss.backward()
            x_adv = (x_in + eps * x_in.grad.sign()).detach()

        with torch.no_grad():
            adv_pred = target(x_adv).argmax(1)
            fooled   = (adv_pred != pred)
            if fooled.sum() > 0:
                x_adv_list.append(x_adv[fooled].cpu())
                orig_pred_list.append(pred[fooled].cpu())

    return torch.cat(x_adv_list), torch.cat(orig_pred_list)

def adv_transfer_score(suspect, x_adv_all, orig_preds, device, batch_size=64):
    """
    Fraction of target adversarial examples that also fool the suspect.
    High → shared decision boundaries → likely stolen.
    """
    sus_preds = []
    with torch.no_grad():
        for i in range(0, len(x_adv_all), batch_size):
            batch = x_adv_all[i:i+batch_size].to(device)
            sus_preds.append(suspect(batch).argmax(1).cpu())
    sus_preds = torch.cat(sus_preds)
    return (sus_preds != orig_preds).float().mean().item()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_path = hf_hub_download(REPO_ID, "target_model/weights.safetensors", cache_dir=HF_CACHE)
    target   = load_model(target_path, device)
    target_w = flat_weights(target)
    print(f"Target loaded. Weight vector: {target_w.shape[0]:,}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    full_test = datasets.CIFAR100(
        root=os.path.join(BASE_DIR, "cifar100"), train=False, download=True, transform=transform
    )
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_test, list(range(EVAL_IMAGES))),
        batch_size=64, shuffle=False, num_workers=0
    )
    adv_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_test, list(range(ADV_IMAGES))),
        batch_size=64, shuffle=False, num_workers=0
    )

    print(f"Building adversarial set (eps={ADV_EPS}, {ADV_IMAGES} images)...")
    x_adv_all, orig_preds = build_adversarial_set(target, adv_loader, device, ADV_EPS)
    print(f"Adversarial examples: {len(x_adv_all)} / {ADV_IMAGES} "
          f"(success rate: {len(x_adv_all)/ADV_IMAGES:.3f})")

    rows = []
    for i in range(360):
        print(f"\n[{i:03d}/359] suspect_{i:03d}...")
        path = hf_hub_download(
            REPO_ID, f"suspect_models/suspect_{i:03d}.safetensors", cache_dir=HF_CACHE
        )
        suspect = load_model(path, device)
        sus_w   = flat_weights(suspect)

        w_cos = weight_cosine(target_w, sus_w)
        soft  = soft_agree(target, suspect, eval_loader, device)
        adv   = adv_transfer_score(suspect, x_adv_all, orig_preds, device)

        w_norm = (w_cos + 1.0) / 2.0   # map cosine [-1,1] → [0,1]
        score  = max(w_norm, soft, adv) # each signal independently detects a theft strategy

        rows.append({"id": i, "score": score,
                     "weight_cos": w_cos, "w_norm": w_norm,
                     "soft_agree": soft, "adv_transfer": adv})
        print(f"  w={w_cos:.4f}  soft={soft:.4f}  adv={adv:.4f}  ->  score={score:.4f}")

        del suspect, sus_w

    df = pd.DataFrame(rows)
    out_detail = os.path.join(BASE_DIR, "detailed_results.csv")
    out_sub    = os.path.join(BASE_DIR, "submission.csv")
    df.to_csv(out_detail, index=False)
    df[["id", "score"]].to_csv(out_sub, index=False)
    print(f"\nSaved submission.csv -> {out_sub}")
    print("\n── Score distribution ──────────────────────────────")
    print(df["score"].describe())
    print(f"  score > 0.8: {(df['score'] > 0.8).sum()}")
    print(f"  score > 0.7: {(df['score'] > 0.7).sum()}")
    print(f"  score < 0.6: {(df['score'] < 0.6).sum()}")
    print("\nPer-signal stats:")
    for col in ["weight_cos", "w_norm", "soft_agree", "adv_transfer"]:
        print(f"  {col:14s}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")
    print("\nRun submission.py to submit to the leaderboard.")

if __name__ == "__main__":
    main()
