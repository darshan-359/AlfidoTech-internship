"""
CIFAR-10 Image Classifier using Transfer Learning (MobileNetV2)
================================================================
- Transfer learning from ImageNet-pretrained MobileNetV2
- Data augmentation: random crop, flip, color jitter, rotation
- Trains for 15 epochs on CIFAR-10 (10 classes)
- Saves model + plots training curves + prints evaluation metrics
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
NUM_EPOCHS  = 15
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DATA_DIR    = "/home/claude/data"
MODEL_PATH  = "/mnt/user-data/outputs/cifar10_mobilenetv2.pt"
PLOT_PATH   = "/mnt/user-data/outputs/training_curves.png"
METRICS_PATH= "/mnt/user-data/outputs/evaluation_metrics.json"
SEED        = 42

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ─── Transforms ───────────────────────────────────────────────────────────────
# CIFAR-10 images are 32x32; MobileNetV2 expects ≥32; we upsample to 96
train_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.RandomCrop(96, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2023, 0.1994, 0.2010]),
])

val_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std =[0.2023, 0.1994, 0.2010]),
])

# ─── Datasets ─────────────────────────────────────────────────────────────────
print("Downloading / loading CIFAR-10 …")
full_train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_transform)
test_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_transform)

# 90 / 10 train-val split
n_val   = int(0.1 * len(full_train))
n_train = len(full_train) - n_val
train_set, val_set = random_split(full_train, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(SEED))
# Validation uses val_transform → re-wrap
val_set.dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                    transform=val_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_set)}")

# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 10) -> nn.Module:
    """MobileNetV2 with custom classifier head (transfer learning)."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks for fine-tuning
    for layer in list(model.features.children())[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model

model = build_model().to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ─── Training setup ───────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

# ─── Train / eval loop ────────────────────────────────────────────────────────
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "lr":[]}

def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total

print("\n" + "─"*60)
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>9} {'Time':>6}")
print("─"*60)

best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    cur_lr = scheduler.get_last_lr()[0]

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc * 100)
    history["val_acc"].append(vl_acc * 100)
    history["lr"].append(cur_lr)

    flag = " ✓" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc*100:>9.2f}% {vl_loss:>10.4f} "
          f"{vl_acc*100:>8.2f}%{flag:>2} {cur_lr:>9.2e} {time.time()-t0:>5.1f}s")

print("─"*60)
print(f"Best Val Acc: {best_val_acc*100:.2f}%  →  model saved to {MODEL_PATH}")

# ─── Test evaluation ──────────────────────────────────────────────────────────
print("\nLoading best model for test evaluation …")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = (all_preds == all_labels).mean() * 100

print(f"\nTest Accuracy: {test_acc:.2f}%\n")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

metrics = {
    "test_accuracy": round(test_acc, 4),
    "best_val_accuracy": round(best_val_acc * 100, 4),
    "classification_report": classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True
    ),
    "history": history,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {METRICS_PATH}")

# ─── Plot training curves ─────────────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
cm = confusion_matrix(all_labels, all_preds)

fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ACCENT = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#f78166"
YELLOW = "#d29922"
GRID_C = "#21262d"
TEXT_C = "#c9d1d9"

plt.rcParams.update({"text.color": TEXT_C, "axes.labelcolor": TEXT_C,
                     "xtick.color": TEXT_C, "ytick.color": TEXT_C})

def style_ax(ax, title):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color=TEXT_C, fontsize=12, fontweight="bold", pad=10)
    ax.grid(color=GRID_C, linewidth=0.8)
    ax.spines[:].set_color(GRID_C)

# 1. Loss curves
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Loss")
ax1.plot(epochs, history["train_loss"], color=ACCENT,  lw=2, marker="o", ms=4, label="Train")
ax1.plot(epochs, history["val_loss"],   color=ORANGE,  lw=2, marker="s", ms=4, label="Val",   linestyle="--")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend(facecolor="#21262d", edgecolor=GRID_C, labelcolor=TEXT_C)

# 2. Accuracy curves
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Accuracy (%)")
ax2.plot(epochs, history["train_acc"], color=GREEN,  lw=2, marker="o", ms=4, label="Train")
ax2.plot(epochs, history["val_acc"],   color=YELLOW, lw=2, marker="s", ms=4, label="Val",   linestyle="--")
ax2.axhline(test_acc, color=ORANGE, lw=1.5, linestyle=":", label=f"Test {test_acc:.1f}%")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(facecolor="#21262d", edgecolor=GRID_C, labelcolor=TEXT_C)

# 3. Learning-rate schedule
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Learning Rate Schedule")
ax3.plot(epochs, history["lr"], color="#bc8cff", lw=2, marker="o", ms=4)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("LR")
ax3.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

# 4. Per-class accuracy bar chart
report_dict = metrics["classification_report"]
class_accs  = [report_dict[c]["recall"] * 100 for c in CLASSES]
ax4 = fig.add_subplot(gs[1, 0:2])
style_ax(ax4, "Per-Class Accuracy (%)")
bar_colors = [GREEN if v >= 80 else YELLOW if v >= 60 else ORANGE for v in class_accs]
bars = ax4.bar(CLASSES, class_accs, color=bar_colors, edgecolor=GRID_C, linewidth=0.6)
ax4.set_ylim(0, 105)
ax4.set_ylabel("Recall (%)")
ax4.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
for bar, val in zip(bars, class_accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=TEXT_C)
ax4.axhline(test_acc, color=ACCENT, lw=1.2, linestyle="--", label=f"Mean {test_acc:.1f}%")
ax4.legend(facecolor="#21262d", edgecolor=GRID_C, labelcolor=TEXT_C)

# 5. Confusion matrix
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "Confusion Matrix")
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax5.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax5.set_xticks(range(10)); ax5.set_yticks(range(10))
ax5.set_xticklabels([c[:3] for c in CLASSES], rotation=45, ha="right", fontsize=7)
ax5.set_yticklabels([c[:3] for c in CLASSES], fontsize=7)
ax5.set_xlabel("Predicted"); ax5.set_ylabel("True")
for i in range(10):
    for j in range(10):
        ax5.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                 fontsize=6, color="white" if cm_norm[i,j] > 0.5 else "#555")
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

fig.suptitle(f"CIFAR-10  ·  MobileNetV2 Transfer Learning  ·  Test Acc {test_acc:.2f}%",
             color=TEXT_C, fontsize=14, fontweight="bold", y=0.98)

plt.savefig(PLOT_PATH, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Training curves saved to {PLOT_PATH}")
print("\nDone! ✓")
