import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ======================= IEEE TMI FONT =======================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Palatino Linotype"]
rcParams["font.size"] = 11
rcParams["axes.titlesize"] = 12
rcParams["axes.labelsize"] = 11
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10

# ======================= READ LOG FILE =======================
log_text ="""
Epoch [1/100] | Loss 1.0499 | TrainAcc 33.44% | ValAcc 7.81% | LR 5.00e-05
Epoch [2/100] | Loss 1.0382 | TrainAcc 34.65% | ValAcc 9.38% | LR 5.00e-05
Epoch [3/100] | Loss 1.0282 | TrainAcc 39.20% | ValAcc 7.81% | LR 5.00e-05
Epoch [4/100] | Loss 1.0138 | TrainAcc 40.89% | ValAcc 9.38% | LR 5.00e-05
Epoch [5/100] | Loss 0.9863 | TrainAcc 43.85% | ValAcc 7.81% | LR 5.00e-05
Epoch [6/100] | Loss 0.9145 | TrainAcc 49.66% | ValAcc 10.94% | LR 5.00e-05
Epoch [7/100] | Loss 0.8397 | TrainAcc 54.25% | ValAcc 7.81% | LR 5.00e-05
Epoch [8/100] | Loss 0.7892 | TrainAcc 56.37% | ValAcc 9.38% | LR 5.00e-05
Epoch [9/100] | Loss 0.7688 | TrainAcc 57.11% | ValAcc 12.50% | LR 5.00e-05
Epoch [10/100] | Loss 0.7112 | TrainAcc 59.32% | ValAcc 9.38% | LR 5.00e-05
Epoch [11/100] | Loss 0.7179 | TrainAcc 58.16% | ValAcc 9.38% | LR 5.00e-05
Epoch [12/100] | Loss 0.6821 | TrainAcc 59.32% | ValAcc 12.50% | LR 5.00e-05
Epoch [13/100] | Loss 0.6470 | TrainAcc 61.12% | ValAcc 10.94% | LR 5.00e-05
Epoch [14/100] | Loss 0.6431 | TrainAcc 61.97% | ValAcc 18.75% | LR 5.00e-05
Epoch [15/100] | Loss 0.6022 | TrainAcc 63.76% | ValAcc 12.50% | LR 5.00e-05
Epoch [16/100] | Loss 0.5891 | TrainAcc 63.76% | ValAcc 18.75% | LR 5.00e-05
Epoch [17/100] | Loss 0.5564 | TrainAcc 66.24% | ValAcc 17.19% | LR 5.00e-05
Epoch [18/100] | Loss 0.5373 | TrainAcc 68.09% | ValAcc 10.94% | LR 5.00e-05
Epoch [19/100] | Loss 0.5122 | TrainAcc 68.62% | ValAcc 20.31% | LR 5.00e-05
Epoch [20/100] | Loss 0.4991 | TrainAcc 69.89% | ValAcc 23.44% | LR 5.00e-05
Epoch [21/100] | Loss 0.4706 | TrainAcc 71.05% | ValAcc 25.00% | LR 5.00e-05
Epoch [22/100] | Loss 0.4411 | TrainAcc 72.95% | ValAcc 34.38% | LR 5.00e-05
Epoch [23/100] | Loss 0.4293 | TrainAcc 75.12% | ValAcc 46.88% | LR 5.00e-05
Epoch [24/100] | Loss 0.4177 | TrainAcc 76.81% | ValAcc 17.19% | LR 5.00e-05
Epoch [25/100] | Loss 0.4168 | TrainAcc 75.12% | ValAcc 29.69% | LR 5.00e-05
Epoch [26/100] | Loss 0.3710 | TrainAcc 79.03% | ValAcc 37.50% | LR 5.00e-05
Epoch [27/100] | Loss 0.3643 | TrainAcc 78.98% | ValAcc 43.75% | LR 5.00e-05
Epoch [28/100] | Loss 0.3328 | TrainAcc 81.51% | ValAcc 31.25% | LR 5.00e-05
Epoch [29/100] | Loss 0.3373 | TrainAcc 80.45% | ValAcc 64.06% | LR 5.00e-05
Epoch [30/100] | Loss 0.3037 | TrainAcc 83.99% | ValAcc 46.88% | LR 5.00e-05
Epoch [31/100] | Loss 0.3034 | TrainAcc 83.73% | ValAcc 43.75% | LR 5.00e-05
Epoch [32/100] | Loss 0.2754 | TrainAcc 84.84% | ValAcc 54.69% | LR 5.00e-05
Epoch [33/100] | Loss 0.2481 | TrainAcc 87.11% | ValAcc 48.44% | LR 5.00e-05
Epoch [34/100] | Loss 0.2365 | TrainAcc 88.01% | ValAcc 53.12% | LR 5.00e-05
Epoch [35/100] | Loss 0.2337 | TrainAcc 88.54% | ValAcc 31.25% | LR 5.00e-05
Epoch [36/100] | Loss 0.2484 | TrainAcc 87.27% | ValAcc 39.06% | LR 5.00e-05
Epoch [37/100] | Loss 0.2110 | TrainAcc 89.70% | ValAcc 56.25% | LR 5.00e-05
Epoch [38/100] | Loss 0.2034 | TrainAcc 90.23% | ValAcc 62.50% | LR 5.00e-05
Epoch [39/100] | Loss 0.1845 | TrainAcc 91.81% | ValAcc 65.62% | LR 5.00e-05
Epoch [40/100] | Loss 0.1705 | TrainAcc 91.97% | ValAcc 64.06% | LR 5.00e-05
Epoch [41/100] | Loss 0.1661 | TrainAcc 92.66% | ValAcc 53.12% | LR 5.00e-05
Epoch [42/100] | Loss 0.1503 | TrainAcc 93.40% | ValAcc 71.88% | LR 5.00e-05
Epoch [43/100] | Loss 0.1641 | TrainAcc 93.08% | ValAcc 79.69% | LR 5.00e-05
Epoch [44/100] | Loss 0.1264 | TrainAcc 94.77% | ValAcc 81.25% | LR 5.00e-05
Epoch [45/100] | Loss 0.1851 | TrainAcc 91.60% | ValAcc 78.12% | LR 5.00e-05
Epoch [46/100] | Loss 0.1167 | TrainAcc 95.40% | ValAcc 53.12% | LR 5.00e-05
Epoch [47/100] | Loss 0.1268 | TrainAcc 94.88% | ValAcc 64.06% | LR 5.00e-05
Epoch [48/100] | Loss 0.1145 | TrainAcc 95.40% | ValAcc 75.00% | LR 5.00e-05
Epoch [49/100] | Loss 0.1097 | TrainAcc 95.99% | ValAcc 85.94% | LR 5.00e-05
Epoch [50/100] | Loss 0.0997 | TrainAcc 95.99% | ValAcc 78.12% | LR 5.00e-05
Epoch [51/100] | Loss 0.1010 | TrainAcc 96.04% | ValAcc 43.75% | LR 5.00e-05
Epoch [52/100] | Loss 0.0889 | TrainAcc 96.67% | ValAcc 81.25% | LR 5.00e-05
Epoch [53/100] | Loss 0.0787 | TrainAcc 97.31% | ValAcc 81.25% | LR 5.00e-05
Epoch [54/100] | Loss 0.0652 | TrainAcc 98.10% | ValAcc 76.56% | LR 5.00e-05
Epoch [55/100] | Loss 0.1173 | TrainAcc 94.35% | ValAcc 89.06% | LR 5.00e-05
Epoch [56/100] | Loss 0.0668 | TrainAcc 97.89% | ValAcc 87.50% | LR 5.00e-05
Epoch [57/100] | Loss 0.0631 | TrainAcc 97.62% | ValAcc 84.38% | LR 5.00e-05
Epoch [58/100] | Loss 0.0611 | TrainAcc 97.73% | ValAcc 87.50% | LR 5.00e-05
Epoch [59/100] | Loss 0.0499 | TrainAcc 98.68% | ValAcc 79.69% | LR 5.00e-05
Epoch [60/100] | Loss 0.0643 | TrainAcc 97.57% | ValAcc 89.06% | LR 5.00e-05
Epoch [61/100] | Loss 0.0683 | TrainAcc 97.57% | ValAcc 75.00% | LR 5.00e-05
Epoch [62/100] | Loss 0.0444 | TrainAcc 98.94% | ValAcc 75.00% | LR 5.00e-05
Epoch [63/100] | Loss 0.0379 | TrainAcc 98.89% | ValAcc 85.94% | LR 5.00e-05
Epoch [64/100] | Loss 0.1182 | TrainAcc 95.30% | ValAcc 79.69% | LR 5.00e-05
Epoch [65/100] | Loss 0.0383 | TrainAcc 98.94% | ValAcc 87.50% | LR 5.00e-05
"""

# ======================= PARSE EPOCH METRICS =======================
epochs, loss, train_acc, val_acc = [], [], [], []

epoch_re = re.compile(
    r"Epoch\s+\[(\d+)/\d+\].*?Loss\s+([\d.]+).*?TrainAcc\s+([\d.]+)%.*?ValAcc\s+([\d.]+)%"
)

for line in log_text.splitlines():
    m = epoch_re.search(line)
    if m:
        epochs.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_acc.append(float(m.group(4)))

epochs = np.array(epochs)
train_acc = np.array(train_acc)
val_acc = np.array(val_acc)
loss = np.array(loss)

# ======================= BEST EPOCH =======================
best_idx = np.argmax(val_acc)
best_epoch = epochs[best_idx]
best_val = val_acc[best_idx]
best_loss = loss[best_idx]

# ======================= 95% CI (Bootstrap on validation) =======================
def bootstrap_ci(data, n=2000):
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

ci_lo, ci_hi = [], []
win = 5  # rolling window
for i in range(len(val_acc)):
    s = max(0, i-win)
    e = min(len(val_acc), i+win)
    lo, hi = bootstrap_ci(val_acc[s:e])
    ci_lo.append(lo)
    ci_hi.append(hi)

ci_lo = np.array(ci_lo)
ci_hi = np.array(ci_hi)

# ======================= FIGURE 1: TRAIN vs VAL ACC =======================
plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, linewidth=1.5, label="Train Accuracy")
plt.plot(epochs, val_acc, linewidth=1.8, label="Validation Accuracy")
plt.fill_between(epochs, ci_lo, ci_hi, alpha=0.2, color='orange', label="95% CI")
plt.scatter(best_epoch, best_val, s=60, color='red', zorder=5, label=f"Best Epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig1_Accuracy_CI.png", dpi=300)
plt.close()

# ======================= FIGURE 2: LOSS CURVE =======================
plt.figure(figsize=(6,4))
plt.plot(epochs, loss, color='blue', linewidth=1.8, label="Training Loss")
plt.scatter(best_epoch, best_loss, s=60, color='red', zorder=5, label=f"Best Epoch ({best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig2_Loss_Curve.png", dpi=300)
plt.close()

print("✅ Figures generated: Fig1_Accuracy_CI.png and Fig2_Loss_Curve.png")
