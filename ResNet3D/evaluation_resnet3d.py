import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from datetime import datetime
from math import sqrt
import torchvision.models.video as models

from dataset import SimpleOCTDataset


# ===============================
# IEEE TMI STYLE SETTINGS
# ===============================
plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30


# ===============================
# ResNet3D-18 Architecture
# ===============================
class ResNet3D18_OCT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.r3d_18(weights=None)

        # Modify first conv for single-channel OCT
        self.model.stem[0] = nn.Conv3d(
            1, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


# ===============================
# Evaluation Function
# ===============================
def evaluate_resnet3d():

    print("📊 Evaluating 3D ResNet-18 Baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    data_dir = r"g:\My Drive\PhD_BanglaOCT2025_data\BanglaOCT2025\FFSwin_classifi_Split_Train_Valid_Test_DenoisData\testdata"
    model_path = r"checkpoints_resnet3d\resnet3d18_best.pth"

    save_dir = "./resnet3d_results_last"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"evaluation_log_{timestamp}.txt")

    dataset = SimpleOCTDataset(root_dir=data_dir, return_label=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ResNet3D18_OCT(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for vol, label in loader:

            vol = vol.to(device)
            label = label.to(device)

            output = model(vol)
            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(output, 1)

            all_preds.append(pred.item())
            all_labels.append(label.item())
            all_probs.append(probs.cpu().numpy()[0])

            # Extract penultimate features
            features = model.model.avgpool(
                model.model.layer4(
                    model.model.layer3(
                        model.model.layer2(
                            model.model.layer1(
                                model.model.stem(vol)
                            )
                        )
                    )
                )
            )
            features = features.flatten(1)
            all_features.append(features.cpu().numpy()[0])

    all_probs = np.array(all_probs)
    all_features = np.array(all_features)

    class_names = ['Dry', 'Wet', 'NonAMD']
    cm = confusion_matrix(all_labels, all_preds)

    accuracy = np.trace(cm) / np.sum(cm)
    accuracy_percent = accuracy * 100

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    n = len(all_labels)
    ci = 1.96 * sqrt((accuracy * (1 - accuracy)) / n)
    ci_percent = ci * 100

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   digits=4)

    # ===============================
    # SAVE LOG FILE
    # ===============================
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("3D ResNet-18 Evaluation Report\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Test Samples: {n}\n")
        f.write(f"Accuracy: {accuracy_percent:.2f}%\n")
        f.write(f"95% CI: ±{ci_percent:.2f}%\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Weighted-F1: {weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"3D ResNet-18 Confusion Matrix\nAcc={accuracy_percent:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ConfusionMatrix_{timestamp}.png"), dpi=300)
    plt.close()

    # ===============================
    # ROC CURVE
    # ===============================
    y_bin = label_binarize(all_labels, classes=[0,1,2])

    plt.figure(figsize=(10,8))

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=3,
                 label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--',linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("3D ResNet-18 Multi-class ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ROC_{timestamp}.png"), dpi=300)
    plt.close()

    # ===============================
    # t-SNE
    # ===============================
    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(10,8))

    for i, label in enumerate(class_names):
        idx = np.array(all_labels) == i
        plt.scatter(tsne_results[idx,0],
                    tsne_results[idx,1],
                    s=120,
                    label=label)

    plt.title("3D ResNet-18 t-SNE Embedding")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"TSNE_{timestamp}.png"), dpi=300)
    plt.close()

    print("\n✅ 3D ResNet-18 results saved in:", save_dir)
    print(f"Accuracy: {accuracy_percent:.2f}% | CI ±{ci_percent:.2f}% | Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    evaluate_resnet3d()
