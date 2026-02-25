import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from dataset import SimpleOCTDataset  # your dataset file

# ===============================
# 🔹 IMPORT YOUR MODELS
# ===============================
from model_architecture import FFSwinClassifier
import torchvision.models.video as models


# ===============================
# 🔹 CONFIGURATION
# ===============================
TEST_DATA_DIR = r"G:\My Drive\PhD_BanglaOCT2025_data\BanglaOCT2025\FFSwin_classifi_Split_Train_Valid_Test_DenoisData\testdata"
FFSWIN_MODEL_PATH = r"G:\My Drive\OCT_Project_PhD_Implementation\Experiment_3_Swin_Main_Train_vali_test_Clinical_Experiment\checkpoints\classifier_best.pth"
RESNET_MODEL_PATH = r"checkpoints_resnet3d\resnet3d18_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ===============================
# 🔹 RESNET 3D DEFINITION
# ===============================
class ResNet3D18_OCT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.r3d_18(weights=None)

        # Single channel OCT
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
# 🔹 PARAMETER COUNT
# ===============================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# 🔹 FLOPs CALCULATION
# ===============================
def compute_flops(model):
    model.eval()

    # Depth must be divisible by 2 for window partition
    D, H, W = 33, 256, 256
    if D % 2 != 0:
        D += 1  # auto padding (34)

    dummy_input = torch.randn(1, 1, D, H, W).to(DEVICE)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    return flops / 1e9  # GFLOPs


# ===============================
# 🔹 INFERENCE TIME
# ===============================
def compute_inference_time(model, dataloader, warmup=10):
    model.eval()

    # Warm-up (important for CUDA)
    with torch.no_grad():
        for i, (vol, _) in enumerate(dataloader):
            if i >= warmup:
                break
            vol = vol.to(DEVICE)
            _ = model(vol)

    torch.cuda.synchronize() if DEVICE.type == "cuda" else None

    start = time.time()

    with torch.no_grad():
        for vol, _ in dataloader:
            vol = vol.to(DEVICE)
            _ = model(vol)

    torch.cuda.synchronize() if DEVICE.type == "cuda" else None

    end = time.time()

    total_time = end - start
    avg_time = total_time / len(dataloader)

    return avg_time


# ===============================
# 🔹 MAIN FUNCTION
# ===============================
def evaluate_model(model_name, model, model_path):

    print(f"\n==============================")
    print(f"Evaluating {model_name}")
    print(f"==============================")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Dataset
    test_ds = SimpleOCTDataset(root_dir=TEST_DATA_DIR, return_label=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Parameters
    params = count_parameters(model)

    # FLOPs
    gflops = compute_flops(model)

    # Inference time
    avg_time = compute_inference_time(model, test_loader)

    print(f"📌 Trainable Parameters: {params:,}")
    print(f"📌 GFLOPs per volume: {gflops:.3f}")
    print(f"📌 Avg Inference Time per Patient: {avg_time:.4f} sec")

    return params, gflops, avg_time


# ===============================
# 🔹 RUN
# ===============================
if __name__ == "__main__":

    # ---- FFSWIN ----
    ffswin_model = FFSwinClassifier(num_classes=3)
    ffswin_results = evaluate_model(
        "FFSwin",
        ffswin_model,
        FFSWIN_MODEL_PATH
    )

    # ---- RESNET 3D ----
    resnet_model = ResNet3D18_OCT(num_classes=3)
    resnet_results = evaluate_model(
        "ResNet3D-18",
        resnet_model,
        RESNET_MODEL_PATH
    )

    print("\n==============================")
    print("FINAL COMPUTATIONAL SUMMARY")
    print("==============================")

    print(f"Swin 3D (No-Shift) -> Params: {ffswin_results[0]:,}, "
          f"GFLOPs: {ffswin_results[1]:.3f}, "
          f"Time: {ffswin_results[2]:.4f} sec")
    print(f"FFSwin -> Params: {ffswin_results[0]:,}, "
          f"GFLOPs: {ffswin_results[1]:.3f}, "
          f"Time: {ffswin_results[2]:.4f} sec")


    print(f"ResNet3D-18 -> Params: {resnet_results[0]:,}, "
          f"GFLOPs: {resnet_results[1]:.3f}, "
          f"Time: {resnet_results[2]:.4f} sec")
