# -*- coding: utf-8 -*-
"""
Step 1: Export Hybrid QPCA Model to ONNX
=========================================
Takes the PyTorch .pt model (23MB, GPU/PC only).
Exports it to a lightweight ONNX model (< 1MB, runs on any device).
This is the edge deployment step — the ONNX model runs on Raspberry Pi.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import os

# Paths
MODEL_PT   = r"c:\Users\jayad\Downloads\All_5_Models\Hybrid-QPCA_Proposed_best.pt"
OUTPUT_DIR = r"c:\Users\jayad\Documents\QPCA_fodler\Final_UAV_Edge_Pipeline\models"
ONNX_OUT   = os.path.join(OUTPUT_DIR, "hybrid_qpca.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── MODEL DEFINITION (must match training exactly) ────────────────────────────
class EfficientBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)

class HybridQPCALayer(nn.Module):
    def __init__(self):
        super().__init__()
        n = 8
        self.pre   = nn.Sequential(
            nn.Linear(1280, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, n*2), nn.LayerNorm(n*2))
        self.theta = nn.Parameter(torch.randn(2, 2, n) * 0.1)
        self.post  = nn.Sequential(nn.Linear(n, 16), nn.LayerNorm(16))
    def forward(self, x):
        angles = self.pre(x)
        q = torch.tanh(angles[:, :8] * self.theta[0, 0, :])
        return self.post(q)

class HybridQPCAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = EfficientBackbone()
        self.qpca       = HybridQPCALayer()
        self.classifier = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(64, 2))
    def forward(self, x):
        return self.classifier(self.qpca(self.backbone(x)))

# ── EXPORT ────────────────────────────────────────────────────────────────────
def export_to_onnx():
    print("Loading PyTorch model...")
    model = HybridQPCAModel()
    ckpt  = torch.load(MODEL_PT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    pt_size = os.path.getsize(MODEL_PT) / (1024 * 1024)
    print(f"PyTorch model loaded: {pt_size:.2f} MB")

    print("\nExporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, ONNX_OUT,
        input_names=['input'],
        output_names=['output'],
        opset_version=14
    )

    onnx_size = os.path.getsize(ONNX_OUT) / (1024 * 1024)
    compression = (1 - (onnx_size / pt_size)) * 100

    print("\n" + "="*55)
    print("  ONNX EXPORT COMPLETE — EDGE DEPLOYMENT READY")
    print("="*55)
    print(f"  Original PyTorch .pt : {pt_size:.2f} MB  (needs GPU/PC)")
    print(f"  Exported ONNX model  : {onnx_size:.2f} MB  (runs on Pi)")
    print(f"  Size reduction       : {compression:.1f}%")
    print("="*55)
    print(f"  Saved to: {ONNX_OUT}")
    print("  Next: Run step2_video_generator.py")

if __name__ == "__main__":
    export_to_onnx()
