# -*- coding: utf-8 -*-
"""
Step 3 - Raspberry Pi Version
Fire Alert Extractor using ONNX Model
"""
import cv2, os, time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Pi paths (from shared folder)
ONNX_MODEL_PATH = "/mnt/qpca/models/hybrid_qpca.onnx"
VIDEO_DIR       = "/mnt/qpca/videos"
ALERTS_DIR      = "/mnt/qpca/Fire_Alerts"

CONFIDENCE_THRESHOLD = 0.80
os.makedirs(ALERTS_DIR, exist_ok=True)

def preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

print("Loading ONNX model on Raspberry Pi...")
session    = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("Model loaded!")

videos = list(Path(VIDEO_DIR).glob("*.mp4"))
print(f"Found {len(videos)} videos to scan")

fire_saved = 0
for vid_path in videos:
    print(f"Scanning: {vid_path.name}")
    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % int(fps) == 0:
            t0 = time.time()
            probs     = softmax(session.run(None, {input_name: preprocess_image(frame)})[0])
            fire_prob = probs[0][1]
            ms        = (time.time() - t0) * 1000
            if fire_prob >= CONFIDENCE_THRESHOLD:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (10,10), (450,80), (0,0,0), -1)
                cv2.putText(frame, "FIRE ALERT - HIGH CONFIDENCE", (20,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,255), 2)
                cv2.putText(frame, f"Prob: {fire_prob*100:.1f}%  |  Latency: {ms:.0f}ms",
                            (20,62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 8)
                fname = f"{vid_path.stem}_f{frame_idx}_c{fire_prob*100:.0f}.jpg"
                cv2.imwrite(os.path.join(ALERTS_DIR, fname), frame)
                fire_saved += 1
                print(f"  [ALERT] Frame {frame_idx} | {fire_prob*100:.1f}% | {ms:.0f}ms")
        frame_idx += 1
    cap.release()

print(f"\nDone! {fire_saved} fire alerts saved to {ALERTS_DIR}")
