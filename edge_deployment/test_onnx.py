# Quick ONNX Model Test - Scan Multiple Frames
import cv2, numpy as np, onnxruntime as ort

ONNX  = r"c:\Users\jayad\Documents\QPCA_fodler\Final_UAV_Edge_Pipeline\models\hybrid_qpca.onnx"
VIDEO = r"c:\Users\jayad\Documents\QPCA_fodler\Final_UAV_Edge_Pipeline\videos\clip1_evening_long.mp4"

session    = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224)).astype(np.float32)/255.0
    img[:,:,0]=(img[:,:,0]-0.485)/0.229
    img[:,:,1]=(img[:,:,1]-0.456)/0.224
    img[:,:,2]=(img[:,:,2]-0.406)/0.225
    return np.expand_dims(np.transpose(img,(2,0,1)), axis=0)

def softmax(x):
    e=np.exp(x-np.max(x)); return e/e.sum()

cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS) or 5)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames at {fps}fps\n")
print(f"{'Frame':<8} {'NoFire%':<12} {'Fire%':<12} {'Decision'}")
print("-"*45)

for target_frame in [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
    if target_frame >= total:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret: break
    raw   = session.run(None, {input_name: preprocess(frame)})[0]
    probs = softmax(raw[0])
    label = "FIRE" if probs[1] > probs[0] else "safe"
    print(f"{target_frame:<8} {probs[0]*100:<12.1f} {probs[1]*100:<12.1f} {label}")

cap.release()
print("\nMax fire prob frame tells us the right threshold to use.")
