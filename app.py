from flask import Flask, request, jsonify
import numpy as np
import cv2
import json
import gdown
import os

app = Flask(__name__)

DATASET_PATH = "dataset_completo.json"
GDRIVE_URL = "https://drive.google.com/uc?id=1BSxqZG9BhsbsuFHbzL9D2SUm2uOHZcVj"

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        print("📥 Scarico dataset da Google Drive...")
        gdown.download(GDRIVE_URL, DATASET_PATH, quiet=False)
        print("✅ Dataset scaricato.")

def carica_dataset_vettorizzato():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds_vec = {}
    for et, samples in data.items():
        vecs = []
        for d in samples:
            try:
                v = np.array(d["radiale"] + d["spaziale"] +
                             [d["circularity"], d["aspect_ratio"]] + d["hu"], dtype=float)
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    continue
                vecs.append(v)
            except:
                continue
        if vecs:
            ds_vec[et] = vecs
    return ds_vec

def distanza(a, b):
    return np.linalg.norm(a - b)

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity, aspect_ratio = 0.0, 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0.0

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    raggi = [int(min(h, w) * r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m / 255.0 for m in mean])

    spaziale = []
    for (x1, y1, x2, y2) in [(0, 0, cx, cy), (cx, 0, w, cy), (0, cy, cx, h), (cx, cy, w, h)]:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m / 255.0 for m in mean])

    return {"radiale": radiale, "spaziale": spaziale,
            "circularity": circularity, "aspect_ratio": aspect_ratio, "hu": hu}

def vettore_descrittore(d):
    return np.array(d["radiale"] + d["spaziale"] +
                    [d["circularity"], d["aspect_ratio"]] + d["hu"], dtype=float)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nessuna immagine inviata"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Immagine non valida"}), 400

    d_test = estrai_descrittori(img)
    v_test = vettore_descrittore(d_test)

    best_label, best_dist = None, float("inf")
    for et, vecs in ds_vec.items():
        for v_ref in vecs:
            dist = distanza(v_test, v_ref)
            if dist < best_dist:
                best_dist, best_label = dist, et

    return jsonify({"label_predetta": best_label, "distanza": best_dist})

if __name__ == "__main__":
    ensure_dataset()
    ds_vec = carica_dataset_vettorizzato()
    print(f"✅ Dataset caricato ({len(ds_vec)} classi). Server in avvio...")
    app.run(host="0.0.0.0", port=5000)
