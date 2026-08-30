import os
import sys
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from paddleocr import PaddleOCR

# 1. SETUP PATHS
SCRIPT_DIR = Path(__file__).parent.absolute()
# Weight path from the user's latest fine-tuning run
MODEL_PATH = Path(r"d:\Internship\Invoice Intelligence Document\YOLOv8x\concentrix_finetune2\weights\best.pt")
# Local ultralytics for 6-channel support
sys.path.insert(0, str(SCRIPT_DIR))

# 2. WINDOWS DLL FIX
VENV_BASE = r"C:\Users\DELL\.virtualenvs\LayoutLMv3-gMFc_kGI"
if sys.platform == "win32":
    possible_paths = [
        os.path.join(VENV_BASE, "Lib", "site-packages", "torch", "lib"),
        os.path.join(VENV_BASE, "Lib", "site-packages", "paddle", "libs"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except AttributeError:
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

try:
    from ultralytics import YOLO
    print("Ultralytics/YOLO loaded successfully.")
except Exception as e:
    print(f"Error loading ultralytics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- CHARGRID LOGIC ---
characters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b',
              'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', '|', '£', '°', 'À', 'Ç', 'È', 'É', 'Ô', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ô', 'û', '€']

def create_encodings_three_digit_0(chars):
    chars = ["", *chars]
    number_digits = 3
    base = np.ceil(np.cbrt(len(chars))).astype(int)
    encodings = {c: np.base_repr(i, base=base).zfill(number_digits) for i, c in enumerate(chars)}
    encodings = {c: np.array([int(v) for v in e]) for c, e in encodings.items()}
    encodings = {c: e / (base - 1) for c, e in encodings.items()}
    return encodings

def get_char_grid(ocr_data, target_size, encodings):
    h, w = target_size
    grid = np.zeros((h, w, 3), dtype=np.float32)
    for box, text in ocr_data:
        x0, y0, x1, y1 = box
        for i, char in enumerate(text):
            if char in encodings:
                char_w = (x1 - x0) / len(text)
                cx0 = int(x0 + i * char_w)
                cx1 = int(x0 + (i + 1) * char_w)
                grid[y0:y1, cx0:cx1, :] = encodings[char]
    return grid

# --- INFERENCE ENGINE ---
def run_inference(image_path):
    print(f"--- Running Inference on: {Path(image_path).name} ---")
    
    # Load Model
    model = YOLO(str(MODEL_PATH))
    
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # Read Image
    img_bgr = cv2.imread(image_path)
    # Ensure it is at the training resolution 1280x1664
    img_bgr = cv2.resize(img_bgr, (1280, 1664), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # 1. OCR
    results = ocr.ocr(img_rgb, cls=True)
    ocr_data = []
    if results and results[0]:
        for line in results[0]:
            pts, (txt, score) = line[0], line[1]
            x0, y0 = int(min(p[0] for p in pts)), int(min(p[1] for p in pts))
            x1, y1 = int(max(p[0] for p in pts)), int(max(p[1] for p in pts))
            ocr_data.append([[x0, y0, x1, y1], txt])
            
    # 2. Chargrid
    encodings = create_encodings_three_digit_0(characters)
    grid = get_char_grid(ocr_data, (h, w), encodings)
    grid = (grid * 255).astype(np.uint8)
    
    # 3. Stack (6-channel)
    stacked = np.concatenate((grid, img_rgb), axis=2)
    
    # 4. YOLO Predict
    # YOLO usually expects 3-channel input, but our custom ultralytics handles 6-channel .npy or numpy arrays
    results = model.predict(stacked, imgsz=(1664, 1280), conf=0.25)
    
    # 5. Process Detections
    output = []
    names = model.names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            c = int(box.cls)
            conf = float(box.conf)
            label = names[c]
            
            # Match text
            match_text = ""
            best_iou = 0
            for obox, otxt in ocr_data:
                # Simple intersection check
                ix0, iy0 = max(b[0], obox[0]), max(b[1], obox[1])
                ix1, iy1 = min(b[2], obox[2]), min(b[3], obox[3])
                if ix1 > ix0 and iy1 > iy0:
                    area = (ix1 - ix0) * (iy1 - iy0)
                    if area > best_iou:
                        best_iou = area
                        match_text = otxt
            
            output.append({
                "label": label,
                "confidence": conf,
                "bbox": b,
                "text": match_text
            })
            
            # Draw on image
            cv2.rectangle(img_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label}: {conf:.2f}", (int(b[0]), int(b[1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save Results
    out_img_path = SCRIPT_DIR / f"result_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_img_path), img_bgr)
    
    out_json_path = SCRIPT_DIR / f"result_{Path(image_path).stem}.json"
    with open(out_json_path, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"Done! Results saved to:\n  - {out_img_path}\n  - {out_json_path}")

if __name__ == "__main__":
    img_to_test = r"d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\processed_data\images_source\invoice_CPC000692108.jpg"
    run_inference(img_to_test)
