@echo off
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
call "C:\Users\DELL\.virtualenvs\LayoutLMv3-gMFc_kGI\Scripts\activate"

echo Starting Fine-tuning on 43 Concentrix Invoices...
python "d:\Internship\Invoice Intelligence Document\YOLOv8x\yolov8\train.py" ^
  --model "d:\Internship\Invoice Intelligence Document\YOLOv8x\Inference-base\yolov8x.pt" ^
  --data_path "d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\concentrix.yaml" ^
  --epochs 100 ^
  --imgsz 1280 ^
  --batch 8 ^
  --ch 6 ^
  --name "concentrix_finetune" ^
  --project "d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\runs"

pause
