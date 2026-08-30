@echo off
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
call "C:\Users\DELL\.virtualenvs\LayoutLMv3-gMFc_kGI\Scripts\activate"
python "d:\Internship\Invoice Intelligence Document\YOLOv8x\Fine-tuning\prepare_data.py"
