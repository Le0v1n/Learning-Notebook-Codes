python export.py \
    --weights weights/yolov5s.pt \
    --imgsz 640 \
    --batch-size 1 \
    --device cpu \
    --simplify \
    --include onnx