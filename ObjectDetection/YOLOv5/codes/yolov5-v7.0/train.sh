python train.py \
    --weights weights/yolov5s.pt \
    --data data/coco128.yaml \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --epochs 2 \
    --batch-size 4 \
    --imgsz 640 \
    --project runs/train \
    --name exp
    