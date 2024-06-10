python utils/dataset/detection/yolo2json.py \
    --image-path Datasets/VOC2012-5000/VOC2012/images \
    --label-path Datasets/VOC2012-5000/VOC2012/yolos \
    --target-path Datasets/VOC2012-5000/VOC2012/jsons \
    --num-threading 16 \
    --classes "aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" "train" "tvmonitor"
