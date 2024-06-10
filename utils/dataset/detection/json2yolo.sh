python utils/dataset/detection/json2yolo.py \
    --image-path Datasets/VOC2012-5000/VOC2012/images \
    --label-path Datasets/VOC2012-5000/VOC2012/jsons \
    --target-path Datasets/VOC2012-5000/VOC2012/yolos \
    --num-threading 16 \
    --override \
    --classes "aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" "train" "tvmonitor"
