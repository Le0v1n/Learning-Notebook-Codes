python utils/dataset/detection/collect_negative_samples.py \
    --image-path Datasets/VOC2012-5000/VOC2012/images \
    --label-path Datasets/VOC2012-5000/VOC2012/jsons \
    --label-format json \
    --target-path Datasets/VOC2012-5000/VOC2012/negative_samples \
    --num-threading 8 \
    --classes "aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" "train" "tvmonitor"