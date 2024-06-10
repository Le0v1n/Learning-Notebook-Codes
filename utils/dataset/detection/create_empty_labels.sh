python utils/dataset/detection/create_empty_labels.py \
    --image-path Datasets/VOC2012-5000/VOC2012/images \
    --target-path Datasets/VOC2012-5000/VOC2012/negative_labels \
    --target-format txt \
    --num-threading 16 \
    --override