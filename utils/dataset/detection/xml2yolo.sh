python utils/dataset/detection/xml2yolo-remaster.py \
    --image-path Datasets/VOC2012-5000/VOC2012/images \
    --label-path Datasets/VOC2012-5000/VOC2012/xmls.bak \
    --target-path Datasets/VOC2012-5000/VOC2012/yolos \
    --num-threading 32 \
    --classes "aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" "train" "tvmonitor"
