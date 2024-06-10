python utils/dataset/detection/create_dataset.py \
    --image-path Datasets/raw_data/images \
    --label-path Datasets/raw_data/jsons \
    --label-format json \
    --target-path Datasets/abc/voc2012-Le0v1n \
    --split-ratio 0.7 0.2 0.1 \
    --num-threading 1