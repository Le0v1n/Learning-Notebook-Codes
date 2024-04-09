import cv2

DET_TASKS = ('detection', 'det', 'detect')
SEG_TASKS = ('segmentation', 'semantic segmentation', 'seg')
ImageFormat = ('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.svg', '.raw', 'webp', '.heic', '.heif')
LabelFormat = ('.csv', '.json', '.txt', '.xml', '.yaml', '.yml')
VideoFormat = ('.mp4', '.avi', '.mkv', '.wmv', '.flv', '.mov', 'rmvb', '.webm', '.mpg', '.vob')

fourcc = {
    '.mp4': cv2.VideoWriter_fourcc(*'mp4v'),
    '.avi': cv2.VideoWriter_fourcc(*'XVID'),
    '.mkv': cv2.VideoWriter_fourcc(*'XVID'),
    '.wmv': cv2.VideoWriter_fourcc(*'WMV1'),
    '.flv': cv2.VideoWriter_fourcc(*'FLV1'),
    '.mov': cv2.VideoWriter_fourcc(*'mp4v'),
    '.webm': cv2.VideoWriter_fourcc(*'VP80'),
    '.mpg': cv2.VideoWriter_fourcc(*'MPG2'),
    '.vob': cv2.VideoWriter_fourcc(*'XVID'),
}