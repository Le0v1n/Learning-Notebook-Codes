from pathlib import Path
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


def main(args):
    # read csv file
    df = pd.read_csv(args.csv)
    # 清理列名中的空格
    df.columns = df.columns.str.strip()

    # 清理每个字段值中的空格
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    dirpath = Path(args.csv).parent

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(dirpath)

    # 遍历CSV文件的每一行
    for index, row in df.iterrows():
        epoch = row['epoch']
        # 将每一列的数据写入TensorBoard
        writer.add_scalar('train/box_loss', row['train/box_loss'], epoch)
        writer.add_scalar('train/cls_loss', row['train/cls_loss'], epoch)
        writer.add_scalar('train/dfl_loss', row['train/dfl_loss'], epoch)
        writer.add_scalar('metrics/precision(B)', row['metrics/precision(B)'], epoch)
        writer.add_scalar('metrics/recall(B)', row['metrics/recall(B)'], epoch)
        writer.add_scalar('metrics/mAP50(B)', row['metrics/mAP50(B)'], epoch)
        writer.add_scalar('metrics/mAP50-95(B)', row['metrics/mAP50-95(B)'], epoch)
        writer.add_scalar('val/box_loss', row['val/box_loss'], epoch)
        writer.add_scalar('val/cls_loss', row['val/cls_loss'], epoch)
        writer.add_scalar('val/dfl_loss', row['val/dfl_loss'], epoch)
        writer.add_scalar('lr/pg0', row['lr/pg0'], epoch)
        writer.add_scalar('lr/pg1', row['lr/pg1'], epoch)
        writer.add_scalar('lr/pg2', row['lr/pg2'], epoch)

    # 关闭SummaryWriter
    writer.close()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--csv', type=str, help='The filepath of csv file wanna convert')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
