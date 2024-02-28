import os
import sys
from mmengine.config import Config

sys.path.append(os.getcwd())
from utils.common_fn import xprint


cfg = Config.fromfile('OpenMMLab/MMEngine/codes/example_configs/learn_read_config.py')
xprint(cfg, color='red', hl='>')

xprint(cfg.test_int, color='green', hl='-', hl_style='full')
xprint(cfg['test_list'], color='green')
xprint(cfg['test_dict'], color='green')

cfg['test_list'][1] = 3  # 修改字典中的值
xprint(cfg['test_list'], color='green', hl='-', hl_style='full')