import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
_C = CfgNode()
cfg = _C
_C.MODEL = CfgNode()
_C.MODEL.ARCH = 'Standard'
_C.MODEL.ADAPTATION = 'source'
_C.MODEL.EPISODIC = False
_C.CORRUPTION = CfgNode()
_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]
_C.CORRUPTION.NUM_EX = 10000
_C.BN = CfgNode()
_C.BN.EPS = 1e-05
_C.BN.MOM = 0.1
_C.OPTIM = CfgNode()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 0.001
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WD = 0.0
_C.TEST = CfgNode()
_C.TEST.BATCH_SIZE = 128
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
_C.DESC = ''
_C.RNG_SEED = 1
_C.SAVE_DIR = './output'
_C.DATA_DIR = './data'
_C.CKPT_DIR = './ckpt'
_C.LOG_DEST = 'log.txt'
_C.LOG_TIME = ''
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

def assert_and_infer_cfg():
    err_str = 'Unknown adaptation method.'
    assert _C.MODEL.ADAPTATION in ['source', 'norm', 'tent']
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ['stdout', 'file'], err_str.format(_C.LOG_DEST)

def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, 'r') as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)

def dump_cfg():
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, 'w') as f:
        _C.dump(stream=f)

def load_cfg(out_dir, cfg_dest='config.yaml'):
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)

def reset_cfg():
    cfg.merge_from_other_cfg(_CFG_DEFAULT)

def load_cfg_fom_args(description='Config options.'):
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True, help='Config file location')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='See conf.py for all options')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = (current_time, log_dest)
    cfg.freeze()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s', datefmt='%y/%m/%d %H:%M:%S', handlers=[logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)), logging.StreamHandler()])
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info('PyTorch Version: torch={}, cuda={}, cudnn={}'.format(*version))
    logger.info(cfg)
