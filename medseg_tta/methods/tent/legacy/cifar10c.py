import logging
import torch
import torch.optim as optim
from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy
import tent
import norm
from conf import cfg, load_cfg_fom_args
logger = logging.getLogger(__name__)

def evaluate(description):
    load_cfg_fom_args(description)
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == 'source':
        logger.info('test-time adaptation: NONE')
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == 'norm':
        logger.info('test-time adaptation: NORM')
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == 'tent':
        logger.info('test-time adaptation: TENT')
        model = setup_tent(base_model)
    for severity in cfg.CORRUPTION.SEVERITY:
        for corruption_type in cfg.CORRUPTION.TYPE:
            try:
                model.reset()
                logger.info('resetting model')
            except:
                logger.warning('not resetting model')
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
            x_test, y_test = (x_test.cuda(), y_test.cuda())
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1.0 - acc
            logger.info(f'error % [{corruption_type}{severity}]: {err:.2%}')

def setup_source(model):
    model.eval()
    logger.info(f'model for evaluation: %s', model)
    return model

def setup_norm(model):
    norm_model = norm.Norm(model)
    logger.info(f'model for adaptation: %s', model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f'stats for adaptation: %s', stat_names)
    return norm_model

def setup_tent(model):
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)
    logger.info(f'model for adaptation: %s', model)
    logger.info(f'params for adaptation: %s', param_names)
    logger.info(f'optimizer for adaptation: %s', optimizer)
    return tent_model

def setup_optimizer(params):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM, dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WD, nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError
if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
