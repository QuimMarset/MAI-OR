import os
import time
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes


def load_config(config_path, checkpoints_path):
    config = mmcv.Config.fromfile(config_path)
    config.model.pretrained = None
    config.data.test.test_mode = True
    config.model.train_cfg = None
    config.work_dir = checkpoints_path
    return config


def prepare_env(config, seed):
    setup_multi_processes(config)
    if config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    set_random_seed(seed, deterministic=False)
    config.seed = seed


def init_logger(config, seed, logger_path):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(logger_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=config.log_level)
    
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Config:\n{config.pretty_text}')
    logger.info(f'Set random seed to {seed}')
    
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    meta['env_info'] = env_info
    meta['seed'] = seed
    meta.update(config.checkpoint_config.meta)

    return meta, timestamp, logger
    

def build_model(config, classes):
    model = build_segmentor(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'))
    model.init_weights()
    model = revert_sync_batchnorm(model)
    model.CLASSES = classes
    return model


def train_model(config_path, seed, save_path):

    config = load_config(config_path, save_path)
    prepare_env(config, seed)
    
    datasets = [build_dataset(config.data.train)]

    config.checkpoint_config.meta = dict(
        CLASSES=datasets[0].CLASSES,
        PALETTE=datasets[0].PALETTE)

    model = build_model(config, datasets[0].CLASSES)

    meta, timestamp, logger = init_logger(config, seed, save_path)
    logger.info(model)

    """
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    """

    train_segmentor(
        model,
        datasets,
        config,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    train_model()
