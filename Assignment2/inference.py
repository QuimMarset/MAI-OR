import os
import mmcv
import torch
import numpy as np
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from plot_utils import plot_segmentation
from load_utils import load_image, load_segmentation


def generate_segmentation_comparison_images(npy_paths, gt_paths, image_paths, save_path):
    for npy_name in os.listdir(npy_paths):
        image_name = npy_name[:-4]
        npy_path = os.path.join(npy_paths, npy_name)
        
        image = load_image(image_paths, image_name)
        gt_segmentation = load_segmentation(gt_paths, image_name)
        predicted_segmentation = np.load(npy_path)
        plot_segmentation(image, gt_segmentation, predicted_segmentation, save_path, image_name)



def load_config(config_path):
    config = mmcv.Config.fromfile(config_path)
    config.model.pretrained = None
    config.data.test.test_mode = True
    config.model.train_cfg = None
    return config


def prepare_env(config):
    setup_multi_processes(config)
    if config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


def build_data_loader(config):
    # build the dataloader
    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=config.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    return data_loader


def build_model(config, checkpoint_path):
    # build the model and load checkpoint
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=config.gpu_ids)
    model.eval()
    return model


def perform_inference(config_path, checkpoint_path, save_json_path, save_predictions_path):
    # Json file where the segmentation metrics will be dumped
    json_file = os.path.join(save_json_path, f'validation_segmentation_metrics.json')
   
    config = load_config(config_path)
    prepare_env(config)
    data_loader = build_data_loader(config)
    model = build_model(config, checkpoint_path)
    
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for data in data_loader:
        image_meta = data['img_metas'][0].data[0][0]
        image_name = image_meta['ori_filename'][:-4]

        with torch.no_grad():
            predicted_segm = model(return_loss=False, **data)

        np.save(os.path.join(save_predictions_path, f'{image_name}.npy'), predicted_segm[0])
        results.extend(predicted_segm)

        batch_size = len(predicted_segm)
        for _ in range(batch_size):
            prog_bar.update()

    eval_kwargs = {}
    eval_kwargs.update(metric=['mIoU', 'mDice'])
    metric = dataset.evaluate(results, **eval_kwargs)
    metric_dict = dict(config=config_path, metric=metric)
    mmcv.dump(metric_dict, json_file, indent=4)


if __name__ == '__main__':

    root_path = './'

    config_path = os.path.join(root_path, 'config.py')

    results_path = os.path.join(root_path, 'results')
    metrics_path = os.path.join(results_path, 'experiment_metrics')
    statistics_path = os.path.join(results_path, 'data_statistics')
    predictions_path = os.path.join(results_path, 'test_predictions')
    experiments_path = os.path.join(results_path, 'experiments')

    data_path = os.path.join(root_path, 'fashionpedia')
    train_images_path = os.path.join(data_path, 'images', 'train')
    test_images_path = os.path.join(data_path, 'images', 'val')
    test_segmentations_path = os.path.join(data_path, 'segmentations', 'val')

    best_model_path_path = os.path.join(results_path, 'best_model')
    checkpoint_path = os.path.join(best_model_path_path, 'latest.pth')

    # Setting plot_segmentations=True will generate all the predicted segmentations compared to their ground-truth (the process can take some time)
    perform_inference(config_path, checkpoint_path, test_images_path, test_segmentations_path, predictions_path, metrics_path, plot_segmentations=False)
