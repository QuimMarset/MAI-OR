import os
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from plot_utils import plot_segmentation
from load_utils import load_image, load_segmentation


def compare_segmentations(image_name, predicted_segm, image_paths, segmentation_paths, save_path):
    image = load_image(image_paths, image_name)
    gt_segmentation = load_segmentation(segmentation_paths, image_name)
    plot_segmentation(image, gt_segmentation, predicted_segm, save_path, image_name)


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


def perform_inference(config_path, results_path, predictions_path, images_path, segmentations_path, checkpoints_path, plot_segmentations=False):

    os.makedirs(results_path, exist_ok=True)
    json_file = os.path.join(results_path, f'validation_segmentation_metrics.json')
   
    config = load_config(config_path)
    prepare_env(config)
    data_loader = build_data_loader(config)
    model = build_model(config, checkpoints_path)
    
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for data in data_loader:
        with torch.no_grad():
            predicted_segm = model(return_loss=False, **data)

        image_meta = data['img_metas'][0].data[0][0]
        image_name = image_meta['ori_filename'][:-4]
        
        if plot_segmentations:
            compare_segmentations(image_name, predicted_segm[0], images_path, segmentations_path, predictions_path)

        results.extend(predicted_segm)

        batch_size = len(predicted_segm)
        for _ in range(batch_size):
            prog_bar.update()

    eval_kwargs = {}
    eval_kwargs.update(metric='mIoU')
    metric = dataset.evaluate(results, **eval_kwargs)
    metric_dict = dict(config=config_path, metric=metric)
    mmcv.dump(metric_dict, json_file, indent=4)


if __name__ == '__main__':
    perform_inference('config.py', './', './', 'fashionpedia/images/val', 'fashionpedia/segmentations/val', './results_3/latest.pth', plot_segmentations=False)
