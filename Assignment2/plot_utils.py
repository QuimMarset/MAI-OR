import cv2
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os
import glob
from colors_labels import *
import seaborn as sns
from load_utils import load_json_log, compute_class_frequencies


# ==============================================
# Functions to plot the inferenced segmentations 
# ==============================================


def colour_segmentation(segmentation):
    colored_segmentation = colors[segmentation].astype('uint8')
    return colored_segmentation


def overlap_segmentation(image, segmentation):
    image_copy = image.copy()
    colored_segmentation = colour_segmentation(segmentation)
    
    overlapped_objects = cv2.addWeighted(image[segmentation != 0], 0.15, 
        colored_segmentation[segmentation != 0], 0.85, 0)

    image_copy[segmentation != 0] = overlapped_objects
    return image_copy


def create_segmentation_legend(ground_truth, predicted):
    unique_ground = np.unique(ground_truth)
    unique_predicted = np.unique(predicted)
    unique_values = set(unique_ground).union(unique_predicted)
    unique_values.remove(0)

    patches = []

    for unique_value in unique_values:
        patch = Patch(facecolor=colors[unique_value]/255.0, label=labels[unique_value], edgecolor='black')
        patches.append(patch)

    return patches


def plot_segmentation(image, ground_truth, predicted_segm, path, image_name):
    fig, axs = plt.subplots(1, 3, figsize=(12, 7))

    for ax in axs:
        ax.axis('off')

    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    
    overlapped_ground = overlap_segmentation(image, ground_truth)
    axs[1].imshow(overlapped_ground)
    axs[1].set_title('Overlapped Ground-truth')

    overlapped_predicted = overlap_segmentation(image, predicted_segm)
    axs[2].imshow(overlapped_predicted)
    axs[2].set_title('Overlapped Predicted')

    legend = create_segmentation_legend(ground_truth, predicted_segm)
    fig.legend(handles=legend, loc='lower center', ncol=8)

    fig.tight_layout()
    plt.savefig(os.path.join(path, f'segmented_{image_name}.jpg'))
    plt.close()


# ===============================================================================
# Functions to plot the validation metrics of the different experiments
# ===============================================================================


def plot_val_segmentation_metric_experiments(log_dicts, save_path, metric_name='mIoU'):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for (i, log_dict) in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        mIoU_values = []
        for epoch in epochs:
            mIoU_values.append(log_dict[epoch][metric_name][0])

        if i == 0:
            epochs_range = range(1, epochs[-1]+1)
        
        plt.plot(epochs_range, mIoU_values, label=f'Experiment {i+1}', marker='o')
    
    plt.xticks(epochs_range)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Experiments validation {metric_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'val_mIoU_experiments.png'))
    plt.close()


def plot_val_segmentation_metrics_best_experiment(json_log_path, save_path):
    sns.set(style="whitegrid")
    file_path = glob.glob(os.path.join(json_log_path, '*.log.json'))[0]
    log_dict = load_json_log(file_path)
    plt.figure(figsize=(10, 6))

    epochs = list(log_dict.keys())
    epochs_range = range(1, epochs[-1]+1)

    for metric in ['mIoU', 'mDice']:
        metric_values = []
        for epoch in epochs:
            metric_values.append(log_dict[epoch][metric][0])

        plt.plot(epochs_range, metric_values, label=f'{metric}', marker='o')
    
    plt.xticks(epochs_range)
    plt.xlabel('Epoch')
    plt.ylabel('Validation segmentation metric')
    plt.legend()
    plt.title(f'Best model validation segmentation metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'val_metrics_best_model.png'))
    plt.close()


def plot_experiments_metrics(experiments_path, save_path):
    json_log_paths = glob.glob(os.path.join(experiments_path, '**/*.log.json'))
    log_dicts = [load_json_log(json_log_path) for json_log_path in json_log_paths]
    plot_val_segmentation_metric_experiments(log_dicts, save_path)


# ================================================
# Functions to plot the dataset classes statistics
# ================================================


def plot_classes_histogram(json_file_path, save_path, is_train=True):
    sns.set(style="whitegrid")
    frequencies = compute_class_frequencies(json_file_path)
    num_classes = frequencies.shape[0]
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), frequencies, label='Class proportion', log=True)
    plt.xticks([i for i in range(num_classes)], labels[1:], rotation='vertical')
    plt.ylabel('Class proportion')
    plt.legend()
    plt.title(f'Fashionpedia class frequency in the training set')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_class_frequencies.png'))
    plt.close()


if __name__ == '__main__':

    root_path = './'
    results_path = os.path.join(root_path, 'results')

    #plot_experiments_metrics(results_path, root_path)
    plot_val_segmentation_metrics_best_experiment(os.path.join(results_path, 'best_model'), root_path)
