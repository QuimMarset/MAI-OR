import cv2
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os
from colors_labels import *
import seaborn as sns
from load_utils import load_json_log


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


def plot_val_segm_metrics_curve(log_dict, save_path):
    sns.set_style('dark')
    epochs = list(log_dict.keys())
    plt.figure(figsize=(8, 6))

    for metric in ['mIoU', 'mAcc', 'aAcc']:
        plot_epochs = []
        plot_values = []
        
        for epoch in epochs:
            epoch_logs = log_dict[epoch]
            
            plot_epochs.append(epoch)
            plot_values.append(epoch_logs[metric][0])
        
        plt.plot(plot_epochs, plot_values, label=metric, marker='o') 
    
    plt.xticks(plot_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Segmentation metric')
    plt.legend()
    plt.title('Validation segmentation metrics')
    plt.savefig(os.path.join(save_path, 'val_segmentation_metrics.jpg'))
    plt.close()


def plot_train_loss_curve(log_dict, save_path):
    sns.set_style('dark')
    plt.figure(figsize=(8, 6))
    plot_iters = []
    plot_values = []
    pre_iter = -1

    epochs = list(log_dict.keys())
    for epoch in epochs:
        epoch_logs = log_dict[epoch]

        for idx in range(len(epoch_logs['loss'])):
            if pre_iter > epoch_logs['iter'][idx]:
                # avoid number of validation iterations
                continue
            pre_iter = epoch_logs['iter'][idx]

            plot_iters.append(epoch_logs['iter'][idx])
            plot_values.append(epoch_logs['loss'][idx])

    plt.xlabel('Iteration')
    plt.ylabel('Categorical cropss-entropy loss')
    plt.plot(plot_iters, plot_values, label='Training loss')
    plt.legend()
    plt.title('Training loss')
    plt.savefig(os.path.join(save_path, 'train_loss_curve.jpg'))
    plt.close()


def plot_metrics(json_log_path, save_path):
    log_dict = load_json_log(json_log_path)
    plot_train_loss_curve(log_dict, save_path)
    plot_val_segm_metrics_curve(log_dict, save_path)


if __name__ == '__main__':
    plot_metrics('./results_3/20220409_094929.log.json', './')
