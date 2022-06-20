from model import DepthEstimationModel
import os
from tensorflow import keras
from utils.load_save_utils import *
from utils.plot_utils import plot_depth_estimation



def load_model(weights_path):
    model = DepthEstimationModel()
    model.load_weights(os.path.join(weights_path, 'model_weights')).expect_partial()
    return model


def perform_inference(weights_path, images_path, npys_path, test_files, image_size, predictions_path):
    model = load_model(weights_path)

    for test_file in test_files[:1000]:
        frame, depth, mask = load_sample_from_files(images_path, npys_path, test_file, (image_size, image_size))
        
        frame_mask = np.concatenate([frame, mask[:, :, np.newaxis]], axis=-1)
        frame_mask = np.expand_dims(frame_mask, axis=0)

        prediction = model.predict(frame_mask)
        plot_depth_estimation(frame, depth, prediction.squeeze(), predictions_path, test_file)