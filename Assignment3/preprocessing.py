import os
from DataReader.read import DataReader
from utils.frame_preprocessing_utils import *
from utils.load_save_utils import *
import time
from PIL import Image



def preprocess_video_frames(reader, samples_path, sample_name, images_path, npys_path):
    sample_path = os.path.join(samples_path, sample_name)

    frames_path = os.path.join(sample_path, 'frames')
    frames_names = os.listdir(frames_path)

    for (frame_num, frame_name) in enumerate(frames_names):
        frame = load_png_image(frames_path, frame_name[:-4])
        depth, mask = get_frame_depth_mask(reader, sample_name, frame_num)

        if depth is None and mask is None:
            # The mask is full of 0. There is no subject in the frame at all
            continue

        crop_coords = get_crop_coordinates(mask)

        if crop_coords is None:
            # Skip frame because subject is out of the scene
            continue

        frame, depth, mask = crop_frame(frame, depth, mask, crop_coords)

        img = Image.fromarray(frame)
        img.save(os.path.join(images_path, f'{sample_name}_{frame_num}.jpg'))

        save_depth_mask(depth, mask, sample_name, frame_num, npys_path)


def preprocess_all_video_frames(data_path, images_path, npys_path):
    data_folder_names = os.listdir(data_path)
    reader = DataReader(data_path)

    start_time = time.time()

    for data_folder_name in data_folder_names:
        preprocess_video_frames(reader, data_path, data_folder_name, images_path, npys_path)

    print(f'Time: {time.time() - start_time} seconds')