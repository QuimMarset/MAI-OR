import os
import numpy as np
from DataReader.util import intrinsic, extrinsic
from DataReader.depth_render import Render
# from utils.load_save_utils import load_frame, save_sample_data


def get_metadata_info(info):

    human_metadata = {
        'num_frames' : info['poses'].shape[1],
        'gender' : info['gender']
    }

    garment_type = list(info['outfit'])[0]
    texture_type = info['outfit'][garment_type]['texture']['type']
    texture_colour = None
    if texture_type == 'color':
        texture_colour = info['outfit'][garment_type]['texture']['data']

    garment_metadata = {
        'type' : garment_type,
        'fabric_type' : info['outfit'][garment_type]['fabric'],
        'texture_type' : texture_type,
        'texture_colour' : texture_colour,
    }

    return human_metadata, garment_metadata


# Display utils used on this notebook require triangulated faces
def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)


def get_3D_mesh_data(reader, sample_name, frame_num):
    # Human mesh vertices and faces
    vertices, faces = reader.read_human(sample_name, frame_num)
    faces = np.array(faces)

    # Garments vertices and faces
    info = reader.read_info(sample_name)
    garment_names = list(info['outfit'].keys())

    for garment_name in garment_names:
        garment_i_vertices = reader.read_garment_vertices(sample_name, garment_name, frame_num)
        garment_i_faces = reader.read_garment_topology(sample_name, garment_name)
        garment_i_faces = quads2tris(garment_i_faces)

        # Merge human and garment meshes into one
        faces = np.concatenate((faces, garment_i_faces + vertices.shape[0]), axis=0)
        vertices = np.concatenate((vertices, garment_i_vertices), axis=0)

    return vertices, faces


def compute_mesh_depth_and_mask(reader, sample_name, vertices, faces, max_depth=10):
    info = reader.read_info(sample_name)

    render = Render(max_depth=max_depth)
    render.set_mesh(vertices, faces)
    render.set_image(640, 480, intrinsic(), extrinsic(info['camLoc']))
    depth = render.render().squeeze()

    thresh = max_depth - 1
    mask = depth < thresh

    if np.all(mask == False):
        return None, None

    dmin = depth.min()
    # Change depth to mm (currently in meters)
    depth[mask] = (depth[mask] - dmin) * 1000 + 1
    depth[~mask] = 0.0

    mask = mask.astype(np.uint8)

    return depth, mask


def get_frame_depth_mask(reader, sample_name, frame_num):
    vertices, faces = get_3D_mesh_data(reader, sample_name, frame_num)
    depth, mask = compute_mesh_depth_and_mask(reader, sample_name, vertices, faces)
    return depth, mask


def check_subject_out_the_scene(coords, mask_shape):
    height, width = mask_shape
    x_min, y_min, x_max, y_max = coords

    x_condition = x_min < 0 or x_max >= width
    y_condition = y_min < 0 or y_max >= height

    return x_condition or y_condition
    

def get_crop_coordinates(mask, border=10):
    indices = np.where(mask == 1.0)

    x_min = min(indices[1])
    y_min = indices[0][0]
    x_max = max(indices[1])
    y_max = indices[0][-1]

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_dist = x_max - x_min + 1
    y_dist = y_max - y_min + 1
    max_dist = max(x_dist, y_dist) // 2
    
    x_min = x_center - max_dist - border
    y_min = y_center - max_dist - border
    x_max = x_center + max_dist + border
    y_max = y_center + max_dist + border
    coords = [x_min, y_min, x_max, y_max]

    if check_subject_out_the_scene(coords, mask.shape):
        return None

    return coords


def crop_frame(frame, depth, mask, crop_coords):
    x_min, y_min, x_max, y_max = crop_coords
    frame = frame[y_min:y_max+1, x_min:x_max+1]
    depth = depth[y_min:y_max+1, x_min:x_max+1]
    mask = mask[y_min:y_max+1, x_min:x_max+1]
    return frame, depth, mask


