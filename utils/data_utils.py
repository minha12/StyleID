"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import numpy as np
from PIL import Image
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    results = [path for path in images if 'ipynb_checkpoints' not in path]
    return results

LATENT_EXTENSIONS = ['.npy', '.pt']

def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in LATENT_EXTENSIONS)


def make_latentset(dir):
    latent_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npy_file(fname):
                path = os.path.join(root, fname)
                latent_paths.append(path)
    results = [path for path in latent_paths if 'ipynb_checkpoints' not in path]
    return results

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def extract_and_save(path, root_dir, data, n_levels = 2):
    assert n_levels == 2, 'Currently, only support 2 levels tree'
    if n_levels == 2:
        first_layer = path.split('/')[-3]
        second_layer = path.split('/')[-2]
        filename = path.split('/')[-1].split('.')[0]
        extension = path.split('/')[-1].split('.')[1]
        #make dir if it not exist
        new_path = os.path.join(root_dir, str(first_layer), str(second_layer))

        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = os.path.join(new_path, filename)
        if type(data) == np.ndarray or type(data) == list or type(data) == torch.Tensor or type(data) == np.float32:
            np.save(save_path, data)
        if type(data) == Image.Image:
            Image.fromarray(np.array(data)).save(f'{save_path}.jpg')

