from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
import os

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

class MaskDataset(Dataset):

    def __init__(self, root, opts=None, transform=None):
        self.paths = sorted(make_dataset(root))
        self.transform = transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = Image.open(from_path)
        if self.opts is not None:
            from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
        if self.transform:
            from_im = self.transform(from_im)
        return from_im