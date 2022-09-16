import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

LATENT_EXTENSIONS = ['.npy']

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

transform = transforms.ToTensor()

transfrom_ffhq = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

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

class LatentsDataset(Dataset):

    def __init__(self, latents, opts):
        self.latents = latents
        self.opts = opts

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):

        return self.latents[index]

class LatentsOnlyDatasets(Dataset):

    def __init__(self, src_latents, tar_latents, opts):
        self.src_latents = src_latents
        self.tar_latents = tar_latents
        self.opts = opts
        assert len(src_latents) == len(tar_latents)

    def __len__(self):
        return self.src_latents.shape[0]

    def __getitem__(self, index):
        return self.src_latents[index], self.tar_latents[index]

class LatentsDatasets(Dataset):

    def __init__(self, src_latents, tar_latents, opts, mask_root = None):
        self.src_latents = src_latents
        self.tar_latents = tar_latents
        self.mask_root = mask_root
        if not self.mask_root==None:
            self.mask_paths = sorted(make_dataset(mask_root))
        self.opts = opts
        assert len(src_latents) == len(tar_latents)

    def __len__(self):
        return self.src_latents.shape[0]

    def __getitem__(self, index):
        if self.mask_root==None:
            return self.src_latents[index], self.tar_latents[index]
        else:
            mask_path = self.mask_paths[index]
            mask_im = Image.open(mask_path)
            mask_im = transform(mask_im)
            return self.src_latents[index], self.tar_latents[index], mask_im

class EmbedImagePairs(Dataset):

    def __init__(self, img_dir, embed_dir, opts):
        self.img_dir = img_dir
        self.embed_dir = embed_dir
        
        self.img_paths = sorted(make_dataset(img_dir))
        self.embed_paths = sorted(make_latentset(embed_dir))
        self.opts = opts
        print('Total images: ', len(self.img_paths))
        print('Total embeds: ', len(self.embed_paths))
        assert len(self.img_paths) == len(self.embed_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        im = Image.open(img_path)
        im = transfrom_ffhq(im)
        embed = np.float32(np.load(self.embed_paths[index]))
        
        return im, embed

class EmbedLatentImages(Dataset):

    def __init__(self, img_dir, embed_dir, latent_dir, opts):
        self.img_dir = img_dir
        self.embed_dir = embed_dir
        self.latent_dir = latent_dir
        
        self.img_paths = sorted(make_dataset(img_dir))
        self.embed_paths = sorted(make_latentset(embed_dir))
        self.latent_paths = sorted(make_latentset(latent_dir))
        self.opts = opts
        print('Total images: ', len(self.img_paths))
        print('Total embeds: ', len(self.embed_paths))
        print('Total latents: ', len(self.latent_paths))
        assert len(self.img_paths) == len(self.embed_paths) 
        assert len(self.embed_paths)== len(self.latent_paths)
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        im = Image.open(img_path)
        im = transfrom_ffhq(im)
        embed = np.float32(np.load(self.embed_paths[index]))
        latent = np.load(self.latent_paths[index])
        
        return im, embed, latent

class EmbedDataset(Dataset):

    def __init__(self, embed_dir, opts=None):
        self.embed_dir = embed_dir
        self.embed_paths = sorted(make_latentset(embed_dir))
        self.opts = opts
        print('Total embeds: ', len(self.embed_paths))


    def __len__(self):
        return len(self.embed_paths)

    def __getitem__(self, index):
        embed = np.float32(np.load(self.embed_paths[index]))
        
        return embed

class EmbedNamesDataset(Dataset):

    def __init__(self, embed_names, root, opts=None):
        #self.embed_dir = embed_dir
        self.embed_paths = sorted([os.path.join(root, f'{f}.npy') for f in embed_names])
        self.opts = opts
        print('Total embeds: ', len(self.embed_paths))


    def __len__(self):
        return len(self.embed_paths)

    def __getitem__(self, index):
        embed = np.float32(np.load(self.embed_paths[index]))
        
        return embed

class LatentsMappedDatasets(Dataset):

    def __init__(self, src_root, tar_root, truth_root, opts):
        self.src_paths = sorted(make_latentset(src_root))
        self.tar_paths = sorted(make_latentset(tar_root))
        self.truth_paths = sorted(make_latentset(truth_root))
        #self.mask_root = mask_root
        self.opts = opts
        assert len(self.src_paths) == len(self.tar_paths)

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, index):
        src_latent = np.load(self.src_paths[index])
        tar_latent = np.load(self.tar_paths[index])
        truth_latent = np.load(self.truth_paths[index])
        return src_latent,tar_latent, truth_latent
    
class StyleSpaceLatentsDataset(Dataset):

    def __init__(self, latents, opts):
        padded_latents = []
        for latent in latents:
            latent = latent.cpu()
            if latent.shape[2] == 512:
                padded_latents.append(latent)
            else:
                padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
                padded_latent = torch.cat([latent, padding], dim=2)
                padded_latents.append(padded_latent)
        self.latents = torch.cat(padded_latents, dim=2)
        self.opts = opts

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index):
        return self.latents[index]


