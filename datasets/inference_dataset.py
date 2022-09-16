from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np


class InferenceDataset(Dataset):

    def __init__(self, root, opts=None, transform=None):
        self.paths = np.array(sorted(data_utils.make_dataset(root)))
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
            from_im = from_im.convert('RGB')
            from_im = self.transform(from_im)
        return from_im

from models.mtcnn.mtcnn import MTCNN

class InferenceFace(Dataset):

    def __init__(self, paths, opts=None, transform=None):
        self.paths = paths
        self.transform = transform
        self.mtcnn = MTCNN()
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        #print(from_path)
        from_im = Image.open(from_path).convert('RGB')
        from_im, _ = self.mtcnn.align(from_im) #crop img?
        #if from_im is None: 
        if self.transform:
            from_im = self.transform(from_im)
                
        return from_im

class InferenceLatent(Dataset):

    def __init__(self, root):
        self.paths = sorted(data_utils.make_latentset(root))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_latent = np.load(from_path)

        return from_latent