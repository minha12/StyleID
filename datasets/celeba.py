import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

class CelebA(Dataset):
  def __init__(self, data_path=None, label_path=None):
    self.data_path = data_path
    self.label_path = label_path

    # Data transforms
    self.transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
  def __len__(self):
    return len(self.data_path)
  
  def __getitem__(self, idx):
    image_set = Image.open(self.data_path[idx])
    image_tensor = self.transform(image_set)
    image_label = torch.Tensor(self.label_path[idx])

    return image_tensor, image_label

def sampling_idUnique_celebAHQ(celebaHq_to_celeba_mapping = './notebooks/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', 
                               identity_celeba = './notebooks/CelebAMask-HQ/identity_CelebA.txt',
                               n_samples = 1000
                              ):
    df_mapping = pd.read_csv(celebaHq_to_celeba_mapping, delim_whitespace=True)
    df_id_map = pd.read_csv(identity_celeba, delim_whitespace=True, header=None,names=["orig_file", "identity"])
    df_mapped_id = df_mapping.merge(df_id_map, how="inner", on=["orig_file"] )
    df_cleaned = df_mapped_id.drop_duplicates(subset=['identity'])

    df_AB = df_cleaned.sample(2*n_samples)
    df_A = df_AB.iloc[:n_samples]
    df_B = df_AB.iloc[n_samples:]
    return df_A, df_B

