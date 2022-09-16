from models.stylegan2.model import Generator
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from criteria import id_loss, w_norm
from models.encoders.model_irse import Backbone
from models.encoders.model_irse import IR_101
from configs.paths_config import model_paths, dataset_paths
from models.mobilenet import MobileNet
import torchvision.transforms as transforms
from unet import unet
from utils_seg import generate_label_plain, generate_label
from datasets import augmentations
from PIL import Image
from PIL import ImageFilter
from models.encoders import psp_encoders
from argparse import Namespace


class StyleID:
    def __init__(self, 
                 checkpoint = model_paths['stylegan_ffhq']):
        self.config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        self.pre_trained_path = checkpoint
        self.gen = Generator(
            size=1024,
            style_dim = self.config['latent'],
            n_mlp= self.config['n_mlp'],
            #channel_multiplier = self.config["channel_multiplier"]
        )
        
        #id model
        self.facenet = IR_101(input_size=112)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        #attr model
        self.attrnet = MobileNet()
        self.attr_list = np.array(open(dataset_paths['celeba_attr']).readlines()[1].split())
        #segmentation model
        self.unet = unet()
        self.seg_size = 512
        self.set_encoder()
        self.load_weights()
        #segmentation mask encoder
    
    def set_encoder(self):
        opts = torch.load(model_paths['stylegan_seg'], map_location='cpu')['opts']
        opts['checkpoint_path'] = model_paths['stylegan_seg']
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        self.opts = Namespace(**opts)
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        self.encoder = encoder
        
    
    def load_weights(self):
        #load stylegan2
        self.gen.load_state_dict(torch.load(self.pre_trained_path)['g_ema'])
        self.gen.eval()
        self.gen.cuda()
        #load facenet
        self.facenet.load_state_dict(torch.load(model_paths['circular_face']))
        self.facenet.eval()
        self.facenet.cuda()
        #load attrnet
        self.attrnet.load_state_dict(torch.load(model_paths['mobilenet'])['model_state_dict'])
        self.attrnet.eval()
        self.attrnet.cuda()
        #load unet
        self.unet.load_state_dict(torch.load(model_paths['unet']))
        self.unet.eval()
        self.unet.cuda()
        #seg encoder
        ckpt = torch.load(model_paths['stylegan_seg'],map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.encoder.eval()
        self.encoder.cuda()
        self.__load_latent_avg(ckpt)
    
    def image_from_w(self, w_latent):
        output_image, _ = self.gen([w_latent.cuda()], input_is_latent=True)
        return output_image
    
    def image_from_s(self, s_latent, return_latents=False):
        if return_latents:
            print('check!')
            return self.gen([s_latent], input_is_stylespace=True,w_latent_only=True)
        else:
            img, _ = self.gen([s_latent], input_is_stylespace=True,return_latents=return_latents)
            return img
    
    def image_from_z(self, z_latent, return_latents=False):
        if return_latents:
            image, latent, style_vector = self.gen([z_latent], return_latents=return_latents)
            return image, latent, style_vector
        else:
            image,_ = self.gen([z_latent], return_latents=return_latents)
            return image
        
    def get_w(self, z_latent):
        return self.gen([z_latent], w_latent_only=True)
    
    def get_s(self, z_latent):
        _, _, style_vector = self.gen([z_latent], return_latents=True)
        return style_vector
        
    
    def extract_id_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def id_dist(self, x, y):
        n_samples = x.shape[0]
        x_feats = self.extract_id_feats(x)
        y_feats = self.extract_id_feats(y)  # Otherwise use the feature from there
        x_feats = x_feats.detach()
        y_feats = y_feats.detach()
        
        #dists = []
        #for i in range(n_samples):
        #diffs = torch.cdist(x_feats,y_feats).cpu().numpy() #y vs x
        #diff = np.linalg.norm(x_feats-y_feats)
        #dists += [diff]
        diffs = F.pairwise_distance(x_feats, y_feats).cpu().numpy()

        return np.array(diffs)
    
    def extract_attr(self,x):
        if x.shape[2] != 256:
            x = transforms.Resize((256, 256))(x)
        attr_x = self.attrnet(x).detach()
        return attr_x
    
    def attr_dist(self, x, y, logit_convert=False):
        attr_x = self.extract_attr(x)
        attr_y = self.extract_attr(y)
        if logit_convert:
            attr_dist = float(torch.dist( torch.logit(attr_x), torch.logit(attr_y) ))
            return attr_dist
        return float(torch.dist( attr_x, attr_y ))
    
    def get_predicted_attr(self,x):
        attr_x = self.extract_attr(x)
        res = []
        for item in attr_x:
            fil = item.cpu().numpy() > 0.5
            res.append(self.attr_list[fil])
        return res
    
    def get_attr_score(self, x, labels):
        indexes = [self.attr_list.tolist().index(label)
                   for label in labels ]
        attr_x = self.extract_attr(x)
        return attr_x[:, indexes]
    
    def get_mask(self, imgs):
        imgs = transforms.Resize((self.seg_size, self.seg_size))(imgs)
        labels_predict = self.unet(imgs)
        labels_predict_plain = generate_label_plain(labels_predict,self.seg_size)
        labels_predict_color = generate_label(labels_predict, self.seg_size)
        return labels_predict_plain, labels_predict_color
    
    def w_from_mask(self,mask):
        mask = [seg_transform(item.astype('int32')) for item in mask]
        mask = torch.stack(mask).float().cuda() 
        codes = self.encoder(mask)
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes
    
    def rand_face_from_mask(self, mask, return_latents=False, latent_mask = [5,6,7,8,9], return_rand_w = False):
        codes = self.w_from_mask(mask)
        z_latent = torch.randn(codes.shape[0], 512)
        rand_w = self.get_w(z_latent.cuda())
        w_latent = swap_w_layers(codes.cuda(), rand_w, latent_mask) #taking layers in latent_mask from rand_w
        if return_rand_w:
            return self.image_from_w(w_latent), w_latent, rand_w
        if return_latents:
            return self.image_from_w(w_latent), w_latent
        return self.image_from_w(w_latent)
    
    def face_from_attr_mask(self, conditions, mask, latent_mask=[5,6,7,8,9,10,11]):
        z_latent = torch.randn(mask.shape[0],512)
        _, w_condition, _ = self.conditioned_sampling(z_latent, conditions, 
                                                      shared_condition=False,
                                                      return_latent=True)
        w_mask = self.w_from_mask(mask)
        w_res = swap_w_layers(w_mask, w_condition,latent_mask)
        return self.image_from_w(w_res)
        
    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

class Encoder():
    def __init__(self, 
                 checkpoint = model_paths['stylegan_seg']):
        self.encoder_path = checkpoint
        self.encoder = self.set_encoder()
        self.load_weights()
        self.resize_dims = (256, 256)
        self.transform = transforms.Compose([transforms.Resize(self.resize_dims),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        #segmentation mask encoder
    
    def set_encoder(self):
        opts = torch.load(self.encoder_path, map_location='cpu')['opts']
        opts['checkpoint_path'] = self.encoder_path
        self.opts = Namespace(**opts)

        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder
        
    def load_weights(self):
        #seg encoder
        ckpt = torch.load(self.encoder_path,map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.encoder.eval()
        self.encoder.cuda()
        self.__load_latent_avg(ckpt)
        
    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
            
    def get_w(self,imgs):
        imgs = torch.stack([self.transform(img) for img in imgs]).cuda()

        codes = self.encoder(imgs)

        if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes
    
class Decoder():
    def __init__(self, 
                 checkpoint = model_paths['stylegan_ffhq']):
        self.config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
        self.pre_trained_path = checkpoint
        self.gen = Generator(
            size=1024,
            style_dim = self.config['latent'],
            n_mlp= self.config['n_mlp'],
            channel_multiplier = self.config["channel_multiplier"]
        )
        self.load_weights()
    
    def load_weights(self):
        self.gen.load_state_dict(torch.load(self.pre_trained_path)['g_ema'])
        self.gen.eval()
        self.gen.cuda()
        
    def image_from_w(self, w_latent):
        output_image, _ = self.gen([w_latent.cuda()],randomize_noise=True, input_is_latent=True)
        return output_image
    
    def image_from_s(self, s_latent):
        output_image, _ = self.gen([s_latent], input_is_stylespace=True)
        return output_image
    
    def image_from_z(self, z_latent, return_latents=False):
        if return_latents:
            image, latent, style_vector = self.gen([z_latent], return_latents=return_latents)
            return image, latent, style_vector
        else:
            image,_ = self.gen([z_latent], return_latents=return_latents)
            return image
        
    def get_w(self, z_latent):
        return self.gen([z_latent], w_latent_only=True)
    
    def get_s(self, z_latent):
        _, _, style_vector = self.gen([z_latent], return_latents=True)
        return style_vector
    
# utils


def swap_w_layers(x, y, layers):
    '''
    x: [n_batch, 18, 512]
    y: [n_batch, 18, 512]
    layers in [0...17]
    '''
    temp_x = x.detach().clone()
    temp_y = y.detach().clone()
    temp_x[:, layers, :] = temp_y[:, layers, :]

    return temp_x

def detach_to_cpu(x):
    return x.detach().cpu().numpy()

def numpy_to_gpu(x):
    return torch.tensor(x).cuda()


def print_fig(output, name, size=128):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    im = Image.fromarray(output).resize((size,size), Image.ANTIALIAS)
    print(im)
    
def tensor_to_rgb(x):
    output = (x + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    outputs = [item.detach().cpu().permute(1,2,0).numpy() for item in output]
    outputs = [(item*255).astype(np.uint8) for item in outputs]
    return outputs

def visual(output):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    outputs = [item.detach().cpu().permute(1,2,0).numpy() for item in output]
    outputs = [(item*255).astype(np.uint8) for item in outputs]
    f = plt.figure()
    n = len(outputs)
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(outputs[i])
        plt.axis('off')
    plt.show(block=True)
    

# for segmentation swap
seg_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                augmentations.ToOneHot(n_classes=19),
                transforms.ToTensor()
])


def facial_mask(transformed_mask):
    mask = ((
        transformed_mask[1]
        +transformed_mask[2]
        +transformed_mask[4]
        +transformed_mask[5]
        +transformed_mask[6]
        +transformed_mask[7]
        +transformed_mask[10]
        +transformed_mask[11]
        +transformed_mask[12]
    ) >= 1)*1.0

    mask = Image.fromarray((mask.numpy()*255).astype('uint8'))

    #mask = mask.filter(ImageFilter.GaussianBlur(2))
    mask = mask.point(lambda p: p > 125 and 255)  
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    mask = mask.point(lambda p: p > 125 and 255) 
    
    return mask

def area_mask(transformed_mask):
    mask = ((
        #transformed_mask[1]
        transformed_mask[2]
        +transformed_mask[4]
        +transformed_mask[5]
        +transformed_mask[6]
        +transformed_mask[7]
        +transformed_mask[10]
        +transformed_mask[11]
        +transformed_mask[12]
        #+transformed_mask[13]
    ) >= 1)*1.0

    mask = Image.fromarray((mask.numpy()*255).astype('uint8'))

    mask = mask.filter(ImageFilter.GaussianBlur(2))
    mask = mask.point(lambda p: p > 125 and 255)  
    mask = mask.filter(ImageFilter.GaussianBlur(5))
    mask = mask.point(lambda p: p > 30 and 255) 
    
    return mask

def matched_swap(ref_img, img, mask, full_swap=False):
    mask = seg_transform(mask.astype('int32'))
    mask_full = facial_mask(mask)
    if full_swap:
        mask = mask_full
    else:
        mask = area_mask(mask)
    if ref_img.size != (256,256):#background
        ref_img = ref_img.resize((256, 256))
    if img.size != (256, 256):
        img = img.resize((256, 256))
    from skimage.exposure import match_histograms
    ref_face = Image.composite(ref_img, 
                               Image.new('RGB', (256, 256)), mask_full)#background
    face = Image.composite(img, 
                           Image.new('RGB', (256, 256)), mask_full)
    matched = match_histograms(np.asarray(face), 
                               np.asarray(ref_face), 
                               multichannel=True)
    img = Image.composite(Image.fromarray(matched), img, mask_full)

    mask = mask.filter(ImageFilter.GaussianBlur(2))

    return Image.composite(img, ref_img, mask)
    
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt