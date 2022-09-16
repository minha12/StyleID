import torch
from torch import nn

from models.mobilefacenet import MobileFaceNet
from torchvision import transforms

l2_criterion = torch.nn.MSELoss(reduction='mean')


class LMLoss(nn.Module):
    def __init__(self, opts):
        super(LMLoss, self).__init__()
        print('Loading Landmark Encoder')
        self.landmark_model = MobileFaceNet([112, 112], 136)
        checkpoint = torch.load(opts.mobilefacenet_weights)
        self.landmark_model.load_state_dict(checkpoint['state_dict'])
        self.landmark_model.eval()
        self.landmark_model.cuda()
        self.resize = transforms.Resize(112)
        self.opts = opts
    
    def preprocess(self, imgs):
        return self.resize(imgs)
    
    def extract_feats(self, imgs):
        """
        without preprocess (face crop)
        :param imgs: shape: torch.Size([batch size, 3, 112, 112])
        :return:
        outputs - model results scaled to img size, shape: torch.Size([batch size, 136])
        landmarks - reshaped outputs + no jawline, shape: torch.Size([batch size, 51, 2])
        """
        resized_images = self.preprocess(imgs)
        outputs, _ = self.landmark_model(resized_images)

        batch_size = resized_images.shape[0]
        landmarks = torch.reshape(outputs*112, (batch_size, 68, 2))

        return landmarks[:, 17:, :]
    
    def landmark_loss(self, input_attr_lnd, output_lnd):
        loss = l2_criterion(input_attr_lnd, output_lnd)
        return loss

    def forward(self, y_hat, y):
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = self.landmark_loss(y_hat_feats, y_feats)
        return loss 

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
