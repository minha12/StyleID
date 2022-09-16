import torch
from torch import nn

from models.facial_recognition.model_irse import Backbone
l2_criterion = torch.nn.MSELoss(reduction='mean')
facenet_model = './pretrained_models/model_ir_se50.pth'

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

class IDLoss(nn.Module):
    def __init__(self, opts):
    #def __init__(self, facenet_model = './pretrained_models/model_ir_se50.pth'):    
        super(IDLoss, self).__init__()
        #print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count

#     def forward(self, y_hat, y,l2_loss=True):
#         n_samples = y.shape[0]
#         y_feats = self.extract_feats(y)  # Otherwise use the feature from there
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         sim_improvement = 0
#         if l2_loss:
#             #loss = 1.25 - torch.norm(y_feats - y_hat_feats, 2)
#             loss = 0
#             count = 0
#             for i in range(n_samples):
#                 diff_target = 1.25 - torch.norm(y_feats[i] - y_hat_feats[i], 2)
#                 loss += diff_target
#                 count += 1
#             loss = loss / count
#             sim_improvement = sim_improvement / count
#         else:
#             loss = 0
#             count = 0
#             for i in range(n_samples):
#                 diff_target = y_hat_feats[i].dot(y_feats[i])-0.35 #cosine similarity
#                 loss += diff_target
#                 count += 1
#             loss = loss / count
#             sim_improvement = sim_improvement / count

#         return loss, sim_improvement

class IDLoss_t(nn.Module):
    def __init__(self, opts):
    #def __init__(self, facenet_model = './pretrained_models/model_ir_se50.pth'):    
        super(IDLoss_t, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x_hat, x, y=None, l2_loss=False):
        n_samples = x.shape[0]
        if not y==None:
            y_feats = self.extract_feats(y)
            y_feats = y_feats.detach()
        x_feats = self.extract_feats(x)  # Otherwise use the feature from there
        x_hat_feats = self.extract_feats(x_hat)
        #x_feats = x_feats.detach()
        
        dist2_src = 0
        if l2_loss:
            #loss = 1.25 - torch.norm(y_feats - y_hat_feats, 2)
            loss = 0
            count = 0
            for i in range(n_samples):
                if not y==None:
                    diff_target = torch.norm(x_hat_feats[i] - y_feats[i], 2)
                    diff_src    = torch.norm(x_hat_feats[i] - x_feats[i], 2)
                    sim_improvement += diff_src
                    loss += diff_target
                    count += 1
                    loss = loss / count
                    sim_improvement = sim_improvement / count
                else:
                    loss = 0
                    sim_improvement = 0
        else:
            loss = 0
            count = 0
            for i in range(n_samples):
                #print("check!") 
                #if not y==None:
                #diff_target = x_hat_feats[i].dot(y_feats[i]) #cosine similarity
                #diff_src    = torch.norm(x_hat_feats[i] - x_feats[i], 2)
                diff_src    = 1 - x_hat_feats[i].dot(x_feats[i]) - 0.3 #similarity to the src
                #dist2_src += 1 - diff_src
                loss += diff_src
                count += 1
            loss = loss / count
            #dist2_src = dist2_src / count

        return loss