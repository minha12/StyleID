import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

class MapperBase(Module):

    def __init__(self, opts, latent_dim=512):
        super(MapperBase, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x

class MapEmbed(Module):

    def __init__(self, opts, in_dim = 512, latent_dim=512):
        super(MapEmbed, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            if i == 0:
                layers.append(
                    EqualLinear(
                        in_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x

class EmbedMapper(Module):

    def __init__(self, opts):
        super(EmbedMapper, self).__init__()

        self.opts = opts

        self.mapping = MapEmbed(opts)
        self.upscale = UpScaleEmbed(opts)

    def forward(self, x):
        #print(f'x shape: {x.shape}')
        x = self.upscale(x)
        #print(f'x shape after upscale: {x.shape}')
        out = self.mapping(x)
        #print(f'out shape: {out.shape}')
        return out

class UpScaleEmbed(Module):
    def __init__(self,opts,embed_dim=512, #adaptive
                 latent_dim=512, n_layer=18):
        super(UpScaleEmbed, self).__init__()
        self.opts = opts
        self.linear = EqualLinear(embed_dim, n_layer * latent_dim, lr_mul=1)
        self.n_layer = n_layer
        self.latent_dim = latent_dim
    def forward(self,x):
        x = self.linear(x)
        return x.view(-1, self.n_layer, self.latent_dim)

class Mapper(Module):

    def __init__(self, opts, latent_dim=512):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(2):
            if i == 0:
                layers.append(
                        EqualLinear(
                        2*latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        out = self.mapping(x)
        return torch.sigmoid(out)


class SingleMapper(Module):

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out

    
class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = MapperBase(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = MapperBase(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = MapperBase(opts)

    def forward(self, x):
        x_coarse = x[:, :5, :]
        x_medium = x[:, 5:12, :]
        x_fine = x[:, 12:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

class LevelsMapperId(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = MapperBase(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = MapperBase(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = MapperBase(opts)

    def forward(self, x, lock_tar_id=True):
        if lock_tar_id:
            x_coarse = x[:, :5, :]
            x_id = x[:,5:8, :]
            x_medium = x[:, 8:12, :]
            x_fine = x[:, 12:, :]
        else:
            x_coarse = x[:, :5, :]
            #x_id = x[:,5:8, :]
            x_medium = x[:, 5:12, :]
            x_fine = x[:, 12:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)
        
        if lock_tar_id:
            x_id = torch.zeros_like(x_id)

            out = torch.cat([x_coarse, x_id, x_medium, x_fine], dim=1)
            
        else:
            out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out
    
class LevelsJoinMapper(Module):

    def __init__(self, opts):
        super(LevelsJoinMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x,y, lock_tar_id=True):
        #print(f'x shape: {x.shape}')
        if lock_tar_id:
            x_coarse = x[:, :5, :]
            x_id = x[:,5:8, :]
            x_medium = x[:, 8:12, :]
            x_fine = x[:, 12:, :]
            
            y_coarse = y[:, :5, :]
            y_id = y[:,5:8,:]
            y_medium = y[:, 8:12, :]
            y_fine = y[:, 12:, :]
        else:
            x_coarse = x[:, :5, :]
            #x_id = x[:,5:8, :]
            x_medium = x[:, 5:12, :]
            x_fine = x[:, 12:, :]
            y_coarse = y[:, :5, :]
            #y_id = y[:,5:8,:]
            y_medium = y[:, 5:12, :]
            y_fine = y[:, 12:, :]
            
        
        

        if not self.opts.no_coarse_mapper:
            out_coarse = self.course_mapping( torch.cat([x_coarse,y_coarse], dim=2) )
        else:
            out_coarse = torch.zeros_like(x_coarse)
            
        if not self.opts.no_medium_mapper:
            out_medium = self.medium_mapping(torch.cat([x_medium,y_medium], dim=2))
        else:
            out_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            out_fine = self.fine_mapping(torch.cat([x_fine,y_fine], dim=2))
        else:
            out_fine = torch.zeros_like(x_fine)

        #print(out_coarse.shape)
        #print(out_medium.shape)
        #print(out_fine.shape)
        if lock_tar_id:
            out_id = torch.ones_like(x_id)
            out = torch.cat([out_coarse, 
                             out_id, 
                             out_medium, out_fine], dim=1)
        else:
            out = torch.cat([out_coarse, 
                             #out_id, 
                             out_medium, out_fine], dim=1)
        return out

class FullStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(FullStyleSpaceMapper, self).__init__()

        self.opts = opts

        for c, c_dim in enumerate(STYLESPACE_DIMENSIONS):
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c, x_c in enumerate(x):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            out.append(x_c_res)

        return out


class WithoutToRGBStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(WithoutToRGBStyleSpaceMapper, self).__init__()

        self.opts = opts

        indices_without_torgb = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
        self.STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in indices_without_torgb]

        for c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=STYLESPACE_DIMENSIONS[c]))

    def forward(self, x):
        out = []
        for c in range(len(STYLESPACE_DIMENSIONS)):
            x_c = x[c]
            if c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            else:
                x_c_res = torch.zeros_like(x_c)
            out.append(x_c_res)

        return out