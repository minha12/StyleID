import gdown

google_drive_paths = {
    "stylegan2": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",

    "celeba_samples": "https://drive.google.com/uc?id=1VlkFMkDaT25V00rk1tuovtkqd2h6Ggqo",
    "channels_rank": "https://drive.google.com/uc?id=12UlGf6xVbteJtzo8JzysuhPmujkzACiw",
    "group_256_rank": "https://drive.google.com/uc?id=1tRGYGIqkpcD3t67mIKXs-bLlPZn1j25E",
    "mapper": "https://drive.google.com/uc?id=1juhw71f7YQbApYYpOQI_kU3BOEk1Z4f5",
    "psp_seg": "https://drive.google.com/uc?id=1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz",
    "IR_SE50": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
    "curricular_face": "https://drive.google.com/uc?id=1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj", 
    "attr_net": "https://drive.google.com/uc?id=1wjRJM8O7RYOKJEN2X-4Dtg5XDkXhp9dh", 
    "shape_predictor": "https://drive.google.com/uc?id=1sePXzGZBzm1PAKvbEhcQ07LLojS2AkMu",
    "unet": "https://drive.google.com/uc?id=112SilQfnCM3_-Zugik6fu_EuXlmDP4Vi"
    
}

# StyleGAN2 FFHQ config f model
gdown.download(google_drive_paths['stylegan2'], 'pretrained_models/stylegan2-ffhq-config-f.pt', quiet=False)

# Samples of latent codes from CelebA-HQ
gdown.download(google_drive_paths['celeba_samples'], 'data/celeba_samples.pt', quiet=False)

# Channels rank
gdown.download(google_drive_paths['channels_rank'], 'statistics/channels_rank.npy', quiet=False)

# Group 256 channels rank
gdown.download(google_drive_paths['channels_rank'], 'statistics/group_256_rank.npy', quiet=False)

# mapper pretrained model
gdown.download(google_drive_paths['mapper'], 'pretrained_models/mapper.pt', quiet=False)

# PSP segmentation
gdown.download(google_drive_paths['psp_seg'], 'pretrained_models/psp_celebs_seg_to_face.pt', quiet=False)

# IR-SE50
gdown.download(google_drive_paths['IR_SE50'], 'pretrained_models/model_ir_se50.pth', quiet=False)

# Curricular
gdown.download(google_drive_paths['curricular_face'], 'pretrained_models/CurricularFace_Backbone.pth', quiet=False)

# Attrnet
gdown.download(google_drive_paths['attr_net'], 'pretrained_models/mobilenet_celeba.pth', quiet=False)

# Shape predictor
gdown.download(google_drive_paths['shape_predictor'], 'pretrained_models/shape_predictor_68_face_landmarks.dat', quiet=False)

# unet
gdown.download(google_drive_paths['unet'], 'pretrained_models/unet_model.pth', quiet=False)