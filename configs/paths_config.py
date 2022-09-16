dataset_paths = {
    'celeba_train': '',
    'celeba_test': '',
    'celeba_train_sketch': '',
    'celeba_test_sketch': '',
    'celeba_train_segmentation': '',
    'celeba_test_segmentation': '',
    'ffhq': '',
    'celeba_attr': 'CelebAMask-HQ-attribute-anno.txt'
}

model_paths = {
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'stylegan_seg': 'pretrained_models/psp_celebs_seg_to_face.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'mobilenet': './pretrained_models/mobilenet_celeba.pth', 
    'unet': './pretrained_models/unet_model.pth',
    'e4e': './pretrained_models/e4e_ffhq_encode.pt'
}
