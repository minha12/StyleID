from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--mapper_type', default='SingleMapper', type=str, help='Which mapper to use')
        self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
        self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
        self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
        self.parser.add_argument('--img_dir', default="", type=str, help="The images directory for both training and testing")
        self.parser.add_argument('--embed_dir', default="", type=str, help="The id embeddings for both training and testing")
        self.parser.add_argument('--latents_src_path', default="latents.pt", type=str, help="The source latents for both training and testing")
        self.parser.add_argument('--latents_tar_path', default="random_latents.pt", type=str, help="The source latents for both training and testing")
        self.parser.add_argument('--latents_train_path', default="train_faces.pt", type=str, help="The latents for the training")
        self.parser.add_argument('--latents_truth_path', default="train_faces.pt", type=str, help="The latents groud truth for the mapper")
        self.parser.add_argument('--latents_test_path', default="test_faces.pt", type=str, help="The latents for the validation")
        self.parser.add_argument('--train_dataset_size', default=5000, type=int, help="Will be used only if no latents are given")
        self.parser.add_argument('--test_dataset_size', default=1000, type=int, help="Will be used only if no latents are given")
        self.parser.add_argument('--work_in_stylespace', default=False, action='store_true', help="trains a mapper in S instead of W+")
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--clip_lambda', default=-1.0, type=float, help='CLIP loss multiplier factor')
        self.parser.add_argument('--latent_l2_lambda', default=-1, type=float, help='Latent L2 loss multiplier factor')
        self.parser.add_argument('--landmark_l2_lambda', default=-1, type=float, help='Landmark L2 loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 pixel loss multiplier factor')
        self.parser.add_argument('--l2w_lambda', default=1.0, type=float, help='L2 W loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=-1, type=float, help='W-norm loss multiplier factor')

        self.parser.add_argument('--stylegan_weights', default='./pretrained_models/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--stylegan_size', default=1024, type=int)
        self.parser.add_argument('--ir_se50_weights', default='./pretrained_models/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
        self.parser.add_argument('--mobilefacenet_weights', default='./pretrained_models/mobilefacenet_model_best.pth.tar', type=str, help="Path to landmark detector used in Landmark loss")
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to StyleCLIPModel model checkpoint')

        self.parser.add_argument('--max_steps', default=30000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=1000, type=int, help='Model checkpoint interval')

        self.parser.add_argument('--description', default='text', type=str, help='Driving text prompt')
        self.parser.add_argument('--mask_dataset_path', default='../project/notebooks/CelebAMask-HQ/Inverted_Masks', type=str, help='Mask dataset path')
        self.parser.add_argument('--background_lambda', default=0.1, type=float, help='Background loss factor')
        


    def parse(self):
        opts = self.parser.parse_args()
        return opts