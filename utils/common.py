import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from tqdm import tqdm
import string

# Log images
def log_input_image(x, opts):
    if opts.label_nc == 0:
        return tensor2im(x)
    elif opts.label_nc == 1:
        return tensor2sketch(x)
    else:
        return tensor2map(x)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    result = Image.fromarray(var.astype('uint8'))
    return result


def tensor2map(var):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


def tensor2sketch(var):
    im = var[0].cpu().detach().numpy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = (im * 255).astype(np.uint8)
    return Image.fromarray(im)


# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 3)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        if 'diff_input' in hooks_dict:
            vis_faces_with_id(hooks_dict, fig, gs, i)
        else:
            vis_faces_no_id(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')

def normalize(input_array):
    if not type(input_array) == np.ndarray:
        input_array = np.array(input_array)
    return (input_array - input_array.mean())/input_array.std()

def moving_average(x, w_size):
    return np.convolve(x, np.ones(w_size), 'valid') / w_size

def minMaxScaler(x):
    if not type(x) == np.ndarray:
        x = np.array(x)
    return (x - x.min())/(x.max() - x.min())

def viz_2d_array(array, xtick_step = 1, label_offset = 0, markers = '.', dpi=80, label_prefix = 'w_size'):
    figure(figsize=(8, 6), dpi=dpi)
    for i in range(array.shape[0]):
        x = range(len(array[i]))
        plt.plot(x, array[i], markers, label = f'{label_prefix}: {i+label_offset}')

    plt.legend()
    plt.xticks(np.arange(0, len(array[0]), step=xtick_step))
    #plt.yticks(np.arange(0, 1.4, step=0.05))
    plt.show()
    
def mean_from_paths(dist_dir, save_path, n_levels = 2):
    stats = []
    if n_levels == 2:
        first_levels = sorted([int(d) for d in os.listdir( dist_dir ) if 'ipynb_checkpoints' not in d])
        for first_level in tqdm(first_levels):
            cur_dir = os.path.join(dist_dir, str(first_level))
            layers = sorted([int(d) for d in os.listdir( cur_dir ) if 'ipynb_checkpoints' not in d])
            layer_stats = {}
            layer_means = []
            for layer in layers:
                layer_dir = os.path.join(cur_dir, str(layer))
                file_paths = [os.path.join(layer_dir, f) for f in os.listdir(layer_dir) if 'ipynb_checkpoints' not in f]
                layer_stats[layer] = np.array([np.load(file_path) for file_path in file_paths])
                layer_mean = np.mean(layer_stats[layer])
                layer_means.append(layer_mean)
            stats.append(layer_means)
    np.save(save_path, stats)
    return np.array(stats)

def minMaxScaler(x):
    if type(x) != np.ndarray or len(x[0]) != len(x[1]):
        x_max = max([max(i) for i in x])
        x_min = min([min(i) for i in x])
        x_delta = x_max-x_min
        return np.array([ (np.array(j) - x_min)/x_delta for j in x]) 
    return (x - x.min())/(x.max() - x.min())

def top_n(arr_2d, n):
    if not type(arr_2d) == np.ndarray:
        arr_2d = np.array(arr_2d)
    assert arr_2d.ndim == 2, 'Input should be 2-d numpy array'
    assert n > 0, 'Invalid input: n should be greater than 0'
    n_col = arr_2d.shape[1]
    n_row = arr_2d.shape[0]
    ind = np.argsort(arr_2d.reshape(-1))[-n:]
    return [(i//n_col, i%n_col) for i in ind][::-1]

def expand_step(inp, step):
    result = []
    for i in inp:
        for j in range(step):
            result.append( (i[0], i[1]*step + j) )
    return result

def swap_by_index(top_choices, x, y):
    temp = y.detach().clone()
    res = x.detach().clone()
    for idx in top_choices:
        i, j = idx
        res[ :, i, j] = temp[:, i, j]
    return res
    
