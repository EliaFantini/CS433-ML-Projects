import numpy as np
import torch
import matplotlib.pyplot as plt

from libseg.losses import DiceLoss
from libseg.nets.pretrained import get_deeplabplus, get_deeplab
from libseg.nets.unet import UNet


def train_valid_split(images, target,
                      valid_size: float,
                      random_state: int):
    """Gets train and valid separation.

    Parameters
    ----------
    images : list
        paths to images
    target : list
        paths to groundtruth
    valid_size : float from 0 to 1, optional
        Fraction of the data to be valid part.
    random_state : int or None, optional
        Random state. If None then no shuffling.

    Returns
    -------
    dict[str:list]
        Dict with keys `train` and `valid` that contains experiments ids.

    """
    ids = np.arange(len(images))
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(ids)
    sep1 = int(len(ids) * (1 - valid_size))
    train_exps = ids[:sep1]
    valid_exps = ids[sep1:]
    split = {"train": train_exps,
             "valid": valid_exps}
    X_train = np.array(images)[split['train']]
    y_train = np.array(target)[split['train']]

    X_valid = np.array(images)[split['valid']]
    y_valid = np.array(target)[split['valid']]

    return X_train, y_train, X_valid, y_valid


def fix_seeds(seed: int):
    """
    Fixes seed for all random functions
    @param seed: int
        Seed to be fixed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # TODO what is this?
    torch.backends.cudnn.benchmark = False  # TODO what is this?


def instantiate_net(net_name):
    """
    Instantiates the neural network
    @param net_name: string
        Name of the neural network. Accepted:'UNet'/ 'DeepLabPlus' / 'DeepLab'
    @return: nn.Module
        Net instance
    """
    if net_name == 'UNet':
        net = UNet(1, num_layers=5, features_start=32)
    elif net_name == 'DeepLabPlus':
        net = get_deeplabplus()
    elif net_name == 'DeepLab':
        net = get_deeplab()
    else:
        print("\n-unknown net in config in config!")
        return None
    return net


def instantiate_criterion(criterion_name):
    """
    Instantiates the criterion to be used for training
    @param criterion_name: str
        Name of the criterion. Accepted:'BCEWithLogitsLoss'/ 'DiceLoss'
    @return: _Loss
        Criterion instance
    """
    if criterion_name == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif criterion_name == 'DiceLoss':
        criterion = DiceLoss()
    else:
        print("\n-unknown criterion name in config!")
        return None
    return criterion


def pixel_to_patch(patch_array, foreground_threshold):

    """
    From a patch return a label (0 = background, 1 = foreground)
    :param patch_array: a patch
    :param foreground_threshold: percentage of pixels > 1 required to assign a foreground label to a patch
    :return: the label
    """

    mean_value = np.mean(patch_array)
    if mean_value > foreground_threshold:
        return 1
    else:
        return 0

def save_scores(train_loss, valid_loss,
                valid_acc, valid_f1):
    """
    Saves scores computed during training as plots in a png image, as well as displaying the plot.
    @param train_loss: list
            List of training losses computed during epochs
    @param valid_loss: list
            List of validation losses computed during epochs
    @param valid_acc: list
            List of accuracies computed during epochs
    @param valid_f1: list
            List of f1 computed during epochs
    """
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(30,10), sharex = True)
    axs[0].plot(valid_loss, label = "Validation loss")
    axs[0].plot(train_loss, label = "Train loss")
    axs[0].set_title("Loss", fontsize = 20)

    axs[1].plot(valid_acc, label = "Accuracy")
    axs[1].plot(valid_f1, label = 'F1 score')
    axs[1].set_title("Metrics", fontsize = 20)

    axs[1].set_xlabel("Epochs", fontsize = 20)
    axs[0].set_xlabel("Epochs", fontsize = 20)


    axs[1].legend(fontsize=20)
    axs[0].legend(fontsize=20)

    fig.savefig("scores.png")