import json
from glob import glob
import numpy as np
from PIL import ImageFile
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('..')

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

from libseg.model import Model
from libseg.utils import fix_seeds, train_valid_split, save_scores
from libseg.dataset import SegmentDataset

CONFIG_PATH = 'config.json'

def main(use_default_config=True, config=None):
    """
    Main function that loads the data, instantiates data loaders and model, trains the model and
    outputs predictions.
    :param use_default_config: bool
    True to use config.json as config dictionary. Default is True
    :param config: dict
    Dictionary containing all parameters, ignored if use_default_config is set to True. Default is None
    """
    if use_default_config:
        config = json.load(open(CONFIG_PATH))
    fix_seeds(config['seed'])
    data_path = config['data_path']

    images = sorted(glob(f'{data_path}/training/images/*.png'))
    target = sorted(glob(f'{data_path}/training/groundtruth/*.png'))

    X_train, y_train, X_valid, y_valid = train_valid_split(images, target,
                                               config['valid_size'],
                                                config['seed'])
    if config['whole_data']:
        X_train = np.array(images)
        y_train = np.array(target)

    training_dataset = SegmentDataset(X_train, y_train, config)

    validation_dataset = SegmentDataset(X_valid, y_valid, config)

    training_loader = DataLoader(training_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=True)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=config['batch_size'],
                                   shuffle=False)

    if config['whole_data']:
        valid_eval_loader = None
    else:
        valid_eval_dataset = SegmentDataset(X_valid, y_valid, config,
                                            target=True,
                                            mode='test')
        valid_eval_loader = DataLoader(valid_eval_dataset,
                                       batch_size=1,
                                       shuffle=False)

    model = Model(backbone=config['backbone'],
                  criterion=config['criterion'],
                  device=config['device'])

    if config['train']:
        train_loss, valid_loss, valid_acc, valid_f1 = model.train(training_loader,
                                                                  validation_loader,
                                                                  valid_eval_loader,
                                                                  config)
        if config['plot']:
            save_scores(train_loss, valid_loss,
                        valid_acc, valid_f1)
            print('Scores were saved!')

    else:

        model.load_state_dict(torch.load('model.pt'))

    if config['create_submission']:
        test_images = sorted(glob(f'{data_path}/test_set_images/*/*'),
                             key = lambda x: int(x.split('/')[-2].split('_')[-1]))
        test_dataset = SegmentDataset(test_images, None, config,
                                      mode='test', target = False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)
        model.make_submission(test_loader, 'submission.csv')


if __name__ == '__main__':
    config = {
        "seed": 23,
        "valid_size": 0.2,
        "data_path": "data/",
        "clahe": True,
        "gamma_correction": False,
        "gamma": 1.1,
        "normalize_and_centre": False,
        "data_augmentation": True,
        "num_rotations": 10,
        "divide_in_patches": False,
        "patches_size": 80,
        "batch_size": 10,
        "backbone": 'DeepLabPlus',
        "criterion": 'DiceLoss',
        "optimizer_learning_rate": 0.001,
        "early_stopping_treshold": 10,
        "early_stopping": False,
        "epochs": 50,
        "epochs_print_gap": 1,
        'foreground_threshold': 0.25,
        "device": 'cuda:4',
        "create_submission": True,
        "postprocessing": True,
        "save_model": True,
        "train": True,
        "plot": False,
        "whole_data": True
    }
    main(False, config)
