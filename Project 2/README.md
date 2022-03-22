# [Machine Learning Course EPFL](https://www.epfl.ch/labs/mlo/machine-learning-cs-433) 2021, Course Project 2

This repository contains the solution of team Pasta-Balalaika of [Road Segmentation Challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) on AI Crowd.
The detailed project describtion is available [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation). 

## Team 
-  **Anastasiia Filippova** ([@nastya236](https://github.com/nastya236))
-  **Elia Fantini** ([@EliaFantini](https://github.com/EliaFantini))
-  **Narek Alvandian** ([@narekvslife](https://github.com/narekvslife))

Contents
========

 * [Installation](#installation)
 * [Usage](#usage)
 * [Results and Examples](#results)
 * [Conclusion and Future work](#conclusion)

### Installation
---
0. Download the project repository:

`git clone https://github.com/CS-433/ml-project-2-pasta_balalaika.git`

`cd  ml-project-2-pasta_balalaika`

1. Create virtual enviroment:

`python3 -m pip install --user virtualenv`

` virtualenv -p python3 pasta-balalaika`

2. Activate the enviroment:

`source pasta-balalaika/bin/activate`

3. Install the requirements:

`python3 -m pip install -r requirements.txt`

### Usage
---
In the repository, the `libseg` python package contains all the code of the project.

The package `experiments` contains the final notebook, in which you can find the running of different pipelines, plots of the losses and metrics. 

`libseg`:
* `dataset`

Contains the code of `class SegmentDataset` for creating items for training, validation or testing.

* `losses` 

Contains the code for `DiceLoss`, which was used during training.

* `model`

In this package you could find the code for the `class Model` - the main class for training and testing.

* `nets`

Comprises the code of 3 different Neural Networks: 

- [UNet](https://arxiv.org/abs/1505.04597)
- [DeepLab](https://arxiv.org/abs/1706.05587)
- [DeepLabPlus](https://arxiv.org/pdf/1802.02611.pdf)

More information about the neural networks as well as our explanations of why we used these architectures can be found in our report.

* `preprocessing`

This module comprises the code for data preprocessing:

- data augmentation (flipping, rotations)
- gamma correction
- clache
- standardization 

More about preprocessing methods you could find in our report.

* `utils`

Consists of the helpers such as `train_valid_split` - function for splitting the data into train and valid, `fix_seed` - function to fix all random processes during the training such as model initialization, spliting and so on.

Also this module includes the code for cropping images into patches for the final submission and functions for choosing the criterion and net.

* `config`

Config file consists of:

`seed` - random state

`valid_size` - size of the validation part,

`data_path` - the global path to the data folder,

`clahe` - the flag for clahe,

`gamma_correction` - the flag for gamma correction,

`gamma` - gamma value,

`normalize_and_centre` - the flag for normalization,

`data_augmentation` - the flag for data augmentation,

`num_rotations` - if `data_augmentation` is `True` than this is the number of random rotations,

`divide_in_patches` - the flag for dividing into patches (small subimages),

`patches_size` - if `divide_in_patches` is `True`, this is the size of the patches,

`batch_size` - batch size for training,

`backbone` - the name of the net,

`criterion` - the name of the criterion,

`optimizer_learning_rate` - learning rate,

`epochs` - the number of the epochs to train,

`epochs_print_gap` - the gap to verbose current metrics and losses,

`foreground_threshold` - the threshold for background,

`device` - the name of the device (cpu or cuda),

`create_submission` - the flag for creating the submission after training,

`save_model` - the flag for saving the model weights after training,

`train` - train or test,

`postprocessing` - the flag for using ensambling during testing (more in the report)

In order to train the model:
1. Download the data for train and test [here](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)
2. Put `data` folder with data into project repository
3. Run script with following command: `python run.py` (`config['train'] = True`):

Using the default config, you can reproduce our result. 

The submission will be saved into the file `submissions.csv` and the model into the file `model.pt` (if all required flags are `True`).

The weights of the final model can also be found [here](https://drive.google.com/file/d/1zFTWKPMWSVhf_J9ONPCJwFa-F1FLk16n/view?usp=sharing). 

In order to evalute our final model:

1. Download the weight from [disc](https://drive.google.com/file/d/1zFTWKPMWSVhf_J9ONPCJwFa-F1FLk16n/view?usp=sharing).
2. Put downloaded file `model.pt` in the project repo (same folder of `run.py`)
3. Change `config['train'] = False`

### Results and Examples
---

We receive the best score on test data (0.91 F1 score) with `DeepLabPlus` Neural Network, `DiceLoss`. We trained the net 50 epoches with `Adam` optimizer.

More important details about our work can be found in the report.

In this picture you can see the example of the evaluation on the test data:

![telegram-cloud-photo-size-2-5341643872939392277-y](https://user-images.githubusercontent.com/41966024/147132914-c5be03bf-1f43-482d-a28b-8184f6aaa4b1.jpg)

### Conclusion and Future work
---

In this project we were solving a road segmentation task. 

With pretrained DeepLabPlus model [link](https://github.com/qubvel/segmentation_models.pytorch), data preprocessing and ensambling predictions, we achieved
**0.910** F1 score on the test dataset.

Many different adaptations, tests, and experiments have been left for the future due to lack of time. The following ideas could be tested in the future:

- Add extra data to increase model's generalization ability. For example, can take [this dataset](https://www.cs.toronto.edu/~vmnih/data/).
- Use the combination of DiceLoss and BCELoss (inspired by [paper](https://arxiv.org/pdf/2006.14822.pdf)).
- Try other optimizers, such as AmsGrad and YOGI (inpired by this [paper](https://arxiv.org/abs/1904.09237) and this [paper](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html#:~:text=Adaptive%20gradient%20methods%20that%20rely,that%20arise%20in%20deep%20learning.), respectively)
- Try an ensemble machine learning algorithm such as Stacked Generalization.
