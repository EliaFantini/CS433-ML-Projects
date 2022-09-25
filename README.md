<p align="center">
  <img alt="🛣️Road_Segmentation" src="https://user-images.githubusercontent.com/62103572/182944121-e735607d-3166-4ed2-9753-fb9a494928ac.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier?abel=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier?style=social">
</p>

This project aims at classifying the pixels who represent a road and those who don't in an aerial/satellite image, thanks to the use of a Convolutional Neural Network (CNN).

The problem was part of an artificial intelligence [Road Segmentation Challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) from AICrowd. Our team, called Pasta-Balalaika, reached the position 12/107 on the leaderboard, with an F1 score of *0.910* and an accuracy	of *0.952*. This project was also done as an assignment of the EPFL course [CS-433 Machine Learning](https://edu.epfl.ch/coursebook/en/machine-learning-CS-433).

The following image shows an example of the prediction made by our final model:
<p align="center">
  <img width="764" alt="Immagine 2022-08-04 222840" src="https://user-images.githubusercontent.com/62103572/182946552-d66d73fc-6b7b-4858-9ff5-ddc4fb6c4cc3.png">
</p>

## Authors 
-  [Anastasiia Filippova](https://github.com/nastya236)
-  [Elia Fantini](https://github.com/EliaFantini)
-  [Narek Alvandian](https://github.com/narekvslife)

Summary
========

 * How to install
 * Usage
 * Results and Examples
 * Conclusion and Future work

## How to install
- Download the project repository:
```shell
git clone https://github.com/EliaFantini/Road-Segmentation-convolutional-neural-network-classifier.git

cd  Road-Segmentation-convolutional-neural-network-classifier
```
- Create a virtual enviroment (download and install [Python 3](https://www.python.org/downloads/) first if you don't have it already):
```shell
python3 -m pip install --user virtualenv

virtualenv -p python3 pasta-balalaika
```
- Activate the enviroment:
```shell
source pasta-balalaika/bin/activate
```
- Install the requirements:
```shell
python3 -m pip install -r requirements.txt
```
## Usage
In the repository, the `libseg` python package contains all the code of the project.

The package `experiments` contains the final notebook, in which you can find the running of different pipelines and experiments, plots of the losses and metrics. 

`libseg`:
  * `dataset`: contains the code of `class SegmentDataset` for creating items for training, validation or testing.

  * `losses`: contains the code for `DiceLoss`, which was used during training.

  * `model`: in this package you could find the code for the `class Model` - the main class for training and testing.

  * `nets`:comprises the code of 3 different Neural Networks: 

    - [UNet](https://arxiv.org/abs/1505.04597)
    - [DeepLab](https://arxiv.org/abs/1706.05587)
    - [DeepLabPlus](https://arxiv.org/pdf/1802.02611.pdf)

    More information about the neural networks as well as our explanations of why we used these architectures can be found in our report.

  * `preprocessing`: this module comprises the code for data preprocessing:

    - data augmentation (flipping, rotations)
    - gamma correction
    - clache
    - standardization 

    More about preprocessing methods you could find in our report.

  * `utils`: consists of the helpers such as `train_valid_split` - function for splitting the data into train and valid, `fix_seed` - function to fix all random processes during the training such as model initialization, spliting and so on.

  Also this module includes the code for cropping images into patches for the final submission and functions for choosing the criterion and net.

  * `config` file consists of:

      - `seed` - random state

      - `valid_size` - size of the validation part,

      - `data_path` - the global path to the data folder,

      -  `clahe` - the flag for clahe,

      -  `gamma_correction` - the flag for gamma correction,

      -  `gamma` - gamma value,

      - `normalize_and_centre` - the flag for normalization,

      - `data_augmentation` - the flag for data augmentation,

      - `num_rotations` - if `data_augmentation` is `True` than this is the number of random rotations,

      - `divide_in_patches` - the flag for dividing into patches (small subimages),

      - `patches_size` - if `divide_in_patches` is `True`, this is the size of the patches,

      - `batch_size` - batch size for training,

      - `backbone` - the name of the net,

      - `criterion` - the name of the criterion,

      - `optimizer_learning_rate` - learning rate,

      - `epochs` - the number of the epochs to train,

      - `epochs_print_gap` - the gap to verbose current metrics and losses,

      - `foreground_threshold` - the threshold for background,

      - `device` - the name of the device (cpu or cuda),

      - `create_submission` - the flag for creating the submission after training,

      - `save_model` - the flag for saving the model weights after training,

      - `train` - train or test,

      - `postprocessing` - the flag for using ensambling during testing (more in the report)
## How to train and test
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

## Results 

We receive the best score on test data (0.91 F1 score) with `DeepLabPlus` Neural Network, `DiceLoss`. We trained the net 50 epoches with `Adam` optimizer.

More important details about our work can be found in the report.

In this picture you can see the plots of train/test loss, F1 score and accuracy for different networks and preprocessing pipelines:
<img width="auto" alt="Immagine 2022-08-05 105935" src="https://user-images.githubusercontent.com/62103572/183042522-7e84c0ea-c7c2-4785-99fb-e5fcebe492eb.png">
## Conclusion and Future work

In this project we were solving a road segmentation task. 

With pretrained DeepLabPlus model [link](https://github.com/qubvel/segmentation_models.pytorch), data preprocessing and ensambling predictions, we achieved
**0.910** F1 score on the test dataset.

Many different adaptations, tests, and experiments have been left for the future due to lack of time. The following ideas could be tested in the future:

- Add extra data to increase model's generalization ability. For example, can take [this dataset](https://www.cs.toronto.edu/~vmnih/data/).
- Use the combination of DiceLoss and BCELoss (inspired by [paper](https://arxiv.org/pdf/2006.14822.pdf)).
- Try other optimizers, such as AmsGrad and YOGI (inpired by this [paper](https://arxiv.org/abs/1904.09237) and this [paper](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html#:~:text=Adaptive%20gradient%20methods%20that%20rely,that%20arise%20in%20deep%20learning.), respectively)
- Try an ensemble machine learning algorithm such as Stacked Generalization.


## 🛠 Skills
Python, PyTorch, Matplotlib, Jupyter Notebooks. Machine learning and convolutional neural network knowledge, analysis of the impact of different preprocessing techniques on training, plotting the experiments, ensuring reproducibility.
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://eliafantini.github.io/Portfolio/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)

