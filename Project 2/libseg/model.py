from torch import nn
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from libseg.utils import instantiate_criterion, instantiate_net
from libseg.utils import pixel_to_patch


class Model(nn.Module):
    def __init__(self, backbone, criterion, device):
        super().__init__()
        self.device = device
        self.backbone = instantiate_net(backbone)
        self.criterion = instantiate_criterion(criterion)
        self.backbone.to(self.device)

    def forward(self, batch_data):
        x = batch_data[0].to(self.device)
        prediction = self.backbone(x)
        return prediction

    def train(self, train_loader, valid_loader, valid_eval_loader, config):
        optimizer = torch.optim.Adam(self.backbone.parameters(),
                                     lr=config['optimizer_learning_rate'])

        train_loss = []
        valid_loss = []
        valid_acc = []
        valid_f1 = []
        best_loss = (0, float('inf'))
        i = 1
        while True:
            train_loss_epoch = self.train_epoch(train_loader,
                                                self.criterion,
                                                optimizer)
            train_loss.append(train_loss_epoch)
            valid_loss_epoch  = self.valid_epoch(valid_loader, self.criterion)
            valid_loss.append(valid_loss_epoch)
            if valid_eval_loader is not None:
                f1, acc = self.evaluate(valid_eval_loader,
                                   config['postprocessing'])
            else:
                #if run on the whole data
                acc = 0
                f1 = 0
            valid_acc.append(acc)
            valid_f1.append(f1)

            if i % config['epochs_print_gap'] == 0 and not config['whole_data']:
                print(f'Epoch {i}: train loss {np.round(train_loss_epoch, 4)}, '
                      f'valid loss {np.round(valid_loss_epoch, 4)}, '
                      f'patch accuracy {np.round(acc, 4)}, '
                      f'patch f1  {np.round(f1, 4)}')

            if valid_loss_epoch < best_loss[1]:
                best_loss = (i, valid_loss_epoch)

            if config['early_stopping'] and  + config['early_stopping_treshold'] < i:
                print('Early stopping; the model was overfitting. Stopped at epoch {}.'.format(i))
                break

            if not config['early_stopping'] and i > (config['epochs']-1):
                print('Training finished: stopped at epoch {}.'.format(i))
                break
            i += 1

        if config['save_model']:
            torch.save(self.state_dict(), f'model.pt')

        return train_loss, valid_loss, valid_acc, valid_f1

    def train_epoch(self, loader, criterion, optimizer):
        """
        Trains the model for a whole epoch
        @param loader: loader containing training data
        @param criterion: criterion to use as loss function
        @param optimizer: optimizer to use to update weights
        @return: float
            Average of the losses computed during the whole epoch
        """
        loss_epoch = []
        self.backbone.train()
        for batch in loader:
            prediction = self.forward(batch)
            target = batch[1].to(self.device)
            loss = criterion(prediction, target).to(self.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            loss_epoch.append(loss)

        return np.mean(loss_epoch)

    def valid_epoch(self, valid_loader, criterion):
        """
        Validates model performance on valid data
        @param valid_loader: loader containing validation data
        @param criterion: criterion to use as a loss function
        @return: float
            Average of the losses computed during the whole epoch
        """
        self.backbone.eval()
        loss_epoch = []
        with torch.no_grad():
            for batch in valid_loader:
                prediction = self.forward(batch)
                target = batch[1].to(self.device)

                loss = criterion(prediction, target).to(self.device)
                loss = loss.cpu().detach().numpy()
                loss_epoch.append(loss)

        return np.mean(loss_epoch)

    def predict(self, test_loader,
                postprocessing):
        """
        Computes prediction on test_loader data, applying sigmoid function
        @param test_loader: loader containing test data
        @param postprocessing: bool
            True to compute predictions on the image from different perspectives, such as rotations and flips.
            Return the average of predictions
        @return: list
            List of predictions
        """
        predicted_probs = []
        self.backbone.eval()
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            for batch in test_loader:
                prediction = sigmoid(self.backbone(batch.to(self.device))).cpu().detach().numpy()
                predicted_probs.append(prediction[0][0])
        if postprocessing:
            predicted_probs = self._merge_predictions(predicted_probs)

        return predicted_probs

    @staticmethod
    def _prediction_to_patch(prediction, number,
                             patch_size=16, step=16,
                             threshold=0.25):
        """
        Converts pixel-wise predictions to patch-wise ones
        @param prediction: np.ndarray
            Array of pixels predictions
        @param number: int
        @param patch_size: int
            Size of the patch
        @param step: int
            Step size in for loops. Usually is the same as patch size.
        @param threshold: float
            If the average of pixels equals to 1 in the patch is greater than the threshold,
            the patch prediction is 1, 0 otherwise.
    @return: list, list
            Two lists, one for labels, one for numbers.
        """
        labels = []
        numbers = []
        for j in range(0, prediction.shape[1], step):
            for i in range(0, prediction.shape[0], step):
                labels.append(pixel_to_patch(prediction[i:i + patch_size, j:j + patch_size],
                                             foreground_threshold=threshold))
                numbers.append("{:03d}_{}_{}".format(number, j, i))

        return labels, numbers

    def make_submission(self,
                        test_loader,
                        path,
                        ids=None):
        """
        Creates the csv document containing predictions to be submitted to AiCrowd platform.
        @param test_loader: loader containing test data
        @param path: string
            Path where to save the csv output file
        @param ids: np.ndarray
            Array of indexes for alternative index. For default index put None.

        """
        sub_ids = []
        sub_labels = []
        predictions = self.predict(test_loader, True)

        if ids is None:
            ids = range(1, len(predictions)+1)

        for pred, number in zip(predictions, ids):
            labels, names = self._prediction_to_patch(pred, number)
            sub_labels.extend(labels)
            sub_ids.extend(names)

        submission_df = pd.DataFrame({'id': sub_ids, 'prediction': sub_labels})
        submission_df.to_csv(path, index=False)

    def evaluate(self, valid_loader,
                 postprocessing):
        """
        Computes metrics (f1, accuracy) on validation data
        @param valid_loader: loader containing validation data
        @param postprocessing: bool
            True to compute predictions on the image from different perspectives, such as rotations and flips.
            Return the average of predictions
        @return: float, float
            F1 and accuracy scores
        """
        self.backbone.eval()
        predicted_labels = []
        labels = []
        merged_predictions = self.predict(valid_loader,
                                          postprocessing)

        target_masks = valid_loader.dataset.gt

        for pred, tar in zip(merged_predictions, target_masks):
            predicted_labels_img, _ = self._prediction_to_patch(pred, 1)
            predicted_labels.extend(predicted_labels_img)
            tar = tar[0].cpu().detach().numpy()
            labels_img, _ = self._prediction_to_patch(tar, 1)
            labels.extend(labels_img)
        f1 = f1_score(labels, predicted_labels)
        accuracy = accuracy_score(labels, predicted_labels)

        return f1, accuracy


    @staticmethod
    def _merge_predictions(predictions,
                           num_predictions=6):
        """
        Maps back patches on flipped and rotated images to original patches, then returns the predictions average
        @param predictions: list
            List of prediction matrix
        @param num_predictions: int
            Number of predictions to compute the average of
        @return: list
            List of merged predictions
        """
        merged_predictions = []
        predictions = np.array(predictions)
        size = predictions[0][0].shape[0]
        prediction_reshaped = predictions.reshape((-1, num_predictions, size, size))
        for images in prediction_reshaped:
            inversed_flr = np.fliplr(images[1])
            inversed_fub = np.flipud(images[2])
            inversed_90 = np.rot90(images[3], k=3)
            inversed_180 = np.rot90(images[4], k=2)
            inversed_270 = np.rot90(images[5], k=1)

            merged_prediction = np.stack([images[0], inversed_flr, inversed_fub,
                                          inversed_90, inversed_180, inversed_270])

            final_prediction = np.mean(merged_prediction, axis=0)
            merged_predictions.append(final_prediction)
        return merged_predictions