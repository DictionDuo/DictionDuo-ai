# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import torch.nn as nn
import pandas as pd
from torch import Tensor
from typing import Tuple
import preprocessing.pronunciation_dataloader as dataloader

# from kospeech.optim import Optimizer
# from kospeech.vocabs import Vocabulary
# from kospeech.checkpoint import Checkpoint
# from kospeech.metrics import CharacterErrorRate
from kospeech.utils import logger


class SupervisedTrainer(object):
    """
    The SupervisedTrainer class sets up the training framework 
    for supervised learning of pronunciation correction using audio features.

    Args:
        optimizer (torch.optim.Optimizer): optimizer for training
        criterion (torch.nn.Module): loss function
        trainset_list (list): list of training datasets (can be concatenated)
        validset (torch.utils.data.Dataset): validation dataset
        num_workers (int): number of CPU cores used for data loading
        device (torch.device): device - 'cuda' or 'cpu'
        print_every (int): number of iterations to print training status
        save_result_every (int): number of iterations to save intermediate results
        checkpoint_every (int): number of iterations to save model checkpoints
    """

    train_dict = {'loss': []}
    valid_dict = {'loss': []}
    train_step_result = {'loss': []}
    
    TRAIN_RESULT_PATH = "train_result.csv"
    VALID_RESULT_PATH = "eval_result.csv"
    TRAIN_STEP_RESULT_PATH = "train_step_result.csv"

    def __init__(
            self,
            optimizer,                                     # optimizer for training
            criterion: nn.Module,                          # loss function
            trainset_list: list,                           # list of training dataset
            validset,                                      # validation dataset
            num_workers: int,                              # number of threads
            device: torch.device,                          # device - cuda or cpu
            print_every: int,                              # number of timesteps to save result after
            save_result_every: int,                        # nimber of timesteps to save result after
            checkpoint_every: int,                         # number of timesteps to checkpoint after
    ) -> None:
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset_list = trainset_list
        self.validset = validset
        self.print_every = print_every
        self.save_result_every = save_result_every
        self.checkpoint_every = checkpoint_every
        self.device = device

        self.log_format = "step: {:4d}/{:4d}, loss: {:.6f}, " \
                  "elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"

    def train(
        self,
        model: nn.Module,                           # model to train
        batch_size: int,                            # batch size for experiment
        epoch_time_step: int,                       # number of time step for training
        num_epochs: int,                            # number of epochs (iteration) for training
    ) -> nn.Module:
        """
        Run training for a given model.

        Args:
            model (torch.nn.Module): model to train
            batch_size (int): batch size for experiment
            epoch_time_step (int): number of time step for training
            num_epochs (int): number of epochs for training
        """
        start_epoch = 0

        logger.info('start')
        train_begin_time = time.time()

        model = model.to(self.device)

        for epoch in range(num_epochs):
            logger.info('Epoch %d start' % epoch)

            for trainset in self.trainset_list:
                trainset.shuffle()

            # Training
            model, train_loss = self._train_epoches(
                model=model,
                epoch=epoch,
                epoch_time_step=epoch_time_step,
                train_begin_time=train_begin_time,
                batch_size=batch_size
            )

            logger.info(f'Epoch %d (Training) Loss %0.4f' % (epoch, train_loss))

            # Validation
            valid_loss = self._validate(model, batch_size=batch_size)

            logger.info(f'Epoch %d (Validation) Loss %0.4f' % (epoch, valid_loss))
            self._save_epoch_result(train_result=[self.train_dict, train_loss],
                                     valid_result=[self.valid_dict, valid_loss])
            logger.info(f'Epoch %d Training result saved as a csv file complete !!' % epoch)
            torch.cuda.empty_cache()

        return model

    def _train_epoches(
            self,
            model: nn.Module,
            epoch: int,
            epoch_time_step: int,
            train_begin_time: float,
            batch_size: int,
    ) -> Tuple[nn.Module, float]:
        """
        Run training one epoch

        Args:
            model (torch.nn.Module): model to train
            epoch (int): number of current epoch
            epoch_time_step (int): total time step in one epoch
            train_begin_time (float): time of train begin

        Returns:
            Tuple[nn.Module, float]: 
                - model (nn.Module): trained model after this epoch
                - loss (float): average training loss for this epoch
        """
        epoch_loss_total = 0.
        total_frames = 0
        timestep = 0

        model.train()

        begin_time = epoch_begin_time = time.time()

        all_audio_file_pairs = []
        for dataset in self.trainset_list:
            all_audio_file_pairs.extend(dataset.audio_file_pairs)

        train_loader = dataloader.create_dataloader(
            audio_file_pairs=all_audio_file_pairs,
            max_length=512,     # 필요에 따라 조정
            batch_size=batch_size,      # 또는 self.batch_size로 받아도 OK
            shuffle=True
        )

        for inputs, targets in train_loader:  # inputs: [B, T, D]
            if inputs is None:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs, loss = self._model_forward(model, inputs, targets)

            loss.backward()
            self.optimizer.step()

            num_frames = inputs.size(1)
            epoch_loss_total += loss.item() * num_frames
            total_frames += num_frames

            timestep += 1
            torch.cuda.empty_cache()

            if timestep % self.print_every == 0:
                current_time = time.time()
                elapsed = current_time - begin_time
                epoch_elapsed = (current_time - epoch_begin_time) / 60.0
                train_elapsed = (current_time - train_begin_time) / 3600.0

                logger.info(self.log_format.format(
                    timestep, epoch_time_step, loss.item(),
                    elapsed, epoch_elapsed, train_elapsed,
                    self.optimizer.param_groups[0]['lr'],
                ))
                begin_time = time.time()

            if timestep % self.save_result_every == 0:
                self._save_step_result(
                    self.train_step_result,
                    epoch_loss_total / total_frames,
                    cer = None
                )

        logger.info('train() completed')

        avg_loss = epoch_loss_total / total_frames
        return model, avg_loss

    def _validate(self, model: nn.Module, batch_size: int) -> float:
        """
        Run training one epoch

        Args:
            model (torch.nn.Module): model to train

        Returns: loss
            - **loss** (float): loss of validation
        """

        model.eval()
        logger.info('validate() start')

        all_audio_file_pairs = []
        for dataset in self.validset:
            all_audio_file_pairs.extend(dataset.audio_file_pairs)

        valid_loader = dataloader.create_dataloader(
            audio_file_pairs=all_audio_file_pairs,
            max_length=512,
            batch_size=batch_size,
            shuffle=False
        )

        val_loss_total = 0.
        total_frames = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                if inputs is None:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, loss = self._model_forward(model, inputs, targets)

                num_frames = inputs.size(1)
                val_loss_total += loss.item() * num_frames
                total_frames += num_frames

        avg_val_loss = val_loss_total / total_frames
        logger.info(f'Validation completed. Average Loss: {avg_val_loss:.6f}')

        return avg_val_loss

    def _model_forward(
            self,
            model: nn.Module,
            inputs: Tensor,
            targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for pronunciation correction model (Conformer-based).
        
        Args:
            model (nn.Module): the model to forward
            inputs (Tensor): input feature tensor [B, T, D]
            targets (Tensor): target mel spectrogram [B, T, 128]
        
        Returns:
            Tuple containing:
                - outputs (Tensor): predicted mel spectrogram [B, T, 128]
                - loss (Tensor): training loss
        """
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)
        
        return outputs, loss

    def _save_result(self, target_list: list, predict_list: list) -> None:
        results = {
            'targets': [t.tolist() if hasattr(t, 'tolist') else t for t in target_list],
            'predictions': [p.tolist() if hasattr(p, 'tolist') else p for p in predict_list]
        }
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        save_path = f"{date_time}-valid.csv"
        
        results = pd.DataFrame(results)
        results.to_csv(save_path, index=False, encoding='utf-8-sig')

    def _save_epoch_result(self, train_result: list, valid_result: list) -> None:
        """ Save result of epoch """
        train_dict, train_loss = train_result
        valid_dict, valid_loss = valid_result

        train_dict["loss"].append(train_loss)
        valid_dict["loss"].append(valid_loss)

        train_df = pd.DataFrame(train_dict)
        valid_df = pd.DataFrame(valid_dict)

        train_df.to_csv(SupervisedTrainer.TRAIN_RESULT_PATH, encoding="utf-8-sig", index=False)
        valid_df.to_csv(SupervisedTrainer.VALID_RESULT_PATH, encoding="utf-8-sig", index=False)

    def _save_step_result(self, train_step_result: dict, loss: float) -> None:
        """ Save result of --save_result_every step """
        train_step_result["loss"].append(loss)

        train_step_df = pd.DataFrame(train_step_result)
        train_step_df.to_csv(SupervisedTrainer.TRAIN_STEP_RESULT_PATH, encoding="utf-8-sig", index=False)