import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import Dataset
from typing import Callable


from spuco.evaluate import Evaluator
from spuco.robust_train import BaseRobustTrain
from spuco.utils import Trainer
from spuco.utils import CustomIndicesSampler, Trainer

from spuco.utils.random_seed import seed_randomness

class GroupWeightedLoss(nn.Module):
    """
    A module for computing group-weighted loss.
    """
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        bias=3.0,
        lambda_=20
    ):
        """
        Initializes GroupWeightedLoss.

        :param criterion: The loss criterion function.
        :type criterion: Callable[[torch.tensor, torch.tensor], torch.tensor]
        :param num_groups: The number of groups to consider.
        :type num_groups: int
        :param group_weight_lr: The learning rate for updating group weights (default: 0.01).
        :type group_weight_lr: float
        :param device: The device on which to perform computations. Defaults to CPU.
        :type device: torch.device
        """
        super(GroupWeightedLoss, self).__init__()
        self.device = device
        self.bias = bias
        self.lambda_ = lambda_
        self.mse = torch.nn.MSELoss()
        self.sftcross = torch.nn.CrossEntropyLoss()

    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = (qm - pm).pow(2)#0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.mean()
        #print("log var1", qv)
        return kl

    def condition_prior(self, scale, label, dim):
        mean = ((label-scale[0])/(scale[1])).reshape(-1, 1).repeat(1, dim)
        var = torch.ones(label.size()[0], dim)
        return mean.to(self.device), var.to(self.device)

    def intervention_loss(self, intervention):
#         print(torch.sqrt(torch.sum(intervention**2)))
#         print(self.bias)
        return torch.norm(torch.sqrt(torch.sum(intervention**2))/4.-self.bias)

    def targets_loss(self, y_pred, int_y_pred):
        y1 = torch.argmax(y_pred, axis=1).clone().detach()
        y2 = torch.argmax(int_y_pred, axis=1).clone().detach()
        return -F.cross_entropy(y_pred, y2) - F.cross_entropy(int_y_pred, y1) #-self.mse(y_pred, int_y_pred)

    def kl_loss(self, m, v, y):
#         print(y)
        pm, pv = self.condition_prior([0., 1.], y, m.size()[1])
        return self.kl_normal(m, pv * 0.0001, pm, pv * 0.0001)

    def forward(self, z, int_z, y, y_pred, int_y_pred, intervention, turn='min'):
        v = torch.zeros_like(z)
        nll = F.cross_entropy(y_pred, y).mean()
        int_nll = -F.cross_entropy(int_y_pred, y).mean()
        kl = self.kl_loss(z, v, y).mean() - self.kl_loss(int_z, v, y).mean()
        inter_norm = self.intervention_loss(intervention).mean()
        targets_loss = self.targets_loss(y_pred, int_y_pred).mean()
#         print(self.targets_loss(y_pred, int_y_pred))

        all = self.lambda_*nll + 0.1*inter_norm + 0.1*targets_loss
        if turn == 'min':
            return all + 0.1*kl
        else:
            return -(0.1*int_nll + 0.1*targets_loss) + self.hparams['kl_lambda']*kl

    def adapt_loss(self, x, env_i):
        m, v, z, int_z, y_pred, int_y_pred, intervention, z_c = self.network(x)
        targets_loss = self.targets_loss(y_pred, int_y_pred).mean()
        return targets_loss

#     def forward(self, outputs, labels):
#         """
#         Computes the group-weighted loss.
#         """
#         # compute loss for different groups and update group weights
#         loss = self.criterion(outputs, labels)
#         group_loss = torch.zeros(self.num_groups).to(self.device)
#         for i in range(self.num_groups):
#             if (groups==i).sum() > 0:
#                 group_loss[i] += loss[groups==i].mean()
#         self.update_group_weights(group_loss)
#
#         # compute weighted loss
#         loss = group_loss * self.group_weights
#         loss = loss.sum()
#
#         return loss

    def update_group_weights(self, group_loss):
        group_weights = self.group_weights
        group_weights = group_weights * torch.exp(self.group_weight_lr * group_loss)
        group_weights = group_weights / group_weights.sum()
        self.group_weights.data = group_weights.data

class GroupBalanceCaSN(BaseRobustTrain):
    """
    Group DRO (https://arxiv.org/abs/1911.08731)
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        group_partition: Dict,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        lr_scheduler=None,
        max_grad_norm=None,
        val_evaluator: Evaluator = None,
        verbose=False,
        bias=3.0,
        lambda_=20
    ):
        """
        Initializes GroupDRO.

        :param model: The PyTorch model to be trained.
        :type model: nn.Module
        :param trainset: The training dataset containing group-labeled samples.
        :type trainset: GroupLabeledDatasetWrapper
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param optimizer: The optimizer used for training.
        :type optimizer: optim.Optimizer
        :param num_epochs: The number of training epochs.
        :type num_epochs: int
        :param device: The device to be used for training (default: CPU).
        :type device: torch.device
        :param verbose: Whether to print training progress (default: False).
        :type verbose: bool
        """

        seed_randomness(torch_module=torch, random_module=random, numpy_module=np)

        super().__init__(val_evaluator=val_evaluator, verbose=verbose)

        assert batch_size >= len(trainset.group_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"


        def forward_pass(self, batch):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            z, int_z, outputs, int_y_pred, intervention = self.model(inputs)
            loss = self.criterion(z, int_z, labels, outputs, int_y_pred, intervention)
            return loss, outputs, labels

        self.casn_loss = GroupWeightedLoss(device=device, bias=bias)

        self.num_epochs = num_epochs
        self.group_partition = group_partition
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=self.casn_loss,
            forward_pass=forward_pass,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            sampler=CustomIndicesSampler(indices=[]),
            verbose=verbose,
            device=device)


        max_group_len = max([len(self.group_partition[key]) for key in self.group_partition.keys()])
        self.base_indices = []
        self.sampling_weights = []
        for key in self.group_partition.keys():
            self.base_indices.extend(self.group_partition[key])
            self.sampling_weights.extend([max_group_len / len(self.group_partition[key])] * len(self.group_partition[key]))
        
    def train_epoch(self, epoch: int):
        """
        Trains the model for a single epoch with a group balanced batch (in expectation)

        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.trainer.sampler.indices = random.choices(
            population=self.base_indices,
            weights=self.sampling_weights, 
            k=len(self.trainer.trainset)
        )
        self.trainer.train_epoch(epoch)
