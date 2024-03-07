import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss module that computes a total loss and optionally returns a dictionary of individual losses.

    Args:
        task_configs (List[Dict]): A list of dictionaries, where each dictionary contains 'loss_func',
                                    'weight' (optional), and 'name' (optional) for each task.
    """
    def __init__(self, task_configs: List[Dict[str, Union[nn.Module, float, str]]]):
        super(MultiTaskLoss, self).__init__()

        self.loss_funcs = nn.ModuleList([config['loss_func'] for config in task_configs])
        self.loss_weights = [config.get('weight', 1.0) for config in task_configs]
        self.loss_names = [config.get('name', f'task_{i}_loss') for i, config in enumerate(task_configs)]

    def forward(self, outputs: List[torch.Tensor], targets: List[torch.Tensor], get_loss_dict: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute the total loss or a dictionary of individual losses.

        Args:
            outputs (List[torch.Tensor]): List of output tensors.
            targets (List[torch.Tensor]): List of target tensors.
            get_loss_dict (bool): If True, returns a dictionary of individual losses along with the total loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]: The total loss or a tuple containing the total loss
            and a dictionary of individual losses (if get_loss_dict is True).
        """
        if len(outputs) != len(targets):
            raise ValueError("The number of outputs and targets must be the same.")

        total_loss = 0.0
        loss_dict = {}

        for idx, (output, target, loss_func, weight, loss_name) in enumerate(zip(outputs, targets, self.loss_funcs, self.loss_weights, self.loss_names)):
            loss = loss_func(output, target)
            if get_loss_dict:
                loss_dict[loss_name] = loss.item()  # Store unweighted loss only
            if weight != 1.0:
                loss = loss * weight
            total_loss += loss

        if get_loss_dict:
            return total_loss, loss_dict
        else:
            return total_loss
