import torch.nn as nn

class MultiLoss(nn.Module):
    def __init__(self, loss_configs):
        super(MultiLoss, self).__init__()
        self.loss_list = nn.ModuleList([config['loss_func'] for config in loss_configs])
        self.loss_names = [config.get('name', f'loss_{i}') for i, config in enumerate(loss_configs)]
        self.loss_weights = [config.get('weight', 1.0) for config in loss_configs]

    def forward(self, outputs, targets):
        total_loss = 0.0
        loss_dict = {}

        for idx, (loss_func, loss_name) in enumerate(zip(self.loss_list, self.loss_names)):
            loss = loss_func(outputs, targets)
            loss_dict[loss_name] = loss.item()  # Store unweighted loss
            if self.loss_weights[idx] != 1.0:
                loss  = loss * self.loss_weights[idx]
            total_loss += loss

        return total_loss, loss_dict

# Usage
# loss_configs = [
#     {'loss_func': MSELoss(), 'name': 'MSE Loss', 'weight': 0.5},
#     {'loss_func': CrossEntropyLoss(), 'name': 'Cross-Entropy Loss', 'weight': 1.0},
# ]
# Create an instance of MultiLoss
# multi_loss = MultiLoss(loss_configs)