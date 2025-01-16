import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.metrics = ['total_loss']

    def forward(self, input, target, *args):
        loss = self.loss_fn(input, target)
        metrics = {'total_loss': loss.detach().cpu().item()}
        return loss, metrics