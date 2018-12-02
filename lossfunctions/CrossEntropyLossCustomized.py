
import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomizedCrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(CustomizedCrossEntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight

    def forward(self, outputs, targets):
        # transform targets to one-hot vector
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)

        # nn.CrossEntropyLoss
        # combines nn.LogSoftmax() and nn.NLLLoss()
        outputs = self.softmax(outputs)

        self.weight = self.weight.expand_as(outputs)
        loss = -targets_onehot.float() * torch.log(outputs)

        return torch.mean(self.weight * loss)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

# define CrossEntropyLoss with weights
weight = torch.Tensor([1, 5, 10])
# define inputs, official and custom loss
outputs = torch.Tensor([[0.9, 0.5, 0.05], [0.01, 0.2, 0.7]])
targets = torch.Tensor([0, 1]).long()
criterion = nn.CrossEntropyLoss(weight=weight)
custom_criterion1 = CustomizedCrossEntropyLoss(weight=weight)
custom_criterion2 = CrossEntropyLoss2d(weight=weight)

# run metrics
loss = criterion(outputs, targets)
custom_loss1 = custom_criterion1(outputs, targets)
custom_loss2 = custom_criterion2(outputs, targets)
print ('official loss: ', loss.item())
print ('custom loss1:   ', custom_loss1.item())
print ('custom loss2:   ', custom_loss2.item())