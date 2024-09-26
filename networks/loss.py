import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_compute(generator):
    action_criterion = Criterion()
    arg1_criterion = Criterion()
    arg2_criterion = Criterion()
    loss_func = LossCompute(generator, action_criterion, arg1_criterion, arg2_criterion)
    return loss_func


def mid_loss_compute(outputs, targets):
    if outputs is None:
        return torch.tensor(0.0), (None, None, None, None)
    pred_q, pred_s, pred_v, pred_o = outputs
    targ_q, targ_s, targ_v, targ_o = targets
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_q, targ_q) + criterion(pred_s, targ_s) + criterion(pred_v, targ_v) + criterion(pred_o, targ_o)
    return loss, (pred_q.argmax(dim=-1), pred_s.argmax(dim=-1), pred_v.argmax(dim=-1), pred_o.argmax(dim=-1))


def obj_loss_compute(outputs, targets):
    if outputs is None:
        return torch.tensor(0.0), None
    batch_loss = 0
    criterion = nn.CrossEntropyLoss()
    for i, target in enumerate(targets):            # todo: why not apply CE to a whole batch
        batch_loss += criterion(outputs[i], target)
    batch_loss /= targets.shape[0]
    return batch_loss, outputs.argmax(dim=-1)


class Criterion(nn.Module):
    """
    Compute loss for predicting action types
    """
    def __init__(self, ignore_index=-100):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, targ):   # pred: B x T or B x N, targ: B
        loss = self.criterion(pred, targ)
        return loss


class LossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, action_criterion, arg1_criterion, arg2_criterion):
        self.generator = generator
        self.action_criterion = action_criterion
        self.arg1_criterion = arg1_criterion
        self.arg2_criterion = arg2_criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, x, y, applicable_actions):
        act_x, arg1_x, arg2_x = self.generator(x)
        device = act_x.device
        shape = [act_x.shape[0], act_x.shape[1], arg1_x.shape[1], arg2_x.shape[1]]
        # applicable_actions is a batch of choices
        prediction = torch.zeros(shape).to(device)
        prediction += act_x.unsqueeze(2).unsqueeze(3)
        prediction += arg1_x.unsqueeze(1).unsqueeze(3)
        prediction += arg2_x.unsqueeze(1).unsqueeze(2)
        batch_loss = 0
        outputs = list()
        for batch_idx, choices in enumerate(applicable_actions):
            target_idx = y[batch_idx]
            masked_pred = torch.zeros(len(choices)).to(device)
            for idx, choice in enumerate(choices):
                masked_pred[idx] = prediction[batch_idx, choice[0], choice[1], choice[2]]   # log(prob)
            outputs.append(F.softmax(masked_pred, dim=-1))
            batch_loss += self.criterion(masked_pred, target_idx)
        batch_loss /= shape[0]
        return batch_loss, (act_x, arg1_x, arg2_x), outputs
