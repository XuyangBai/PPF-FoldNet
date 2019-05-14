import torch
import sys

from loss.common import cdist


def loss_HardNet(anchor, positive, anchor_swap=False, anchor_ave=False, \
                 margin=1.0, batch_reduce='min', loss_type="triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = cdist(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1)))
    # eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag, 1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
            min_neg = torch.min(min_neg, min_neg2)
        if False:
            dist_matrix_a = distance_matrix_vector(anchor, anchor) + eps
            dist_matrix_p = distance_matrix_vector(positive, positive) + eps
            dist_without_min_on_diag_a = dist_matrix_a + eye * 10
            dist_without_min_on_diag_p = dist_matrix_p + eye * 10
            min_neg_a = torch.min(dist_without_min_on_diag_a, 1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p, 0)[0])
            min_neg_3 = torch.min(min_neg_p, min_neg_a)
            min_neg = torch.min(min_neg, min_neg_3)
            print(min_neg_a)
            print(min_neg_p)
            print(min_neg_3)
            print(min_neg)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1, 1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1, 1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1, 1)
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long())
        # idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1, 1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1, idxs.view(-1, 1))
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else:
        print('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log(exp_pos / exp_den)
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


if __name__ == '__main__':
    a = torch.Tensor([[1, 2, 3, 4], [3, 4, 7, 8], [9, 10, 11, 12]])
    b = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 2]])
    loss = loss_HardNet(anchor=a, positive=b, margin=0)
    print(loss)
