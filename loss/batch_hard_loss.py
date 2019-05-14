import numpy as np
import torch
import numbers

from loss.common import cdist


def batch_hard_loss(anchor, positive, margin=1, metric='euclidean'):
    pids = torch.FloatTensor(np.arange(len(anchor)))
    if torch.cuda.is_available():
        pids = pids.cuda()
    return batch_hard(cdist(anchor, positive, metric=metric), pids, margin=margin)


def batch_hard(dists, pids, margin=1, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
            This can be of any type that can be compared, thus also a string.
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
    """
    # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
    # torch.equal is to check whether two tensors have the same size and elements
    # torch.eq is to computes element-wise equality
    same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
    # negative_mask = np.logical_not(same_identity_mask)

    # dists * same_identity_mask get the distance of each valid anchor-positive pair.
    furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
    # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
    closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
    diff = furthest_positive - closest_negative

    if isinstance(margin, numbers.Real):
        diff = torch.max(diff + margin, torch.zeros_like(diff))
    elif margin == 'soft':
        diff = torch.nn.Softplus()(diff)
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    if batch_precision_at_k is None:
        return torch.mean(diff)


if __name__ == '__main__':
    a = torch.Tensor([[1, 2, 3, 4], [3, 4, 7, 8], [9, 10, 21, 12]])
    b = torch.Tensor([[3, 3, 3, 4], [7, 8, 6, 8], [9, 10, 1, 2]])
    # print(cdist(a, b, metric='cityblock'))
    # print(cdist(a, b, metric='sqeuclidean'))
    # print(cdist(a, b, metric='euclidean'))
    pids = torch.FloatTensor([1, 2, 3])
    loss = batch_hard(cdist(a, b, metric='euclidean'), pids, margin=0)
    # print(torch.sum(loss) / len(a))
    print(loss)
    from loss.hardnet_loss import loss_HardNet

    loss = loss_HardNet(anchor=a, positive=b, margin=0)
    print(loss)
