from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from trackerCustom import TrackerSiamAttention


if __name__ == '__main__':
    # setup dataset
    root_dir = 'E:\got10k'
    seq_dataset = GOT10k(root_dir, subset='train')

    pair_dataset = Pairwise(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(pair_dataset, batch_size=8, shuffle=True, pin_memory=cuda, drop_last=True, num_workers=4)

    # restore checkpoint
    tracker = TrackerSiamAttention("./pretrained/siamattention/model_e45.pth")
    #tracker = TrackerSiamAttention()

    # path for saving checkpoints
    net_dir = 'pretrained/siamattention'
    if not os.path.exists(net_dir): os.makedirs(net_dir)

    # training loop
    epoch_num = 50
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            loss = tracker.step(batch, backward=True, update_lr=(step == 0))
            if step % 20 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format(epoch + 1, step + 1, len(loader), loss))
                sys.stdout.flush()

        # save checkpoint
        net_path = os.path.join(net_dir, 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
