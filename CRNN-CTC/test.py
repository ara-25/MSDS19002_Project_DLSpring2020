from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as net
import params

parser = argparse.ArgumentParser()

parser.add_argument('-val', '--valroot', required=True, help='path to test dataset')
parser.add_argument('-mdl', '--pretrained', required=True, help='path to model')

args = parser.parse_args()

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

def data_loader():
    # val
    val_dataset = dataset.lmdbDataset(root=args.valroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return val_loader

val_loader = data_loader()

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    # crnn.apply(weights_init)
    # if params.pretrained != '':
    print('loading pretrained model from %s' % args.pretrained)
    # if params.multi_gpu:
    #     crnn = torch.nn.DataParallel(crnn)
    crnn.load_state_dict(torch.load(args.pretrained))
    
    return crnn

crnn = net_init()
print(crnn)


loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

criterion = CTCLoss()
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


if __name__ == "__main__":
    # for epoch in range(params.nepoch):
        # train_iter = iter(train_loader)
        # i = 0
        # while i < len(train_loader):
            # cost = train(crnn, criterion, optimizer, train_iter)
            # loss_avg.add(cost)
            # i += 1

            # if i % params.displayInterval == 0:
                # print('[%d/%d][%d/%d] Loss: %f' %
                    #   (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                # loss_avg.reset()

    
    val(crnn, criterion)

            # # do checkpointing
            # if i % params.saveInterval == 0:
            #     torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))





