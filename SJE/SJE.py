from __future__ import print_function
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import util
import classifier1
import classifier2

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')

parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

# training options
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--nclass_all', type=int, default=50, help='number of classes for GZSL')
parser.add_argument('--calibrated_stacking', type=float, default=False,  help='calibrated stacking')


parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--pretrained', default=False, help='folder to load the pretrained model checkpoints')
parser.add_argument('--train_id', type=int, default=0)

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
opt.test_seen_label = data.test_seen_label

cls_save_dir = os.path.join(opt.outf, "{}_{}.pth".format(opt.train_id, "cls"))
best_unseen_acc = 0
best_seen_acc = 0
best_H = 0


if opt.gzsl:
    train_X = data.train_feature
    train_Y = data.train_label

    nclass = opt.nclass_all
    if opt.dataset == 'AWA2':
        cls = classifier2.CLASSIFIER(train_X, util.map_label(train_Y, data.seenclasses), data,
                                     nclass, opt, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch,
                                     opt.batch_size, True)
    else:
        cls = classifier1.CLASSIFIER(train_X, util.map_label(train_Y, data.seenclasses), data,
                                     nclass, opt, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch,
                                     opt.batch_size, True)


else:
    train_X = data.train_feature
    train_Y = data.train_label
    if opt.dataset == 'AWA2' or opt.dataset == 'AWA1' or opt.dataset == 'FLO':
        cls = classifier2.CLASSIFIER(train_X, util.map_label(train_Y, data.seenclasses), data,
                                     data.unseenclasses.size(0), opt, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch,
                                     opt.batch_size,
                                     False)
    else:
        cls = classifier1.CLASSIFIER(train_X, util.map_label(train_Y, data.seenclasses), data,
                                     data.unseenclasses.size(0), opt, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch,
                                     opt.batch_size,
                                     False)

    acc = cls.acc