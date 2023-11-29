import argparse
import os
from data_utils.MeshSplatDataset import MeshSplatDataset
from data_utils import preprocess
from model.mesh2splat import Mesh2Splat, GaussianSplatLoss
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='mesh2splat', help='model name [default: mesh2splat]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--device', type=str, default=None, help='GPU to use [default: none]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else torch.device(args.device)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('mesh2splat')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./weights/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/preprocessed'
    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print('preprocess data ...')
    preprocess.preprocess_data(1)
    print("start loading training data ...")
    TRAIN_DATASET = MeshSplatDataset(split='train', root=root, num_points=NUM_POINT, block_size=1.0, sample_rate=1.0, transform=None, device=device)
    print("start loading test data ...")
    TEST_DATASET = MeshSplatDataset(split='test', root=root, num_points=NUM_POINT, block_size=1.0, sample_rate=1.0, transform=None, device=device)
    print(len(TRAIN_DATASET))
    print(len(TEST_DATASET))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Train data: %d, Test data: %d" % (len(trainDataLoader), len(testDataLoader)))
    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    # '''MODEL LOADING'''
    # shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    model = Mesh2Splat(17)
    criterion = GaussianSplatLoss()
    # classifier.apply(inplace_relu)

    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv2d') != -1:
    #         torch.nn.init.xavier_normal_(m.weight.data)
    #         torch.nn.init.constant_(m.bias.data, 0.0)
    #     elif classname.find('Linear') != -1:
    #         torch.nn.init.xavier_normal_(m.weight.data)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(checkpoints_dir)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        # classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    
    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        model = model.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # points = points.transpose(2, 1)

            pred, _ = model(points)
            pred = pred.reshape(-1, 17)
            target = target.reshape(-1, 17)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            # pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            # correct = np.sum(pred_choice == batch_label)
            # total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        # log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        # with torch.no_grad():
        #     num_batches = len(testDataLoader)
        #     total_correct = 0
        #     total_seen = 0
        #     loss_sum = 0
        #     labelweights = np.zeros(NUM_CLASSES)
        #     total_seen_class = [0 for _ in range(NUM_CLASSES)]
        #     total_correct_class = [0 for _ in range(NUM_CLASSES)]
        #     total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        #     classifier = classifier.eval()

        #     log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
        #     for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
        #         points = points.data.numpy()
        #         points = torch.Tensor(points)
        #         points, target = points.float().cuda(), target.long().cuda()
        #         points = points.transpose(2, 1)

        #         seg_pred, trans_feat = classifier(points)
        #         pred_val = seg_pred.contiguous().cpu().data.numpy()
        #         seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

        #         batch_label = target.cpu().data.numpy()
        #         target = target.view(-1, 1)[:, 0]
        #         loss = criterion(seg_pred, target, trans_feat, weights)
        #         loss_sum += loss
        #         pred_val = np.argmax(pred_val, 2)
        #         correct = np.sum((pred_val == batch_label))
        #         total_correct += correct
        #         total_seen += (BATCH_SIZE * NUM_POINT)
        #         tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        #         labelweights += tmp

        #         for l in range(NUM_CLASSES):
        #             total_seen_class[l] += np.sum((batch_label == l))
        #             total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
        #             total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        #     labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        #     mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        #     log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        #     log_string('eval point avg class IoU: %f' % (mIoU))
        #     log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        #     log_string('eval point avg class acc: %f' % (
        #         np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

        #     iou_per_class_str = '------- IoU --------\n'
        #     for l in range(NUM_CLASSES):
        #         iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
        #             seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
        #             total_correct_class[l] / float(total_iou_deno_class[l]))

        #     log_string(iou_per_class_str)
        #     log_string('Eval mean loss: %f' % (loss_sum / num_batches))
        #     log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

        #     if mIoU >= best_iou:
        #         best_iou = mIoU
        #         logger.info('Save model...')
        #         savepath = str(checkpoints_dir) + '/best_model.pth'
        #         log_string('Saving at %s' % savepath)
        #         state = {
        #             'epoch': epoch,
        #             'class_avg_iou': mIoU,
        #             'model_state_dict': classifier.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         torch.save(state, savepath)
        #         log_string('Saving model....')
        #     log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
