import argparse
import os
from data_utils.MeshSplatDataset import MeshSplatDataset
from data_utils import preprocess, postprocessing
from model.mesh2splat import Mesh2Splat
import torch
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name [default: pointnet2]')
    parser.add_argument('--weights_dir', type=str, default='./weights/pointnet2.pth', help='model name [default: pointnet2]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--device', type=str, default=None, help='GPU to use [default: none]')
    parser.add_argument('--output_dir', type=str, default='./data/output', help='Log path [default: None]')
    parser.add_argument('--npoint', type=int, default=16384, help='Point Number [default: 16384]')
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else torch.device(args.device)

    '''CREATE DIR'''
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path(args.weights_dir)
    checkpoints_dir.mkdir(exist_ok=True)

    root = 'data/preprocessed'
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print('preprocess data ...')
    preprocess.preprocess_data(1, args.npoint)
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

    try:
        print(os.path.join(checkpoints_dir, args.model + '.pth'))
        checkpoint = torch.load(os.path.join(checkpoints_dir, args.model + '.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(e)

    model = model.eval()

    for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        pred, _ = model(points)
        postprocessing.save_numpy_array_to_ply(pred, os.path.join(args.output_dir, str(i) + '.ply'))
        
        pred = pred.reshape(-1, 17)
        target = target.reshape(-1, 17)

if __name__ == '__main__':
    args = parse_args()
    main(args)
