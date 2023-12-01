import os
import glob
import numpy as np
from plyfile import PlyData, PlyElement
from .downsampling import everyNth

BASE_DIR = './'
input_path = os.path.join(BASE_DIR, 'data', 'input')
truth_path = os.path.join(BASE_DIR, 'data', 'truth')
preprop_path = os.path.join(BASE_DIR, 'data', 'preprocessed')

def collect_point_label(in_filename, truth_filename, out_filename, split='train' ,file_format='numpy', npoints=4096):
    """ preprocess point and splats
        points: N*6, each row is x,y,z,r,g,b
        splats: N*17, each row is x,y,z,normal_x,normal_y,normal_z,color_r,color_g,color_b,opacity,scale_x,scale_y,scale_z,rotation_1,rotation_2,rotation_3,rotation_4

    Args:
        in_filename: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """    
    points = np.loadtxt(os.path.join(input_path, in_filename))
    with open(os.path.join(truth_path, truth_filename), 'rb') as f:
        plydata = PlyData.read(f)
        splat = np.array(list(map(list, plydata.elements[0])))

    points = everyNth(points, npoints)
    points_min = np.amin(points, axis=0)[0:3]
    points[:, 0:3] -= points_min
    splat = everyNth(splat, npoints)
    
    if file_format=='numpy':
        np.save(os.path.join(preprop_path, split, out_filename)+'.npy', points)
        np.save(os.path.join(preprop_path, split, out_filename)+'_splat.npy', splat)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()

def data_to_obj(data,name='example.obj',no_wall=True):
    fout = open(name, 'w')
    label = data[:, -1].astype(int)
    for i in range(data.shape[0]):
        if no_wall and ((label[i] == 2) or (label[i]==0)):
            continue
        fout.write('v %f %f %f %d %d %d\n' % \
                   (data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5]))
    fout.close()

def preprocess_data(split_ratio=0.9, num_point=4096):
    # check if processed data exists
    if len(os.listdir(os.path.join(preprop_path))) > 0:
        print('preprocessed data found, skipping preprocessing')
        return

    filenames = [os.path.splitext(filename)[0] for filename in os.listdir(input_path)]
    np.random.shuffle(filenames)
    train_files = filenames[:int(len(filenames)*split_ratio)]
    test_files = filenames[int(len(filenames)*split_ratio):]
    for filename in train_files:
        try:
            out_filename = filename # anya.npy
            in_filename = filename+'.txt' # anya.txt
            true_filename = filename+'.ply' # anya.ply

            collect_point_label(in_filename, true_filename, out_filename, 'train', 'numpy', npoints=num_point)
        except Exception as e:
            print(filename, 'ERROR!!')
            print(e)
    
    for filename in test_files:
        try:
            out_filename = filename # anya.npy
            in_filename = filename+'.txt' # anya.txt
            true_filename = filename+'.ply' # anya.ply

            collect_point_label(in_filename, true_filename, out_filename, 'test', 'numpy', npoints=num_point)
        except:
            print(filename, 'ERROR!!')

if __name__ == '__main__':
    preprocess_data(1)
