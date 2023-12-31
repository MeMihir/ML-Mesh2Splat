import numpy as np
from plyfile import PlyData, PlyElement

def save_numpy_array_to_ply(numpy_array:np.ndarray, file_path:str):
    print(numpy_array.shape)
    numpy_array = numpy_array.transpose(1,0)
    print(numpy_array.shape)
    print(numpy_array[0,:])
    
    # Save the numpy array
    np.save(file_path+'.npy', numpy_array)


    # Define the structured array with field names and data types
    dtype = np.dtype([
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
        ('f_dc_0', 'f4'),
        ('f_dc_1', 'f4'),
        ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'),
        ('scale_1', 'f4'),
        ('scale_2', 'f4'),
        ('rot_0', 'f4'),
        ('rot_1', 'f4'),
        ('rot_2', 'f4'),
        ('rot_3', 'f4')
    ])

    # Create a structured array using the provided numpy array and dtype
    structured_array = np.core.records.fromarrays(numpy_array.T, dtype=dtype)

    # Create the PlyElement manually
    vertex_element = PlyElement.describe(structured_array, 'vertex')

    # Save the PLY file
    with open(file_path+'.ply', 'wb') as f:
        PlyData([vertex_element], text=False, byte_order='<').write(f)
    

def test():
    array = np.load("data/preprocessed/train/anya_splat.npy")
    array=array.transpose(1, 0)
    save_numpy_array_to_ply(array, "data/output/anya")

if __name__ == "__main__":
    test()