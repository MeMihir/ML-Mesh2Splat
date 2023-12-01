import numpy as np
from plyfile import PlyData, PlyElement

def save_numpy_array_to_ply(numpy_array, file_path):
    num_vertices = numpy_array.shape[0]

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
    with open(file_path, 'wb') as f:
        PlyData([vertex_element], text=False, byte_order='<').write(f)

def test():
    array = np.load("data/preprocessed/train/anya_splat.npy")
    save_numpy_array_to_ply(array, "data/output/anya.ply")

if __name__ == "__main__":
    test()
