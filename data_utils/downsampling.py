import numpy as np

def everyNth(splats, target_num_splats):
    num_splats = len(splats)

    if num_splats > target_num_splats:
        # Select every nth splat to achieve the target number
        selected_indices = np.arange(0, num_splats, num_splats // target_num_splats)[:target_num_splats]
        reduced_splats = np.array([splats[i] for i in selected_indices])
    else:
        # If already equal or fewer, no need to modify
        reduced_splats = splats

    return reduced_splats