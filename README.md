# ML-Mesh2Splat

## Objaverse-Splats Dataset and its Generation
We used Objaverse-XL as a source dataset that contained 2D images or 3D mesh or point cloud representations of objects. We decided to stick to chair objects only.
We pulled about 391 unique chair images from Objaverse-XL and stored them in a shared Google Drive folder. Note, that each unique chair came with 62 different images of the same chair object from different angles. For our case, we only used one of the 62 images per chair to generate its corresponding 3D Gaussian splat using DreamGaussian.

You may access our benchmark dataset at the following link: [Objaverse-Splats](https://drive.google.com/drive/folders/1XgwX3nH1Q3nNAyCscwPpYkX4gGE4LJr2?usp=sharing)

Currently, there is about 355 pairing of image-Gaussian splat chairs in the benchmark dataset. However, you may run the `DatasetGeneration.ipynb` notebook in Google Colab to generate more pairings and expand the benchmark dataset. Before executing the script on your account, go to the shared Google Drive folder above to access our benchmark dataset and create a shortcut for that folder in your My Drive. This is to enable the script to read and write from the folder on your end. 

The maximum ammount of pairings that can be achieved after running the script to its extent would be about 450 pairings since the number of source chair images is limited within the benchmark dataset. You may fetch more images of chairs through Objaverse-XL or other sources, add them to the Google Drive folder, make sure they follow the same naming and numbering conventions as the other pairings, and execute the dataset generation script again to create the corresponding Gaussian splats for the new images uploaded. Also, note that since we are only using one image from one angle to generate the Gaussian splat for a chair, the quality of the Gaussian splat is not as refined. As an extension to the `DatasetGeneration.ipynb` script, generation can be modified to take all 62 images from different angles for one chair as input and generate a more precise Gaussian splat. It was our decision to not go down this route to prevent high computational costs and long execution times.


## Acknowledgement
These major successful projects were utilized to help us with the project:

- [DreamGaussian](https://arxiv.org/abs/2309.16653), [repo](https://github.com/dreamgaussian/dreamgaussian)
- [Objaverse-XL](https://arxiv.org/abs/2307.05663), [repo](https://github.com/allenai/objaverse-xl)
- [PointNet++](https://arxiv.org/abs/1706.02413), [repo](https://github.com/charlesq34/pointnet2)
