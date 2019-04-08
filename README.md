# PyTorch PPF-FoldNet
This repo is implementation for PPF-FoldNet(https://arxiv.org/abs/1808.10322v1) in pytorch. 

## Milestone
- 3.25: Initial Version
    - Realize the network architecuture of PPF-FoldNet(but actually is very similar with FoldingNet and I have partially achieve the result of FoldingNet, the reconstruction and interpolation looks good.)
    - The input is still the point coordinates for now. And the dataset used is ShapeNet. This is the main aim of next step implementation.
    
- 4.8:
    - Change dataset to Sun3D (downloaded from 3DMatch webpage). And it is easy to extend to the whole 3DMatch dataset.
    - Input preparation is done, the input is the point pair feature. Currently I do this process before I start to train the network. For each point cloud (reconstructed by a RGB image and a depth image and a camera pose), I select 2048 interest point, and collect 1024 neighbor point for each interest point, then calculate the ppf for each pair. So the result for one point cloud is a [2048, 1024, 4] array, I save it in a `.npy` file. 
    - But here comes a questions: for input to the network, how many interest point should I choice for one point cloud? (Or for one batch input, how many point cloud should I choose? and for each point cloud how many local patches should I choose? Obviously I cannot select all the 2048 local patches for a single point cloud.) Currently I just random select 32 local patches from one point cloud, so the shape of the input to network is [bs, 32, 2048, 4]
    - It seems there is no good way to visualize the result and evaluate the performance except the 3DMatch geometry benchmark. The visualization of ppf in the PPF-FoldNet paper does not make sense to me.


## TODO List
- [x] dataset & dataloader for 3DMatch Dataset.
- [x] Input Preparation. (Point coordinates -> point pair feature)
- [ ] Evaluation. (Recall on 3DMatch Dataset)
