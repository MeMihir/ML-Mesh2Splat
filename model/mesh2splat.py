import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from .pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


loss_functions = {
    'mse': F.mse_loss,
    'chamfer': chamfer_distance,
}

class Mesh2Splat(nn.Module):
    def __init__(self, points_dim):
        super(Mesh2Splat, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.convOut = nn.Conv1d(128, points_dim, 1)
        
        self._freeze_layers()

    def _freeze_layers(self):
        for name, param in self.named_parameters():
            if 'convOut' not in name:  # Exclude parameters from the last layer
                param.requires_grad = False

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.convOut(x)
        return x, l4_points


class GaussianSplatLoss(nn.Module):
    def __init__(self, config):
        super(GaussianSplatLoss, self).__init__()
        self.position_lossFn = loss_functions[config['loss_function']['position_loss']]
        self.scaling_lossFn = loss_functions[config['loss_function']['scaling_loss']]
        self.rotation_lossFn = loss_functions[config['loss_function']['rotation_loss']]
        self.opacity_lossFn = loss_functions[config['loss_function']['opacity_loss']]
        self.color_lossFn = loss_functions[config['loss_function']['color_loss']]
        
        self.position_weight = config['loss_weights']['position_weight']
        self.scaling_weight = config['loss_weights']['scaling_weight']
        self.rotation_weight = config['loss_weights']['rotation_weight']
        self.opacity_weight = config['loss_weights']['opacity_weight']
        self.color_weight = config['loss_weights']['color_weight']

    def forward(self, pred, target):
        # print('Loss pred shape: ', pred.shape)
        # print('Loss target shape: ', target.shape)

        position_loss = self.position_lossFn(pred[:,:3], target[:,:3])
        scaling_loss = self.scaling_lossFn(pred[:,10:13], target[:,10:13])
        rotation_loss = self.rotation_lossFn(pred[:,13:], target[:,13:])
        opacity_loss = self.opacity_lossFn(pred[:,9], target[:,9])
        color_loss = self.color_lossFn(pred[:,6:9], target[:,6:9])

        total_loss = position_loss[0] * self.position_weight + scaling_loss * self.scaling_weight + rotation_loss * self.rotation_weight + opacity_loss * self.opacity_weight + color_loss * self.color_weight
        return total_loss, position_loss[0], scaling_loss, rotation_loss, opacity_loss, color_loss

if __name__ == '__main__':
    model = Mesh2Splat(13)
    xyz = torch.rand(6, 9, 2048)
    print(model(xyz)[0].shape)