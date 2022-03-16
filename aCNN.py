import numpy as np
import math as mt
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os.path
import glob



####################################################################

def resize(image, half_grid_size1, half_grid_size2 = 0):
    # Simulate the resized image
    # Input : image with size (h, w), half size of convolution filter
    half_grid_size2 = half_grid_size1
    return image[half_grid_size:image.shape[0]-half_grid_size1, half_grid_size:image.shape[1]-half_grid_size2]

def grid(half_grid_size, device):
    # Compute the index for u and v direction for a grid
    # Input : int representing the half size of the convolution filter
    # Output : index for the 2D filter. ex : -1, 0, 1... for u, -1, -1, -1... for v

    u_local = torch.arange(-half_grid_size, half_grid_size+1).to(device)
    v_local = u_local
    patchU, patchV = torch.meshgrid(u_local, v_local)
    dir_u_index = torch.reshape(patchV, (-1,)).to(device)
    dir_v_index = torch.reshape(patchU, (-1,)).to(device)
    return dir_u_index, dir_v_index

def Neighbors2D(depth, conv_filter = 3, dilation =1, device= device):
    # Compute 2D neighbors for each pixel
    # Input : depth image matrix with size (b , h, w), convolution filter with size n, stand for (n, n)
    # Outputs : Matrix with size (3, h, w, n x n)
    #           Channel n X n (shape of conv_filter) corresponding to the nb of neighborhoods
    #           For each channel, it contains 3 - dimension information about the position u, v and Z
    
    half_grid_size = int(conv_filter / 2) 


    height = depth.shape[1]
    width = depth.shape[2]
    feature_map_height = height - conv_filter +1
    feature_map_width = width - conv_filter + 1
    dir_u_index, dir_v_index = grid(half_grid_size, device)

    new  = torch.zeros(depth.shape[0], conv_filter**2, feature_map_height, feature_map_width,  3 ).to(device)
    coord_V = torch.arange(half_grid_size + dir_v_index[0], half_grid_size + feature_map_height + dir_v_index[0]).to(device) #deplaced coord for the neighborhood
    coord_U = torch.arange(half_grid_size + dir_u_index[0], half_grid_size + feature_map_width + dir_u_index[0]).to(device)
    test_UU, test_VV = torch.meshgrid(coord_U, coord_V)
    torch_UU = test_UU.T.to(device)
    torch_VV = test_VV.T.to(device)
    for i in range(conv_filter) : 
        new[:, i::3,:,:,0] = torch_UU + i
        new[:, i*3:(i +1)*3,:,:,1] = torch_VV + i
        
    new[..., 2] = depth[:, new[...,1].long(), new[...,0].long()][:,0,...].to(device)
    new = new.permute(4,2,3,1,0).to(device)
    
    U_disp = torch.arange(half_grid_size, half_grid_size + feature_map_width).to(device)
    V_disp = torch.arange(half_grid_size, half_grid_size + feature_map_height).to(device)
    mat_coord_U_disp, mat_coord_V_disp = torch.meshgrid(U_disp, V_disp)
    mat_coord_U_disp = mat_coord_U_disp.T.to(device)
    mat_coord_V_disp = mat_coord_V_disp.T.to(device)
    centerFilterPosit = torch.zeros(2,  feature_map_height, feature_map_width).to(device)
    centerFilterPosit[0,:,:] = mat_coord_U_disp
    centerFilterPosit[1,:,:] = mat_coord_V_disp
    dir_u_disp = dilation * dir_u_index
    dir_v_disp = dilation * dir_v_index
    gridReg = torch.zeros(len(dir_u_disp), 2).to(device)
    gridReg[:,0] = dir_u_disp
    gridReg[:,1] = dir_v_disp
    allmap = torch.zeros(feature_map_height, feature_map_width, gridReg.shape[0], gridReg.shape[1]).to(device) + gridReg
    DilatedPosit = centerFilterPosit + allmap.permute(2,3,0,1)                      
    return new , DilatedPosit.permute(2,3,0,1)


def cameraParams():

    K = torch.tensor([[575.8157348632812, 0.0, 250],[0.0, 575.8157348632812, 250],[0.0, 0.0, 1.0]], dtype=torch.double)
    fw = K[0,0] # rapport f/ro_w
    fh = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    # fw=320
    # fh=320
    # u0=320
    # v0=240
    mtx = torch.tensor([[fw, 0.0, u0],[0.0, fh, v0],[0.0, 0.0, 1.0]])
    
    return fw, fh, u0, v0 , mtx


def backProjection(Neighbors2d_posit, device):
    # Compute 3D point cloud of the neighbor for each pixel
    # Input : 2d neighborhood positions with size (3, h, w, n x n, batch)
    # Output: 3d positions (3, h, w, n x n, batch)
    
    fw, fh, u0, v0, _ = cameraParams()
    fw = fw.to(device)
    fh = fh.to(device)
    u0 = u0.to(device)
    v0 = v0.to(device)

    u = Neighbors2d_posit[0,:].to(device)
    v = Neighbors2d_posit[1,:].to(device)
    z = Neighbors2d_posit[2,:].to(device)
    
    Neighbor2dPosit_2_3d = torch.zeros(Neighbors2d_posit.shape, dtype=torch.double).to(device)
    
    Neighbor2dPosit_2_3d[0,:] = (u-u0)/fw * z # X
    Neighbor2dPosit_2_3d[1,:] = (v-v0)/fh * z # Y
    Neighbor2dPosit_2_3d[2,:] = z             # Z
    return Neighbor2dPosit_2_3d


def compute_plane(Neighbors2dPosit_2_3d, device):
    Neighbors2dPosit_2_3d = Neighbors2dPosit_2_3d.to(device)
    # Compute the normal (a, b, c, d) of the associated plane (least square) for each set of neighborhoods :  ax+by+cz+d = 0(supposed to be 1)
    # Input : 3D positions (3, h, w, n x n, batch)
    # Output : normal (batch, h, w, 4, 1)
    nbBatch = Neighbors2dPosit_2_3d.shape[-1]
    height = Neighbors2dPosit_2_3d.shape[1] # la hauteur de la matrice contenant les points_3D (matrice reduite)
    width = Neighbors2dPosit_2_3d.shape[2]
    nbDePoints = Neighbors2dPosit_2_3d.shape[3]


    A = torch.zeros((nbBatch, height, width, nbDePoints, 3), dtype=torch.double).to(device) # 2 colonnes
    A[:,:,:,:,0] = Neighbors2dPosit_2_3d[0,:].permute(3, 0, 1, 2)
    A[:,:,:,:,1] = Neighbors2dPosit_2_3d[1,:].permute(3, 0, 1, 2)
    A[:,:,:,:,2] = 1
    b = torch.zeros((nbBatch, height, width, nbDePoints, 1), dtype=torch.double).to(device) # 1 colonne
    b[:,:,:,:,0] = - Neighbors2dPosit_2_3d[2,:].permute(3, 0, 1, 2)



    A_transpose = A.permute(0,1,2,4,3) # A[:,:,:,i]  --- A_transpose[:,:,i,:]
    ata = torch.matmul(A_transpose, A).to(device)
    Atranspose_x_A_inv = torch.inverse(ata).to(device)
    pseudoInvA = torch.matmul(Atranspose_x_A_inv, A_transpose).to(device)
    abcd = torch.ones((nbBatch, pseudoInvA.shape[1], pseudoInvA.shape[2], pseudoInvA.shape[3]+1, 1), dtype=torch.double).to(device)
    abcd[:,:,:,:2,:] = torch.matmul(pseudoInvA, b)[:,:,:,:2,:].to(device)
    abcd[:,:,:,-1,:] = torch.matmul(pseudoInvA, b)[:,:,:,-1,:].to(device)

    return abcd



def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )
    
def isect_line_plane_v3_4d(p1, plane, p0 = np.array([0,0,0]) ):
    # Compute the intersection of a 3D line and 3D plane
    
    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(plane, u)

    # Calculate a point on the plane
    # (divide can be omitted for unit hessian-normal form).
    p_co = mul_v3_fl(plane, -plane[3] / len_squared_v3(plane))

    w = sub_v3v3(p0, p_co)
    fac = -dot_v3v3(plane, w) / dot
    u = mul_v3_fl(u, fac)
    return add_v3v3(p0, u)
    
    
def projection_Neighbor_2_3DPlane(Neighbors2dPosit_2_3d, abcd):
    # Compute the projection of 3D points on the least square plane
    # Input : 3D point cloud with size (3, h, w, n x n, batch), normal with size (batch, h, w, 4, 1)
    # Output : 3D projected point on the plane with size (3, batch, h, w, n x n)
    
    reshapeNeighbors2dPosit_2_3d = Neighbors2dPosit_2_3d.permute(0,4,1,2,3)
    plane = abcd[:,:,:,:,0].permute(3,0,1,2)
    NeighborsOn3dPlane = torch.zeros(reshapeNeighbors2dPosit_2_3d.shape , dtype=torch.double)
    for i in range(reshapeNeighbors2dPosit_2_3d.shape[-1]):
        points = isect_line_plane_v3_4d(reshapeNeighbors2dPosit_2_3d[:,:,:,:,i], plane)
        NeighborsOn3dPlane[0,:,:,:,i] = points[0]
        NeighborsOn3dPlane[1,:,:,:,i] = points[1]
        NeighborsOn3dPlane[2,:,:,:,i] = points[2]
        
    return NeighborsOn3dPlane
    

def uv3DPlane(NeighborsOn3dPlane, conv_filter, device):
    # Estimate the associated u and v for the projected point cloud
    # Input : 3D projected point on the plane with size (3, batch, h, w, n x n), convolution filter with size (n x n)
    # Output : u 3d with size (3, batch, h, w, 1), v 3d with size (3, batch, h, w, 1)
     
    sizeConv = conv_filter
    u_candidates = NeighborsOn3dPlane[:,:,:,:,int((sizeConv-1)/2)::sizeConv].to(device)
    u_mean = torch.mean(torch.tensor(np.diff(u_candidates.cpu().data.numpy()), dtype = torch.double), axis = -1).to(device)
    u_norm = (u_mean / torch.norm(u_mean, dim= 0))
    u = torch.zeros(u_norm.shape[0],u_norm.shape[1],u_norm.shape[2],u_norm.shape[3], 1, dtype=torch.double).to(device)
    u[:,:,:,:,0] = u_norm
    
    v_candidates = NeighborsOn3dPlane[:,:,:,:,int((NeighborsOn3dPlane.shape[-1] - sizeConv)/2) : int((NeighborsOn3dPlane.shape[-1] + sizeConv)/2)].to(device)
    v_mean = torch.mean(torch.tensor(np.diff(v_candidates.cpu().data.numpy()), dtype = torch.double), axis = -1).to(device)
    v_norm = v_mean / torch.norm(v_mean, dim= 0)
    v = torch.zeros(u.shape, dtype=torch.double).to(device)
    v[:,:,:,:,0] = v_norm
    
    return u, v



def grid3D(NeighborsOn3dPlane, conv_filter, u_proj, v_proj, DepthCentralPixel, bol =False, dilation = 1, device = device):
    # Compute the 3D grid for the convolution
    # Input : 3d point cloud with size (3, batch, h, w, n x n), convolution filter with size (n x n), u 3d vector with size (3, batch, h, w), v 3d vector with size (3, batch, h, w), associated depth of the central point on the image
    # Output : 3d regular grid on a local plane, size (n x n, 3, batch, h, w)
    NeighborsOn3dPlane = NeighborsOn3dPlane.to(device)
    u_proj = u_proj.to(device)
    v_proj = v_proj.to(device)
    fw, fh, _, _ , _= cameraParams()

    point_3D_central = NeighborsOn3dPlane[:,:,:,:,int((conv_filter**2)/2)].to(device) # pt central est situe au milieu de la derniere dimension

    fv_indices, fu_indices = grid(int(conv_filter/2))

    gridReg = torch.zeros((len(fv_indices), 2), dtype=torch.double).to(device)
    gridReg[:,0] = torch.tensor(fu_indices, dtype = torch.double)
    gridReg[:,1] = torch.tensor(fv_indices, dtype = torch.double)
    if bol :
        scale = NeighborsOn3dPlane[2,:,:,:,int((conv_filter**2)/2)].permute(1,2,0)/DepthCentralPixel
        scale[scale < 1] = 1
        scale = scale.to(device)
        u_scale = (dilation * DepthCentralPixel * scale * u_proj).permute(1,0,2,3,4) / fw
        v_scale = (dilation * DepthCentralPixel * scale * v_proj).permute(1,0,2,3,4) / fh
        # scale the norm, 40 is the nb of pixel when parallax
    else :  
        u_scale = dilation * u_proj.permute(1,0,2,3,4) * DepthCentralPixel / fw
        v_scale = dilation * v_proj.permute(1,0,2,3,4) * DepthCentralPixel / fh
    # scale the norm, 40 is the nb of pixel when parallax
    
    # compute the norm for u3d and v3d. By convention, the depth of the central pixel is used. 
    # We supposed there is virtual frontal-parallel plane which is used for the compute

    Directions = torch.cat((u_scale, v_scale), axis=0).to(device)
    grid_3D = point_3D_central + torch.einsum('ik,kj...->ij...', 1 * gridReg, Directions * 1).permute(0,1,4,2,3)
    
    return grid_3D, point_3D_central


def GridProjection(grid_3D, device):
    # Compute the projection of 3d grid on the image
    # Input : 3d grid with size (n x n, 3, batch, h, w) 
    # Output : 2d projection with size (n x n, 2, batch, h, w)

    fw, fh, u0, v0, _ = cameraParams()
    fw = fw.to(device)
    fh = fh.to(device)
    u0 = u0.to(device)
    v0 = v0.to(device)
    grid_3D = grid_3D.to(device)
    X = grid_3D[:,0,:,:,:].to(device)
    Y = grid_3D[:,1,:,:,:].to(device)
    Z = grid_3D[:,2,:,:,:].to(device)
    
    grid_2D = torch.zeros((grid_3D.shape[0], 2, grid_3D.shape[2], grid_3D.shape[3], grid_3D.shape[4]), dtype=torch.double).to(device)
    grid_2D[:,1,:] = X*fw/Z + u0 # u
    grid_2D[:,0,:] = Y*fh/Z + v0 # v
    return grid_2D

def takeMiddleShape(offset, ori_size, conv):
    difference = int((ori_size - conv)/2)
    new = torch.zeros(offset.shape[0], 2 * conv**2, offset.shape[2], offset.shape[3])
    for i in range(conv):
        new[:, 2*i*conv : 2*(i+1)*conv, ...] = offset[:, 2*((i+1)*ori_size + difference) : 2*((i+1)*ori_size + difference + conv), ...]
    new = F.pad(new, (difference, difference, difference, difference), "constant", 0)
    return new

def projRealPlane(Neighbors2dPosit_2_3d, abcd, conv_filter, dilation = 1, device = device):
    Neighbors2dPosit_2_3d = Neighbors2dPosit_2_3d.permute(0,4,1,2,3).to(device)
    nbPoints = Neighbors2dPosit_2_3d.shape[-1]
    positionCenter = int(nbPoints/2)
    DepthCentralPixel = Neighbors2dPosit_2_3d[2, :, :, :, positionCenter].mean(1).mean(1).to(device)
    u_3d , v_3d = orthoProjVect(abcd, device)
    proj,_ = grid3D(Neighbors2dPosit_2_3d, conv_filter, u_3d, v_3d, DepthCentralPixel, bol = True, dilation = dilation, device = device)
    return proj


def orthoProjVect(abcd, device):
    zu = - abcd[..., 0, 0].to(device)
    u_3d = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    v_3d = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    z_3d = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    z_3d[2,: ]= 1
   
   
   
    u_3d[0,:] = 1
    u_3d[2,:] = zu.permute(1,2,0)
    u_3d = u_3d/ torch.norm(u_3d, dim = 0)
   
    abc = abcd[..., :3, 0].permute(3,1,2,0).to(device)
    norm = (torch.norm(abc,dim =0)**2).to(device)
    u_3d = z_3d - abc/norm
    u_3d = u_3d/ torch.norm(u_3d, dim = 0)
    v_3d = torch.cross(abc, u_3d).to(device)
    v_3d = v_3d/ torch.norm(v_3d, dim = 0)
   
    return u_3d.unsqueeze(1), v_3d.unsqueeze(1)


def computeOffset(depth, conv_filter = 3, dilation = 1, device=device):
    depth = depth.to(device)
    Neighbors2d_posit, dilated = Neighbors2D(depth, conv_filter, dilation = dilation, device = device)
    Neighbors2d_posit = Neighbors2d_posit.to(device)
    dilated = dilated.to(device)
    Neighbors2dPosit_2_3d = backProjection(Neighbors2d_posit, device).to(device)
    abcd = compute_plane(Neighbors2dPosit_2_3d, device).to(device)
    grid_3D_Reg = projRealPlane(Neighbors2dPosit_2_3d, abcd, conv_filter, dilation = dilation, device = device).to(device)

    grid_2D = GridProjection(grid_3D_Reg, device).to(device)
    ori = dilated.to(device)
    inv_ori = torch.zeros(ori.shape).to(device)
    inv_ori[..., 0] = ori[..., 1]
    inv_ori[..., 1] = ori[..., 0]
    adapted = grid_2D.permute(2,3,4,0,1).to(device)
    inv_adapted = torch.zeros(adapted.shape).to(device)
    
    for i in range(conv_filter):
        inv_adapted[:,:,:, i*conv_filter: (i+1)*conv_filter] = adapted[:,:,:, i::conv_filter]
    
    diff = inv_adapted - inv_ori
    offset = diff.reshape(diff.shape[0], diff.shape[1], diff.shape[2], -1).to(device)
    offset  = offset.permute(0,3,1,2).to(device)
    return offset.float()



if __name__ == "__main__":
    depth = torch.randn(1,480,640)
    offset = computeAffineTransformationGrid(depth,3)
