import numpy as np
import torch
from PINN.physicalConditions import *

def datapreparation(dataSet):
    innerPoints, boundaryPoints = dataSet
    innerPoints = np.array(innerPoints)
    boundaryPoints = np.array(boundaryPoints)

    #Random permutation of boundaryPoints
    # num_rows = boundaryPoints.shape[0]
    # print(num_rows)
    # boundary_indices = np.random.permutation(num_rows)
    # boundaryPoints = boundaryPoints[boundary_indices]

    N = boundaryPoints.shape[0]
    l=0 # left wall from 0 to N//4-1
    r= N//4 # right wall from N//4 to 2*N//4 - 1
    b = 2* N//4 # bottom wall from 2* N//4 to 3* N//4 -1
    t = 3* N//4 # top wall from 3* N//4 to end

    new_boundaryPoints = []
    i=0
    while( len(new_boundaryPoints) != len(boundaryPoints) and i < N//4):
        new_boundaryPoints.append(boundaryPoints[l+i])
        new_boundaryPoints.append(boundaryPoints[r+i])
        new_boundaryPoints.append(boundaryPoints[b+i])
        new_boundaryPoints.append(boundaryPoints[t+i])
        i = i+1
    
    new_boundaryPoints.extend(boundaryPoints[t+i: N-1])
    boundaryPoints = np.array(new_boundaryPoints)

    count = len(boundaryPoints)

    innerPointsModel = []
    innerE = []
    innerNu = []
    for point in innerPoints:
        innerPointsModel.append([point.x, point.y])
        innerE.append(point.E)
        innerNu.append(point.nu)

    boundaryPointsModel = []
    normalVector = []
    
    for point in boundaryPoints:
        boundaryPointsModel.append([point.x, point.y])
        normalVector.append(point.normalVector)

    innerE = np.array(innerE)
    innerNu = np.array(innerNu)

    innerLambda = (innerE * innerNu) / ((1 + innerNu) * (1- 2*innerNu)) # Lame's 1st parameter
    innerMu = (innerE)/ (2* (1+ innerNu)) # Lame's 2nd parameter

    f = setF(innerPoints)
    boundaryForce, maskForce = setBoundaryForce(boundaryPoints)
    boundaryDisplacement, maskDisplacement = setBoundaryDisplacement(boundaryPoints)

    masks = maskForce, maskDisplacement
    conditions = f, boundaryForce, boundaryDisplacement
    preparedPoints = innerPointsModel, boundaryPointsModel
    utils = innerLambda, innerMu, normalVector
    preparedDataSet = preparedPoints, conditions, masks, utils

    return preparedDataSet

def dataloading(preparedDataSet, batch_size):
    preparedPoints, conditions, masks, utils = preparedDataSet

    innerPoints, boundaryPoints = preparedPoints
    f, boundaryForce, boundaryDisplacement = conditions
    maskForce, maskDisplacement = masks
    innerLambda, innerMu, normalVector = utils

    innerPoints = torch.tensor(innerPoints, dtype= torch.float32, requires_grad=True)
    boundaryPoints = torch.tensor(boundaryPoints, dtype= torch.float32, requires_grad= True)

    innerLambda = torch.tensor(innerLambda, dtype=torch.float32)
    innerMu = torch.tensor(innerMu, dtype=torch.float32)
    normalVector = torch.tensor(normalVector, dtype = torch.float32)

    f = torch.tensor(f, dtype = torch.float32)
    boundaryForce = torch.tensor(boundaryForce, dtype = torch.float32)
    boundaryDisplacement = torch.tensor(boundaryDisplacement, dtype = torch.float32)

    maskForce = torch.tensor(maskForce, dtype = torch.float32)
    maskDisplacement = torch.tensor(maskDisplacement, dtype=torch.float32)

    # Set the size of the batch of the boundaryPoints
    batch_size_boundary = batch_size    
    # Calculate the number of batches for  boundaryPoints
    nr_batches = len(boundaryPoints) // batch_size_boundary

    # boundaryPoints and innerPoints have the same number of batches
    # Calculate the size of the batch of the innerPoints 
    batch_size_inner = len(innerPoints) // nr_batches

    # Get the batches for the DataSet
    inner_batches = np.array_split(innerPoints, nr_batches)
    boundary_batches = np.array_split(boundaryPoints, nr_batches)

    lambda_batches = np.array_split(innerLambda, nr_batches)
    mu_batches = np.array_split(innerMu, nr_batches)
    normalVector_batches = np.array_split(normalVector, nr_batches)

    f_batches = np.array_split(f, nr_batches)
    boundaryForce_batches = np.array_split(boundaryForce, nr_batches)
    boundaryDisplacement_batches = np.array_split(boundaryDisplacement, nr_batches)

    maskForce_batches = np.array_split(maskForce, nr_batches)
    maskDisplacement_batches = np.array_split(maskDisplacement, nr_batches)

    dataloader = list(zip(inner_batches, boundary_batches, lambda_batches, mu_batches, normalVector_batches, maskForce_batches, maskDisplacement_batches, 
                        f_batches, boundaryForce_batches, boundaryDisplacement_batches))

    return dataloader
