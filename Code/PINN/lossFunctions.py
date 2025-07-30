import torch
import numpy as np
import math

def checkLoss(estimation, condition, mask = None):
    
    loss = torch.sum((estimation - condition)**2, dim=1, keepdim=True)
    if mask==None:
        nr_samples = len(loss)
    else:
        nr_samples = torch.count_nonzero(mask)
        loss = loss * mask.reshape(mask.shape[0], 1)
    if nr_samples==0:
        return 0
    
    loss = torch.sum(loss) / nr_samples
    return loss

def getDivergentSigma(points, sigmaFunction):  #points as a tensor with x and y
      
    sigma = sigmaFunction(points)
    sigmas = torch.unbind(sigma, dim = 1)
    
    dsigma00 = torch.autograd.grad(sigmas[0], points, torch.ones_like(sigmas[0]), create_graph=True, allow_unused=True)[0] # [dS00/dx, dS00/dy]
    dsigma01 = torch.autograd.grad(sigmas[1], points, torch.ones_like(sigmas[1]), create_graph=True)[0] # [dS01/dx, dS01/dy]
    dsigma10 = torch.autograd.grad(sigmas[2], points, torch.ones_like(sigmas[2]), create_graph=True)[0] # [dS10/dx, dS10/dy]
    dsigma11 = torch.autograd.grad(sigmas[3], points, torch.ones_like(sigmas[3]), create_graph=True)[0] # [dS11/dx, dS11/dy]

    divergent = torch.stack((dsigma00[:, 0] + dsigma01[:, 1] , dsigma10[:, 0] + dsigma11[:, 1]), dim=1)
    
    return divergent

def getForceEstimationBoundary(points, sigmaFunction, normalVector):

    boundarySigma = sigmaFunction(points)
    sigmas= torch.split(boundarySigma, 2, dim=1)

    t1 = torch.sum((sigmas[0] * normalVector), dim=1)
    t2 = torch.sum((sigmas[1] * normalVector), dim=1)
    t = torch.stack( (t1, t2), dim=1)

    return t

def getDisplacementEstimationBoundary(points, displacementFunction):
    boundaryDisplacement = displacementFunction(points)

    return boundaryDisplacement

def getSigmafromEpsilon(points, displacementFunction, lamb=1, mu=0.5):  #points with x and y only
    
    innerU = displacementFunction(points)
    Us = torch.unbind(innerU, dim = 1)

    dU0 = torch.autograd.grad(Us[0], points, torch.ones_like(Us[0]), create_graph=True)[0] # [dUx/dx, dUx/dy]
    dU1 = torch.autograd.grad(Us[1], points, torch.ones_like(Us[1]), create_graph=True)[0] # [dUy/dx, dUy/dy]
    dUT = torch.stack((dU0[:, 0], dU1[:, 0], dU0[:, 1], dU1[:, 1]), dim=1)
    dU = torch.stack((dU0[:, 0], dU0[:, 1], dU1[:, 0], dU1[:, 1]), dim=1)
    divU = dU0[:, 0] + dU1[:, 1]

    divLambda =  lamb * divU
    zeroVector = torch.zeros_like(divLambda)
    firstPart = torch.stack((divLambda, zeroVector,  zeroVector, divLambda), dim=1)
    secondPart = (dU + dUT) * mu.unsqueeze(1)

    sigmafromEpsion = firstPart + secondPart

    return sigmafromEpsion
