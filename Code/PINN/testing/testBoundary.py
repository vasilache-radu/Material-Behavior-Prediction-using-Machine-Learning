import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import torch

def getSigmaAnal(points, lamb=1, mu=0.5, Q=4): #points as a tensor with x and y
    pi = torch.pi
    cos = torch.cos
    sin = torch.sin
    x = points[:, 0]
    y = points[:, 1]

    sxx = lamb*(Q*(sin(pi*x)*(y**3))-2*pi*sin(2*pi*x) * sin(pi*y))\
        -4*mu*pi*sin(2*pi*x)*sin(pi*y)
    syy = lamb*(Q*sin(pi*x)*y**3-2*pi*sin(2*pi*x)*sin(pi*y))\
        +2*mu*Q*sin(pi*x)*y**3
    sxy = mu*(cos(pi*x)*y**4*pi*Q/4+pi*cos(2*pi*x)*cos(pi*y))

    sigma = torch.stack((sxx, sxy, sxy, syy), dim=1)

    return sigma

def getDisplacementAnal(points, lamb=1, mu=0.5, Q=4):
    pi = torch.pi
    cos = torch.cos
    sin = torch.sin
    x = points[:, 0]
    y = points[:, 1]

    u = cos(2*pi*x)*sin(pi*y)
    v = Q*sin(pi*x)*y**4/4

    U = torch.stack((u, v), dim=1)

    return U


from MicrostructureGeneration.ImageGenerator import generate_image
from PINN.pointsGeneration import *
from PINN.lossFunctions import getForceEstimationBoundary, getDisplacementEstimationBoundary, checkLoss

from PINN.physicalConditions import *
samples = 4000
size_image = 32
_, image = generate_image(D=size_image)

lamb=1
mu=0.5
nu = lamb/ (2*lamb + 2*mu)
E = mu * 2*(1+nu)

boundary_samples = samples
boundaryPoints = generateBoundaryPoints(image, size_image, boundary_samples, E, 1, nu)

def getMasks(boundaryPoints):
    nr_samples = len(boundaryPoints)

    maskForce = np.ones((nr_samples, 1))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        if point.y == 0.:
            maskForce[i] = 0.
    
    maskDisplacement = np.zeros((nr_samples, 1))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        if point.y == 0.:
            maskDisplacement[i] = 1.
    
    return maskForce, maskDisplacement
def setBoundaryForce1(boundaryPoints):
    Q=4

    nr_samples = len(boundaryPoints)
    boundaryForce = np.zeros((nr_samples, 2))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        x = point.x
        y = point.y
        E = point.E
        nu = point.nu

        lamb = 1
        mu = 0.5

        if point.y == 1.:
            boundaryForce[i][0] = mu * np.pi * (-np.cos(2*np.pi*x) + Q/4*np.cos(np.pi * x))
            boundaryForce[i][1] = lamb * Q * np.sin(np.pi * x) + 2*mu*Q*np.sin(np.pi*x)
        if point.x == 1.:
            boundaryForce[i][0] = 0.
            boundaryForce[i][1] = mu * np.pi * (-Q/4* y**4 + np.cos(np.pi * y)) 
        if point.x == 0.:
            boundaryForce[i][0] = 0.
            boundaryForce[i][1] = - mu* np.pi * (Q/4* y**4 + np.cos(np.pi*y))    

    # it will return the boundaryForce conditions and the mask for the boundaryForce
    return boundaryForce
def setBoundaryDisplacement1(boundaryPoints):
    nr_samples = len(boundaryPoints)
    boundaryDisplacement = np.zeros((nr_samples, 2))

    return boundaryDisplacement

maskForce1, maskDisplacement1 = getMasks(boundaryPoints)
boundaryForce1 = setBoundaryForce1(boundaryPoints)
boundaryDisplacement1 = setBoundaryDisplacement1(boundaryPoints=boundaryPoints)

boundaryForce, maskForce = setBoundaryForce(boundaryPoints)
boundaryDisplacement, maskDisplacement = setBoundaryDisplacement(boundaryPoints)

errorMaskForce= maskForce - maskForce1 # same
errorMaskDisplacement = maskDisplacement - maskDisplacement1 # same

errorBoundaryForce = boundaryForce - boundaryForce1 # Here is a difference, but why?
errorBoundaryDisplacement = boundaryDisplacement - boundaryDisplacement1 # same


boundaryPointsModel = []
normalVector = []
for point in boundaryPoints:
    boundaryPointsModel.append([point.x, point.y])
    normalVector.append(point.normalVector)

boundaryPoints = torch.tensor(boundaryPointsModel, dtype= torch.float32, requires_grad= True)
normalVector = torch.tensor(normalVector, dtype = torch.float32)
maskForce = torch.tensor(maskForce, dtype = torch.float32)

boundaryForce = torch.tensor(boundaryForce, dtype=torch.float32)
boundaryDisplacement = torch.tensor(boundaryDisplacement, dtype=torch.float32)
maskDisplacement = torch.tensor(maskDisplacement, dtype=torch.float32)

t = getForceEstimationBoundary(boundaryPoints, getSigmaAnal, normalVector)
boundaryU = getDisplacementAnal(boundaryPoints)

LBoundary = checkLoss(t, boundaryForce, maskForce)
LBoundaryU = checkLoss(boundaryU, boundaryDisplacement, maskDisplacement)

print(LBoundary)
print(LBoundaryU)