import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import matplotlib 
import matplotlib.pyplot as plt
import torch
import numpy as np
import math

from MicrostructureGeneration.ImageGenerator import generate_image
from pointsGeneration import generateInnerPoints
from lossFunctions import checkLoss, getDivergentSigma
#from testBoundary import getSigmaAnal
from physicalConditions import setF

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

samples = 500
size_image = 32
_, image = generate_image(D=size_image)


lamb=1
mu=0.5
nu = lamb/ (2*lamb + 2*mu)
E = mu * 2*(1+nu)

print("Nu is:", nu)
print("E is: ", E)

inner_samples = samples
innerPoints = generateInnerPoints(image, size_image, inner_samples, E, 1, nu)

bodyForce = setF(innerPoints)
bodyForce = torch.tensor(bodyForce, dtype=torch.float32)

innerPoints = np.array([[point.x, point.y] for point in innerPoints])
innerPoints = torch.tensor(innerPoints, dtype=torch.float32, requires_grad=True)  
divergent = getDivergentSigma(innerPoints, sigmaFunction=getSigmaAnal)

loss = checkLoss(-divergent, bodyForce)
print(loss)

