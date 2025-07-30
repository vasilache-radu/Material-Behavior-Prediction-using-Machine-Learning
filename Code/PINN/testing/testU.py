import sys
import os

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np

from PINN.lossFunctions import getSigmafromEpsilon, checkLoss
from testBoundary import getDisplacementAnal, getSigmaAnal
from PINN.pointsGeneration import generateInnerPoints
from MicrostructureGeneration.ImageGenerator import generate_image

samples = 500
size_image = 32
_, image = generate_image(D=size_image)

lamb=1
mu=0.5
nu = lamb/ (2*lamb + mu)
E = mu * 2*(1+nu)

inner_samples = samples
innerPoints = generateInnerPoints(image, size_image, inner_samples, E, 1, nu)

innerPoints = np.array([[point.x, point.y] for point in innerPoints])
innerPoints = torch.tensor(innerPoints, dtype=torch.float32, requires_grad=True)

sigma = getSigmaAnal(innerPoints)
sigmafromEpsilon = getSigmafromEpsilon(innerPoints, getDisplacementAnal)

loss = checkLoss(sigmafromEpsilon, sigma)
print(loss)