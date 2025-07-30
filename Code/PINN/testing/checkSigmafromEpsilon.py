import torch


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

def test_getSigmafromEpsilon():
    # Define a simple displacement function
    def displacementFunction(points):
        return torch.stack((points[:, 0] * 2, points[:, 1] * 3), dim=1)

    # Define test points
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Expected output calculation
    lamb = torch.tensor([1.0, 1.0])
    mu = torch.tensor([0.5, 0.5])
    
    result = getSigmafromEpsilon(points, displacementFunction, lamb, mu)

# Run the test
test_getSigmafromEpsilon()