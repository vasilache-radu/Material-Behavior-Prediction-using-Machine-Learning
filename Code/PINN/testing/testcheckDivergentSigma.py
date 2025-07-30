
import torch


def getDivergentSigma(points, sigmaFunction):  #points as a tensor with x and y
      
    sigma = sigmaFunction(points)
    sigmas = torch.unbind(sigma, dim = 1)
    
    dsigma00 = torch.autograd.grad(sigmas[0], points, torch.ones_like(sigmas[0]), create_graph=True)[0] # [dS00/dx, dS00/dy]
    dsigma01 = torch.autograd.grad(sigmas[1], points, torch.ones_like(sigmas[1]), create_graph=True)[0] # [dS01/dx, dS01/dy]
    dsigma10 = torch.autograd.grad(sigmas[2], points, torch.ones_like(sigmas[2]), create_graph=True)[0] # [dS10/dx, dS10/dy]
    dsigma11 = torch.autograd.grad(sigmas[3], points, torch.ones_like(sigmas[3]), create_graph=True)[0] # [dS11/dx, dS11/dy]

    divergent = torch.stack((dsigma00[:, 0] + dsigma01[:, 1] , dsigma10[:, 0] + dsigma11[:, 1]), dim=1)
    
    return divergent

def test_getDivergentSigma():
    # Define a simple sigma function for testing
    def sigmaFunction(points):
        x, y = points[:, 0], points[:, 1]
        sigma00 = x**2
        sigma01 = y**2
        sigma10 = x * y
        sigma11 = x + y
        return torch.stack((sigma00, sigma01, sigma10, sigma11), dim=1)
    
    # Define test points
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Expected result calculation
    expected_divergent = torch.tensor([[6.0, 3.0], [14.0, 5.0]])
    
    # Run the function
    result = getDivergentSigma(points, sigmaFunction)
    print(result)
    
    # Check if the result matches the expected output
    assert torch.allclose(result, expected_divergent), f"Expected {expected_divergent}, but got {result}"

# Run the test
# test_getDivergentSigma()

def getForceEstimationBoundary(points, sigmaFunction, normalVector):

    boundarySigma = sigmaFunction(points)
    sigmas= torch.split(boundarySigma, 2, dim=1)

    t1 = torch.sum((sigmas[0] * normalVector), dim=1)
    t2 = torch.sum((sigmas[1] * normalVector), dim=1)
    t = torch.stack( (t1, t2), dim=1)

    return t

def test_getForceEstimationBoundary():
    # Define a simple sigma function for testing
    def sigmaFunction(points):
        x, y = points[:, 0], points[:, 1]
        sigma00 = x**2
        sigma01 = y**2
        sigma10 = x * y
        sigma11 = x + y
        return torch.stack((sigma00, sigma01, sigma10, sigma11), dim=1)
    
    # Define test points and normal vector
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    normalVector = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    # Expected result calculation
    expected_force = torch.tensor([[1.0, 2.0], [16.0, 7.0]])
    
    # Run the function
    result = getForceEstimationBoundary(points, sigmaFunction, normalVector)
    
    # Check if the result matches the expected output
    assert torch.allclose(result, expected_force), f"Expected {expected_force}, but got {result}"

# Run the tests
test_getForceEstimationBoundary()