import torch


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
        return torch.cat((points, points), dim=1)

    # Define a normal vector
    normalVector = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Define some test points
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Expected output calculation
    boundarySigma = sigmaFunction(points)
    sigmas = torch.split(boundarySigma, 2, dim=1)
    t1 = torch.sum((sigmas[0] * normalVector), dim=1)
    t2 = torch.sum((sigmas[1] * normalVector), dim=1)
    expected_output = torch.stack((t1, t2), dim=1)

    # Get the actual output from the function
    actual_output = getForceEstimationBoundary(points, sigmaFunction, normalVector)

    # Check if the actual output matches the expected output
    assert torch.allclose(actual_output, expected_output), f"Expected {expected_output}, but got {actual_output}"

# Run the test
test_getForceEstimationBoundary()