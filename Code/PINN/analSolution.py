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