import numpy as np

def setF(innerPoints):
    x = np.array([point.x for point in innerPoints])
    y = np.array([point.y for point in innerPoints])
    innerE = np.array([point.E for point in innerPoints])
    innernu = np.array([point.nu for point in innerPoints])

    innerLambda = (innerE * innernu) / ((1 + innernu) * (1- 2*innernu)) # Lame's 1st parameter
    innerMu = (innerE)/ (2* (1+ innernu)) # Lame's 2nd parameter

    Q=4 # I set it here because it is part of how the inner forces are defined in this case, not a general thing
    fx = innerLambda * (4* np.pi**2 * np.cos(2*np.pi * x) * np.sin(np.pi*y) - np.pi * np.cos(np.pi*x) * Q * y**3) + innerMu * (9* np.pi**2 * np.cos(2*np.pi *x)* np.sin(np.pi*y) - np.pi * np.cos(np.pi*x)* Q * y**3)
    fy = innerLambda * (-3* np.sin(np.pi*x)*Q* y**2 + 2* np.pi**2 * np.sin(2*np.pi*x) * np.cos(np.pi*y)) + innerMu * (-6*np.sin(np.pi*x)*Q*y**2 + 2* np.pi**2 * np.sin(2*np.pi*x)* np.cos(np.pi*y) + np.pi**2 * np.sin(np.pi*x)*Q*y**4/4)
    f = np.stack((fx, fy), axis=1)

    return f

def setBoundaryForce(boundaryPoints):
    # Input: boundaryPoints
    # Output: Force on the boundary + whether the point has a boundaryForce
    Q=4
    nr_samples = len(boundaryPoints)

    maskForce = np.ones((nr_samples, 1))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        if point.y == 0.:
            maskForce[i] = 0.
    
    boundaryForce = np.zeros((nr_samples, 2))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        x = point.x
        y = point.y
        E = point.E
        nu = point.nu

        lamb = (E * nu) / ((1+nu) * (1-2*nu))
        mu = E / (2*(1+nu))

        if point.y == 1.:
            boundaryForce[i][0] = mu * np.pi * (-np.cos(2*np.pi*x) + Q/4*np.cos(np.pi * x))
            boundaryForce[i][1] = lamb * Q * np.sin(np.pi * x) + 2*mu*Q*np.sin(np.pi*x)
        if point.x == 1.:
            boundaryForce[i][0] = 0.
            boundaryForce[i][1] = mu * np.pi * (-Q/4* y**4 + np.cos(np.pi * y)) 
        if point.x == 0.:
            boundaryForce[i][0] = 0.
            boundaryForce[i][1] = - mu* np.pi * (Q/4* y**4 + np.cos(np.pi*y))    

    return boundaryForce, maskForce

def setBoundaryDisplacement(boundaryPoints):
    # Input: boundaryPoints
    # Output: Displacement on the boundary + whether the point has a boundaryDisplacement
   
    nr_samples = len(boundaryPoints)

    maskDisplacement = np.zeros((nr_samples, 1))
    for i in range(nr_samples):
        point = boundaryPoints[i]
        if point.y == 0.:
            maskDisplacement[i] = 1.
    
    boundaryDisplacement = np.zeros((nr_samples, 2))

    return boundaryDisplacement, maskDisplacement
