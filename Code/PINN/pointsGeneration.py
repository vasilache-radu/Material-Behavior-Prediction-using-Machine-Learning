import numpy as np

class Point():
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type # "inner", "boundary"
        self.E = 0
        self.nu = 0
        self.normalVector = 0
    
def locationImage(x, y, D):
    j = np.trunc(x*D).astype(int)
    j = np.where(j != D, j, j-1)

    i = np.trunc((1-y) * D).astype(int)
    i = np.where( i != D, i, i-1)
    return i, j

def getE(image, D, x, y, E_1, E_2):
    image = image.reshape(D, D)
    i, j = locationImage(x, y, D)
    if(j==D):
            j=j-1
    if(i==D):
        i=i-1
    if(image[i][j] == 1):
        E = E_1
    else:
        E = E_2

import numpy as np

def generateInnerPoints2(image, D, N, E_1=1.0, contrast_ratio=2, nu=0.3):
    N = int(np.sqrt(N))

    E_1 = E_1  # E-Modul
    nu = nu  # Poisson's number
    E_2 = contrast_ratio * E_1

    image = image.reshape(D, D)
    num_points = N * N  # Total number of points in a grid of size N x N

    # Generate linearly spaced points from 0.0001 to 1 (excluding 1)
    x = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    y = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    xy = np.meshgrid(x, y)

    # Reshape meshgrid arrays into column vectors
    x_vector = xy[0].reshape(-1, 1)
    y_vector = xy[1].reshape(-1, 1)

    # Stack and squeeze to create an array of points
    Points = np.hstack((x_vector, y_vector))

    innerPoints = []
    for index in range(num_points):
        point = Points[index]
        x = point[0]
        y = point[1]
        i = np.trunc((1 - y) * D).astype(int)
        j = np.trunc(x * D).astype(int)
        
        # Ensure indices are within valid range
        i = np.clip(i, 0, D - 1)
        j = np.clip(j, 0, D - 1)

        if image[i][j] == 1:
            E = E_1
        else:
            E = E_2
            
        innerPoint = Point(x, y, "inner")
        innerPoint.E = E
        innerPoint.nu = nu
        innerPoints.append(innerPoint)

    return innerPoints

    
def generateInnerPoints(image, D, N, E_1=1., contrast_ratio=2, nu=0.3):

    E_1 = E_1  # E-Modul
    nu = nu  # Poisson's number
    E_2 = contrast_ratio * E_1

    image = image.reshape(D, D)
    num_points = N

    x_vector = np.random.rand(num_points)
    y_vector = np.random.rand(num_points)
    j_vector = np.trunc(x_vector*D).astype(int)
    i_vector = np.trunc((1-y_vector) * D).astype(int)

    innerPoints = []
    for index in range(num_points):
        i = i_vector[index]
        j = j_vector[index]
        x = x_vector[index]
        y = y_vector[index]
        if(image[i][j] == 1):
            E = E_1
        else:
            E = E_2
        innerPoint = Point(x, y, "inner")
        innerPoint.E=E
        innerPoint.nu = nu
        innerPoints.append(innerPoint)

    return innerPoints

def generateBoundaryPoints(image, D, N, E_1=1., contrast_ratio=2, nu=0.3):
    
    N = int(N//4)
    
    num_points = 4*N
    boundaryPoints = []
    normalVectors = []
    # x=0, y - (0, 1) - Left wall
    x = np.zeros((N, ))
    y = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    n = np.array([(-1, 0) for _ in range(N)])
    boundaryPoints.extend (np.column_stack((x, y)))
    normalVectors.extend (n)

    # x=1 , y = (0, 1) - Right wall
    x = np.ones((N, ))
    y = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    n = np.array([(1, 0) for _ in range(N)])
    boundaryPoints.extend (np.column_stack((x, y)))
    normalVectors.extend (n)

    # y=0, x = (0, 1) - Bottom wall
    x = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    y = np.zeros(N)
    n = np.array([(0, -1) for _ in range(N)])
    boundaryPoints.extend (np.column_stack((x, y)))
    normalVectors.extend (n)


    # y=1, x = (0, 1) - Top wall
    x = np.linspace(0, 1, N, endpoint=False) + 1/(2*N)
    y = np.ones(N)
    n = np.array([(0, 1) for _ in range(N)])
    boundaryPoints.extend (np.column_stack((x, y)))
    normalVectors.extend (n)

    # For the moment I have only assigned the normal vector to the boundary points
    # Possibly in the future I will also need the respective E.

    E_1 = E_1  # E-Modul
    nu = nu  # Poisson's number
    E_2 = contrast_ratio * E_1

    image = image.reshape(D, D)
    
    bPoints =[]
    for index in range(num_points):
        point = boundaryPoints[index]
        normalVector = normalVectors[index]

        boundaryPoint = Point(point[0], point[1], "boundary")
        boundaryPoint.normalVector = normalVector

        j = np.trunc(boundaryPoints[index][0]*D).astype(int)
        if(j==D):
            j=j-1
        i = np.trunc((1-boundaryPoints[index][1]) * D).astype(int)
        if(i==D):
            i=i-1
        if(image[i][j] == 1):
            E = E_1
        else:
            E = E_2
        
        boundaryPoint.E = E
        boundaryPoint.nu = nu
        
        bPoints.append(boundaryPoint)

    

    return bPoints