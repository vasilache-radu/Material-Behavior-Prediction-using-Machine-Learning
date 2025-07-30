import numpy as np
import scipy
from scipy.stats import norm
import time

def getMeshgrid(n, x_end=1, y_end=1):
    """Gives Back a nxn regular Meshgrid for the midpoints of the nxn cells in the defined area of the I quadrant"""
    dx = x_end/n
    x = np.linspace(dx/2, x_end-dx/2, n)
    y = np.linspace(dx/2, y_end-dx/2, n)
    
    mesh = np.meshgrid(x, y)
    return mesh

    
def getDistances(xx, yy):
    x_cords = xx.flatten()  # 1D array of x_cords for all nodes
    y_cords = yy.flatten()  # 1D array of y_cords for all nodes
    # Now create square matrices (i,j) holding the x/y distance of nodes i & j
    xx_cords = np.tile(x_cords, (len(x_cords), 1))
    yy_cords = np.tile(y_cords, (len(y_cords), 1))
    xx_distances = np.abs(xx_cords - xx_cords.T)
    yy_distances = np.abs(yy_cords - yy_cords.T)

    return xx_distances, yy_distances

def createCovarianceMatrix(xx_distances, yy_distances, A_e, ar, sigma, rel_corr):
    """
    Creates the Covariance Matrix for use in a multivariate Gaussian RNG to
    create Images of blobs of 2 different phases on a 2D rectangle
    """
    lx = np.pi / (A_e * ar)
    ly = (np.pi * ar) / A_e
    d = np.log(1 / rel_corr)
    cov = (sigma ** 2) * np.exp(-d * (lx * xx_distances ** 2 + ly * yy_distances ** 2))

    # Why the offset?
    eps = 1e-13
    offset = np.diag(np.repeat(eps,cov.shape[0]))

    # cov = cov + offset

    return cov

def createRandomVector(mean, cov):

    # approach 1
    ranVector = np.random.multivariate_normal(mean, cov)

    # approach 2
    # ranVector = mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

    return ranVector

def generateImageArray(ranVector, cutoff):
    img=np.zeros(len(ranVector))
    img = np.where(ranVector > cutoff, 1, 0)
    return img

def generate_image(D=10, sigma=1, A_ellipse=0.25, a_r=1, rel_cor=0.01, vol_frac=0.5):
    # Global Parameters:
    D = D   # Dimension of the image    
    sigma = sigma
    A_ellipse = A_ellipse  # Area of the ellipse
    ar = a_r  # AxisRatio of the ellipse's axes rx/ry
    rel_cor = rel_cor  # relative value of correlation along ellipse boundary
    vol_frac = vol_frac  # fraction of the volume that will be filled with phase 1 in a mean sense
    z_cutoff = 0  # cutoff value fixed
    mu = z_cutoff - sigma * norm.ppf(vol_frac)  # mean to achieve vol_frac of phase 1

    xx, yy = getMeshgrid(D)

    start = time.process_time()
    xx_dist, yy_dist = getDistances(xx, yy)
    timeDistances= time.process_time()-start

    start = time.process_time()
    cov = createCovarianceMatrix(xx_dist, yy_dist, A_ellipse, ar, sigma, rel_cor)
    timeCovMatrix= time.process_time() - start

    mean = mu * np.ones(D ** 2)

    start = time.process_time()
    ran = createRandomVector(mean, cov)
    timeRandVectGen = time.process_time()-start

    start = time.process_time()
    img = generateImageArray(ran, z_cutoff)
    timeImgGen = time.process_time()-start

    def getTime():
        print('Time for distances: ', timeDistances)
        print('Time for Covariance Matrix: ', timeCovMatrix)
        print('Time for generating random vector: ', timeRandVectGen)
        print('Time for Image Generation: ', timeImgGen)

    return ran, img




    

