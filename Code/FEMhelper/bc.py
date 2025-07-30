# Rename it to Problem1

import numpy as np

from fenics import *

def f(lambda_, mu, x, Q=4):
    f_x =lambda_(x) * (4 * np.pi**2 * cos(2 * np.pi * x[0]) * sin(np.pi * x[1]) 
                    - np.pi * cos(np.pi * x[0]) * Q * x[1]**3) \
     + mu(x) * (9 * np.pi**2 * cos(2 * np.pi * x[0]) * sin(np.pi * x[1]) 
                  - np.pi * cos(np.pi * x[0]) * Q * x[1]**3)
    f_y = lambda_(x) * (-3 * sin(np.pi * x[0]) * Q * x[1]**2 
                    + 2 * np.pi**2 * sin(2 * np.pi * x[0]) * cos(np.pi * x[1])) \
     + mu(x) * (-6 * sin(np.pi * x[0]) * Q * x[1]**2 
                  + 2 * np.pi**2 * sin(2 * np.pi * x[0]) * cos(np.pi * x[1]) 
                  + np.pi**2 * sin(np.pi * x[0]) * Q * x[1]**4 / 4)
    
    return np.array([f_x, f_y])

def bcX0(lambda_, mu, x, Q=4):
    bcX0_x = 0.
    bcX0_y = - mu(x)* np.pi * (Q/4* x[1]**4 + np.cos(np.pi*x[1]))
    return np.array([bcX0_x, bcX0_y])

def bcX1(lambda_, mu, x, Q=4):
    bcX1_x = 0.
    bcX1_y = mu(x) * np.pi * (-Q/4* x[1]**4 + np.cos(np.pi * x[1]))
    return np.array([bcX1_x, bcX1_y]) 

def bcY0(lambda_, mu, x, Q=4):
    bcY0_x = 0.
    bcY0_y = 0.
    return np.array([bcY0_x, bcY0_y])

def bcY1(lambda_, mu, x, Q=4):
    bcY1_x = mu(x) * np.pi * (-np.cos(2*np.pi*x[0]) + Q/4*np.cos(np.pi * x[0]))
    bcY1_y = lambda_(x) * Q * np.sin(np.pi * x[0]) + 2*mu(x)*Q*np.sin(np.pi*x[0])
    return np.array([bcY1_x, bcY1_y])

