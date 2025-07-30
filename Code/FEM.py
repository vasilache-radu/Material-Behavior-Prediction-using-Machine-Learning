import numpy as np
import ufl
from ImageGenerator import generate_image
from fenics import *
from ufl import nabla_div, nabla_grad

from PINN.pointsGeneration import locationImage
from FEMhelper.bc import getF, getBCX0, getBCX1, getBCY0, getBCY1
from FEMhelper.plot import *

inquire('modelsFNO/model_test')

def getLambda(E, nu):
    return (E * nu) / ((1 + nu) * (1 - 2 * nu))  # Lame's 1st parameter

def getMu(E, nu):
    return E / (2 * (1 + nu))  # Lame's 2nd parameter

class E(UserExpression):
    def __init__(self, image, size_image, e_1, e_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = image
        self.size_image = size_image
        self.e_1 = e_1
        self.e_2 = e_2

    def eval(self, value, x):
        size_image = self.size_image
        image = self.image.reshape(size_image, size_image)
        i, j = locationImage(x[0], x[1], D=size_image)
        if image[i, j] == 1:
            value[0] = self.e_1
        else:
            value[0] = self.e_2
        

    def value_shape(self):
        # return (1,)
        return ()


# Defining the stress and the strain

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(lbd, mu, d, u):
    return lbd*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define the source term

class F(UserExpression):
    def __init__(self, lambda_, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
    
    def eval(self, value, x):
        f_val = getF(self.lambda_, self.mu, x)
        value[0] = f_val[0]  # Assign f_x to the first component
        value[1] = f_val[1]  # Assign f_y to the second component
    
    def value_shape(self):
        return (2,)

# Define the boundary conditions

class BoundaryX0(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, self.tol)
    
class BoundaryX1(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, self.tol)
class BoundaryY0(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, self.tol)
class BoundaryY1(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1, self.tol)

class BoundaryCondition(UserExpression):
    def __init__(self, lambda_, mu, bc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.bc = bc
    
    def eval(self, value, x):
        bc = self.bc(self.lambda_, self.mu, x)
        value[0] = bc[0] 
        value[1] = bc[1] 
    
    def value_shape(self):
        return (2,)
    
# Still need to obtain sigma from the displacement
# plot_function(u, "Displacement")
#checkF(f, f_expression, getF)
# plot_everything(mesh, u, image)

def solutionPlot(mesh, value):
    from matplotlib import pyplot as plt
    marksize=2
    alpha=0.8
    marker='o'
    savepath = './solutions'

    figU, axU = plt.subplots()
    axU.set_aspect('equal')
    cp = axU.contourf(mesh[0], mesh[1], value[:, 0].reshape(100, 100), alpha = alpha-0.1, edgecolors='None', cmap='rainbow', marker=marker, s=int(marksize), vmin=-0.8, vmax=0.8)
    axU.set_xticks([])
    axU.set_yticks([])
    axU.set_xlim([0, 1])
    axU.set_ylim([0, 1])
    axU.set_xlabel("x (m)")
    axU.set_ylabel("y (m)")
    plt.title('PINN U $(m)$')
    figU.colorbar(cp)
    plt.savefig(savepath + '/u_PINN.png', dpi=300)
    plt.show()


def save_sample(E, u, input_file='input_data.npy', output_file= 'output_data.npy'):
    N_x = 32
    N_y = 32
    x_coords = np.linspace(0, 1, N_x)
    y_coords = np.linspace(0, 1, N_y)
    X, Y = np.meshgrid(x_coords, y_coords)
    # X = X.flatten()
    # Y = Y.flatten()
    input = np.zeros((N_x, N_y))
    output = np.zeros((N_x, N_y, 2))
    for i in range(N_x):
        for j in range(N_y):
            input[i, j] = E(X[i, j], Y[i, j])
            output[i, j] = u(X[i, j], Y[i, j])

    input_data = input.reshape((1, N_x, N_y, 1))
    output_data = output.reshape((1, N_x, N_y, 2))
    
    
    try:
        # Load existing data from the file
        existing_input_data = np.load(input_file)
        existing_output_data = np.load(output_file)

        # Append the new data
        updated_input_data = np.concatenate((existing_input_data, input_data), axis=0)
        updated_output_data = np.concatenate((existing_output_data, output_data), axis=0)
    except FileNotFoundError:
        # If file does not exist, initialize with new data
        updated_input_data = input_data
        updated_output_data = output_data
    
    # # Save the input and output data to .npy files
    np.save(input_file, updated_input_data)
    np.save(output_file, updated_output_data)

def create_sample():
    # N - number of samples
    E_1 = 1
    nu = 0.3
    contrast_ratio = 0.5

    size_image = 5
    _, image = generate_image(D=size_image)
    # image = image.reshape(size_image, size_image)

    L = 1
    E1 = E_1  # E-Modul
    nu = nu  # Poisson's number
    frac = 1 / contrast_ratio
    E2 = frac * E_1

    # # Just for the initial case
    # lamb=1
    # mu=0.5s
    # nu = lamb/ (2*lamb + 2*mu)
    # E_example = mu * 2*(1+nu)

    # Create mesh and define function space
    mesh = UnitSquareMesh(size_image, size_image)
    V = VectorFunctionSpace(mesh, 'P', 1) 

    materials = E(image, size_image, E1, E2)  # correct is E1, E2
    Q = FunctionSpace(mesh, 'DG', 0)
    Y = interpolate(materials, V=Q) #Young's Module

    #plot_function(Y, "Young's Module")

    lbd = project(getLambda(Y, nu), Q)  # Lambda function
    mu = project(getMu(Y, nu), Q)   # Mu function

    f_expression = F(lbd, mu, degree=2)
    f = interpolate(f_expression, V=V)  # Source term
    
    # f_x, f_y = ufl.split(f)
    # plot_function(f_x, 'Source term')

    bcX0_expression = BoundaryCondition(lbd, mu, getBCX0, degree=2)
    bcX1_expression = BoundaryCondition(lbd, mu, getBCX1, degree=2)
    bcY0_expression = BoundaryCondition(lbd, mu, getBCY0, degree=2)
    bcY1_expression = BoundaryCondition(lbd, mu, getBCY1, degree=2)

    bX0 = BoundaryX0()
    bX1 = BoundaryX1()
    bY0 = BoundaryY0()
    bY1 = BoundaryY1()

    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    bX0.mark(boundary_markers, 0)
    bX1.mark(boundary_markers, 1)
    bY0.mark(boundary_markers, 2)
    bY1.mark(boundary_markers, 3)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    boundary_conditions = {
        0: bcX0_expression,
        1: bcX1_expression,
        2: bcY0_expression,
        3: bcY1_expression
    }

    bc = DirichletBC(V, Constant((0, 0)), bY0)
    # Make all DirichletBC

    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    a = inner(sigma(lbd, mu, d, u), epsilon(v))*dx
    L = dot(f, v)*dx
    for i, g_i in boundary_conditions.items():
        if i!=2:
            L += dot(g_i, v) * ds(i)

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    u_x, u_y = ufl.split(u)
    # plot_function(u_x, "Solution")
    
    W = TensorFunctionSpace(mesh, 'P', 1)  # Tensor function space for the stress field
    stress = project(sigma(lbd, mu, d, u), W)
    stress_xx, stress_xy, stress_yx, stress_yy = stress.split()
    # plot_function(stress_xx, "xx component of the stress")
    # plot_function(stress_xy, "xy component of the stress")
    # plot_function(stress_yx, "yx component of the stress")
    # plot_function(stress_yy, "yy component of the stress")
    save_sample(materials, u, 'input_data.npy', 'output_data.npy') 

import os

# List of files to delete
files_to_delete = ['input_data.npy', 'output_data.npy']

# Loop through the list and delete each file
for file_path in files_to_delete:
    # Check if the file exists before attempting to delete it
    if os.path.exists(file_path):
        os.remove(file_path)  # Delete the file
        print(f"{file_path} has been deleted successfully.")
    else:
        print(f"The file {file_path} does not exist.")

nr_samples = 1000
for i in range(nr_samples):
    create_sample()





