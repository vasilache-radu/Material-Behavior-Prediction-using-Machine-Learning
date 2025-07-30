from fenics import *
import fenics
from ufl import nabla_div
from ufl import nabla_grad
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy
import time

class E(UserExpression):
    def __init__(self, material, e_1, e_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.material = material
        self.e_1 = e_1
        self.e_2 = e_2

    def eval_cell(self, values, x, cell):
        if self.material[cell.index] == 0:
            values[0] = self.e_1
        else:
            values[0] = self.e_2

    def value_shape(self):
        # return (1,)
        return ()

def lambda_(E, nu):
    return (E * nu) / ((1 + nu) * (1 - 2 * nu))  # Lame's 1st parameter

def mu(E, nu):
    return E / (2 * (1 + nu))  # Lame's 2nd parameter

def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def lambda_prime(lambd, mue):
    return 2 * lambd * mue / (lambd + 2 * mue)


def getIndexForNodes(dof_cords, tol):
    """
    From array dof_cords construct the corresponding index vectors for 
    boundary and inner nodes of a quadratic sample in [0,1] x [0,1]. 
    Bound nodes are at the edges of the sample.
    """
    boundary_left = dof_cords[:, 0] < tol
    boundary_right = dof_cords[:, 0] > 1 - tol
    boundary_top = dof_cords[:, 1] > 1 - tol
    boundary_bottom = dof_cords[:, 1] < tol
    bound = boundary_left + boundary_right + boundary_top + boundary_bottom
    inner = ~bound
    return inner, bound

def getK(V, sigma, volume_force=(0,0)):
      u = TrialFunction(V)
      d = u.geometric_dimension()  # space dimension
      v = TestFunction(V)
      f = Constant(volume_force)
      a = fenics.inner(sigma(u), epsilon(v)) * dx
      L = dot(f, v) * dx

    #   u = Function(V)
    #   solve(a==L, u, bc)
      # Assemble System Matrix
      A = assemble(a)
      K = A.array()

      return K


# calculating the material matrix

def calculate_material_matrix(rand_vec, n=10, E_1=1., contrast_ratio=0.5, nu=0.3):
    # Variables
    frac = 1 / contrast_ratio
    E_2 = frac * E_1
    eps_bar = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])  # according to Miehe, 2002: eps_bar[:,x] = [ \eps_11, \eps_22, 2*\eps_12]

    # Create mesh and define function space
    mesh = UnitSquareMesh(n, n)
    # mesh = RectangleMesh(Point(0., 0.), Point(1., 100.), 3, 20)
    V = VectorFunctionSpace(mesh, 'P', 1)

    # Define boundary condition

    def boundary(x, on_boundary):
        return on_boundary

    # Define mesh function for material boundaries
    tol = 1E-12
    materials = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    domain = rand_vec.reshape(n, n)
    materials_domain = np.zeros((n, 2 * n)).astype(int)  # since mesh is triangular one need double the entries
    for i in range(n):  # due to fenics mesh function,
        materials_domain[n - i - 1, ::2] = domain[i, :]
        materials_domain[n - i - 1, 1::2] = domain[i, :]
    materials.array()[:] = materials_domain.flatten()
    File('materials2D.xml') << materials



    # Define strain and stress
    e_mod = E(materials, E_1, E_2)
    print(type(e_mod))
    # according to https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html

    def sigma(u):
        return lambda_prime(lambda_(e_mod, nu), mu(e_mod, nu)) * nabla_div(u) * Identity(2) + 2 * mu(e_mod, nu) * epsilon(u)
    
    """
    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0))
    a = fenics.inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx

    # Assemble System Matrix
    A = assemble(a)
    K = A.array()
    """

    K = getK(V, sigma, volume_force=(0,0))


    # dof's:
    geometrical_dim = mesh.geometry().dim()
    dof_cords = V.tabulate_dof_coordinates()  # Coordinates of all dofs in system
    vertex_cords = dof_cords[1::2, :]

    """
    # construct boolean index vectors for  boundary and inner nodes
    boundary_left = dof_cords[:, 0] < tol
    boundary_right = dof_cords[:, 0] > 1 - tol
    boundary_top = dof_cords[:, 1] > 1 - tol
    boundary_bottom = dof_cords[:, 1] < tol
    bound = boundary_left + boundary_right + boundary_top + boundary_bottom
    inner = ~bound
    """

    inner, bound = getIndexForNodes(dof_cords, tol)
    
    

    # del boundary_left, boundary_right, boundary_bottom, boundary_top
    # del A, u, d, v, f, a, L

    # print("here 2")
    # Construct submatrices: a = inner nodes, b = boundary nodes
    #  ___________   ___     ___
    # | K_aa K_ab | |u_a| _ |f_a|
    # | K_ba K_bb | |u_b| ‾ |f_b|
    #  ‾‾‾‾‾‾‾‾‾‾‾   ‾‾‾     ‾‾‾

    K_aa = K[inner, :][:, inner]
    # print("K_aa done")
    K_ab = K[inner, :][:, bound]
    # print("K_ab done")
    K_ba = K[bound, :][:, inner]
    # print("K_ba done")
    K_bb = K[bound, :][:, bound]
    # print("K_bb done")

    
    
    # del K, vertex_cords

    # print("here 3")
    # Construct D matrix according to Miehe, 2002
    #  __________________
    # |   x_1       0    |
    # |    0       x_2   | = D_q --> D = [ D_1 D_2 ... D_M ]
    # | 0.5*x_2  0.5*x_1 |
    #  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    start = time.process_time()

    bound_vertex_cords = dof_cords[bound, :][0::2, :]  # Coordinates of all vertices in system, gdim dofs per vertex
    D = np.zeros((3, bound_vertex_cords.size))
    D[0, 0::2] = bound_vertex_cords[:, 0]
    D[1, 1::2] = bound_vertex_cords[:, 1]
    D[2, 0::2] = 0.5 * bound_vertex_cords[:, 1]
    D[2, 1::2] = 0.5 * bound_vertex_cords[:, 0]

    # del bound_vertex_cords

    

    # print("Set up: ", time.process_time() - start)
    start = time.process_time()
    # Calculate Displacement on boundary according to Miehe , 2002
    u_bound = D.T @ eps_bar


    # Solve for inner node displacement
    # K_aa is sparse so scipy.sparse is used
    u_a = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(K_aa), -K_ab @ u_bound)
    

    # Construct dof displacement vector
    u_vec = np.zeros((dof_cords.shape[0], eps_bar.shape[1]))
    u_vec[inner, :] = u_a
    u_vec[bound, :] = u_bound

    # del u_vec


    # print("First matrix mults: ", time.process_time() - start)
    start = time.process_time()

    # Calculate Nodal Forces
    f_b = K_ba @ u_a + K_bb @ u_bound
    # f_b_vec = f_b.reshape(-1, 2 * eps_bar.shape[1])
    # del u_bound
    # Calculate macroscopic stress vector according to Miehe, 2002
    sigma_bar = D @ f_b

    # Calculate tangent moduli according to Miehe, 2002

    # Print infos for validation
    # print('Nodal Coordinates:\n', vertex_cords)
    # print('Nodal Force Vector:\n', f_b)
    # print('Sigma bar:\n', sigma_bar)
    # print('Tangent moduli:\n', C)
    # print("Second matmuls: ", time.process_time() - start)

    return sigma_bar


if __name__ == '__main__':
    vec = np.zeros(100)
    vec[np.random.randint(0, 100, 5)] = 1
    cMatrix = calculate_material_matrix(vec, n=10, E_1=1., contrast_ratio=0.1)
    print('C matrix = ')
    print(cMatrix)