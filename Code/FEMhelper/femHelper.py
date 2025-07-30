from fenics import *
from FEMhelper.plot import plot_all
from ImageGenerator import generate_image
from PINN.pointsGeneration import locationImage
from ufl import nabla_div, nabla_grad
from FEMhelper.bc import bcX0, bcX1, bcY0, bcY1, f

# Define the material properties
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
def getLambda(E, nu):
    return (E * nu) / ((1 + nu) * (1 - 2 * nu))  # Lame's 1st parameter

def getMu(E, nu):
    return E / (2 * (1 + nu))  # Lame's 2nd parameter

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(lbd, mu, d, u):
    return lbd*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define the source term
class F(UserExpression):
    def __init__(self, lambda_, mu, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.f = f
    
    def eval(self, value, x):
        f_val = self.f(self.lambda_, self.mu, x)
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

class Problem():
    def __init__(self, getF, getBCX0, getBCX1, getBCY0, getBCY1):
        self.getF = getF
        self.getBCX0 = getBCX0
        self.getBCX1 = getBCX1
        self.getBCY0 = getBCY0
        self.getBCY1 = getBCY1
    
    def get_setting(self):
        return self.getF, self.getBCX0, self.getBCX1, self.getBCY0, self.getBCY1

# Solve the problem and return the materials, displacement, stress, and strain
def solve_problem(problem=None, E_1=1, nu=0.3, contrast_ratio=2, size_image=32):

    # Define the problem
    if problem is None:
        problem = Problem(f, bcX0, bcX1, bcY0, bcY1)
    getF, getBCX0, getBCX1, getBCY0, getBCY1 = problem.get_setting()
    
    # Generate the image (material structure)
    _, image = generate_image(D=size_image)

    # Define the material properties
    E1 = E_1  # E-Modul
    nu = nu  # Poisson's number
    E2 = contrast_ratio * E_1

    # Create mesh and define function space
    # Resolution must be larger than the image size
    # Currently equal to the image size; Should it have at least a minimum resolution?
    mesh = UnitSquareMesh(size_image, size_image)
    V = VectorFunctionSpace(mesh, 'P', 1) 

    materials = E(image, size_image, E1, E2)
    Q = FunctionSpace(mesh, 'DG', 0)
    Y = interpolate(materials, V=Q) #Young's Module

    lbd = project(getLambda(Y, nu), Q)  # Lambda function
    mu = project(getMu(Y, nu), Q)   # Mu function

    f_expression = F(lbd, mu, getF, degree=2)
    f = interpolate(f_expression, V=V)  # Source term

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
    
    W = TensorFunctionSpace(mesh, 'P', 1)  # Tensor function space for the stress field
    stress = project(sigma(lbd, mu, d, u), W)
    strain = project(epsilon(u), W)
    
    # plot_all(image.reshape(size_image, size_image), mesh, u, Y, stress, strain)  
    return materials, u, stress, strain