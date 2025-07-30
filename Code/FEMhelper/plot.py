import os
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def plot_function(f, title):
    # Plot displacement
    import matplotlib.pyplot as plt
    plt.figure()
    p=plot(f, title = title)
    plt.colorbar(p)
    plt.show()

def plot_everything(mesh, f, image):
    import matplotlib.pyplot as plt
    plt.figure()
    plot(mesh)
    p=plot(f, title="Young's module on the mesh")
    plt.colorbar(p)
    if isinstance(image, np.ndarray) and image.ndim == 2:
        plt.figure()
        plt.imshow(image, cmap='viridis')  # Display the first matrix
        plt.colorbar()  # Add a colorbar
        plt.title("Image")
        plt.show()  # Display the first plot
    else:
        raise ValueError("The 'image' variable must be a 2D numpy array.")

def checkF(f, f_expression, getF, lambda_, mu):
    # Checking f
    x = np.random.rand(10)
    y = np.random.rand(10)
    xy = np.meshgrid(x, y)
    x_mesh = xy[0].reshape(-1, 1)
    y_mesh = xy[1].reshape(-1, 1)
    xy_mesh = np.concatenate((x_mesh, y_mesh), axis=1)

    x_point = np.array([0.252,	0.401])
    f_value = np.zeros(2)
    f_expression.eval(f_value, x_point)

    print(f_value)
    print(f(x_point))
    print(getF(
    lambda_,
    mu,
    x_point
    ))

    #print(xy_mesh)
    length = xy_mesh.shape[0]
    error=0
    high_errors_counter=0
    huge_errors_counter=0
    for i in range(length):
        f_fem = f(xy_mesh[i, :])
        f_value = np.zeros(2)
        f_expression.eval(f_value, xy_mesh[i, :])
        f_anal = getF(lambda_, mu, xy_mesh[i, :])
        delta = f_fem - f_value
        delta_L2 = sqrt(delta[0]**2 + delta[1]**2)
        if(delta_L2>=1.5):
            if delta_L2 >=10:
                huge_errors_counter = huge_errors_counter+1
            else:
                high_errors_counter = high_errors_counter+1
            print(delta)
        else:
            error = error + delta_L2

    print(error)
    print(high_errors_counter)
    print(huge_errors_counter)

def inquire(save_path):
    input_data = np.load(f'{save_path}/input_data.npy')
    output_data = np.load(f'{save_path}/output_data.npy')

    assert not np.isnan(input_data).any(), "NaNs found in inputs"
    assert not np.isnan(output_data).any(), "NaNs found in outputs" # They are found here
    assert not np.isinf(input_data).any(), "Infinities found in inputs"
    assert not np.isinf(output_data).any(), "Infinities found in outputs" # They are found here
    print(input_data.shape)
    print(output_data.shape)

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

def plot_all(image, mesh, u, Y, stress, strain):
    u_x, u_y = u.split()
    stress_xx, stress_xy, stress_yx, stress_yy = stress.split()

    plot_everything(mesh, Y, image)
    plot_function(u_x, title="DisplacementX")
    plot_function(u_y, title="DisplacementY")
    plot_function(stress_xx, title="StressXX")
    plot_function(stress_xy, title="StressXY")
    plot_function(stress_yx, title="StressYX")
    plot_function(stress_yy, title="StressYY")