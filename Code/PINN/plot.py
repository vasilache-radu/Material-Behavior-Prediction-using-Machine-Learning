import numpy as np
import matplotlib.pyplot as plt 
import math

import torch

def printDataSet(dataSet, image):
    size_image = int(math.sqrt(len(image)))
    innerPoints, boundaryPoints = dataSet

    x_vector_inner =np.array([point.x for point in innerPoints])
    y_vector_inner =np.array([point.y for point in innerPoints])
    E_vector_inner = [point.E for point in innerPoints]

    print(x_vector_inner)
    print(y_vector_inner)
    print(E_vector_inner)

    x_vector_boundary = np.array([point.x for point in boundaryPoints])
    y_vector_boundary = np.array([point.y for point in boundaryPoints])
    Normal_vector_boundary= [point.normalVector for point in boundaryPoints]

    print(x_vector_boundary)
    print(y_vector_boundary)
    print(Normal_vector_boundary)


    image = image.reshape(size_image, size_image)
    plt.imshow(image)
    plt.xticks(np.arange(size_image) - 0.5, np.arange(size_image))
    plt.yticks(np.arange(size_image) - 0.5, np.arange(size_image))
    plt.gca().set_xticks(np.arange(-0.5, size_image, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, size_image, 1), minor=True)

    x_plot_boundary = x_vector_boundary * size_image - 0.5
    y_plot_boundary = y_vector_boundary * size_image - 0.5

    plt.scatter(x_plot_boundary, y_plot_boundary)

    x_plot_inner = x_vector_inner * size_image - 0.5
    y_plot_inner = y_vector_inner * size_image - 0.5

    plt.scatter(x_plot_inner, y_plot_inner)

def printDataSet2(dataSet, image):
    size_image = int(math.sqrt(len(image)))
    innerPoints, boundaryPoints = dataSet

    # Extract coordinates and properties from innerPoints
    x_vector_inner = np.array([point.x for point in innerPoints])
    y_vector_inner = np.array([point.y for point in innerPoints])
    E_vector_inner = [point.E for point in innerPoints]

    # Extract coordinates from boundaryPoints
    x_vector_boundary = np.array([point.x for point in boundaryPoints])
    y_vector_boundary = np.array([point.y for point in boundaryPoints])
    Normal_vector_boundary = [point.normalVector for point in boundaryPoints]

    # Reshape the image into a square matrix
    image = image.reshape(size_image, size_image)

    # Create a new figure for the plot
    plt.figure()

    # Display the image
    plt.imshow(image, cmap='gray', extent=[0, size_image, 0, size_image])
    plt.colorbar(label='Intensity')
    
    # Adjust the boundary points to fit the image size
    plt.scatter(x_vector_boundary * size_image, y_vector_boundary * size_image, 
                c='red', label='Boundary Points', edgecolor='k')

    # Adjust the inner points to fit the image size
    plt.scatter(x_vector_inner * size_image, y_vector_inner * size_image, 
                c='blue', label='Inner Points', edgecolor='k')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Set Visualization')
    plt.legend()
    plt.axis('equal')  # Keep aspect ratio equal

    # Show the plot and keep it open
    plt.show()  # This will block further code execution until the window is closed

def checkError(modelSigma, modelDisplacement, getSigmaAnal, getDisplacementAnal, device):
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    xy = np.meshgrid(x, y)

    x_mesh = xy[0].reshape(-1, 1)
    y_mesh = xy[1].reshape(-1, 1)
    xy_mesh = np.concatenate((x_mesh, y_mesh), axis=1)

    xy_mesh = torch.tensor(xy_mesh, dtype=torch.float32).to(device=device)

    predicted_mesh = modelSigma(xy_mesh).to(device='cpu')
    predicted_mesh_U = modelDisplacement(xy_mesh).to(device='cpu')

    xy_mesh = xy_mesh.to('cpu')

    anal_mesh = getSigmaAnal(xy_mesh).to(device='cpu')
    anal_mesh_U = getDisplacementAnal(xy_mesh).to(device='cpu')

    epsilon = 1e-8

    with torch.no_grad():
        diff = torch.norm(predicted_mesh-anal_mesh, p=2, dim=-2, keepdim=False)
        ynorm = torch.norm(anal_mesh, p=2, dim=-2, keepdim=False)
        #rel_error_sigma = torch.sum(diff/(ynorm + epsilon), dim=0) / xy_mesh.shape[0]
        rel_error_sigma = diff/ynorm

        diff = torch.norm(predicted_mesh_U-anal_mesh_U, p=2, dim=-2, keepdim=False)
        ynorm = torch.norm(anal_mesh_U, p=2, dim=-2, keepdim=False)
        # rel_error_displacement = torch.sum(diff/(ynorm + epsilon), dim=0) / xy_mesh.shape[0]
        rel_error_displacement = diff/ynorm
    
    predicted_mesh = predicted_mesh.detach().numpy()
    predicted_mesh_U = predicted_mesh_U.detach().numpy()
    anal_mesh = anal_mesh.detach().numpy()
    anal_mesh_U = anal_mesh_U.detach().numpy()

    error_sigma_array = np.sum((predicted_mesh-anal_mesh)**2, axis=1)
    error_sigma = np.average(error_sigma_array)

    relative_error_sigma = error_sigma / np.average(np.sum(anal_mesh**2, axis=1))
    
    error_displacement_array = np.sum((predicted_mesh_U-anal_mesh_U)**2, axis=1)
    error_displacement = np.average(error_displacement_array)

    relative_error_displacement = error_displacement / np.average(np.sum(anal_mesh_U**2, axis=1))

    # Need to change here
    return relative_error_sigma, relative_error_displacement

def plot_loss(epochMeanLosses, epochMeanInnerLosses, epochMeanBoundaryLosses, finished_epochs, savepath):
    import matplotlib.pyplot as plt
    
    plt.semilogy(range(finished_epochs), epochMeanLosses, label='TotalLoss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.semilogy(range(finished_epochs), epochMeanInnerLosses, label='InnerLoss')

    plt.semilogy(range(finished_epochs), epochMeanBoundaryLosses, label='BoundaryLoss')

    plt.legend()
    plt.savefig(savepath + '/Loss.png', dpi=300)
    plt.show()

def plot_rel_error(epoch_accuracy, accuracy_sigma_list, accuracy_displacement_list, savepath):

    print('Relative Error Sigma: ', accuracy_sigma_list[-1]*100, '%')
    print('Relative Error Displacement: ', accuracy_displacement_list[-1]*100, '%')
    print('Best relative error sigma: ', min(accuracy_sigma_list)*100, '%')
    print('Best relative error displacement: ', min(accuracy_displacement_list)*100, '%')

    plt.semilogy(epoch_accuracy, accuracy_sigma_list, label='rel_error_sigma')
    plt.semilogy(epoch_accuracy, accuracy_displacement_list, label='rel_error_displacement')
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.title('Model Relative Error over Epochs')
    plt.grid(True)
    plt.savefig(savepath + '/rel_errors.png', dpi=300)
    plt.show()

def plot_rel_error_runs(rel_errors_sigma_runs, rel_errors_displacement_runs, name_runs, epochs, savepath):
    fig1 = plt.figure('Stress Model Relative Error')
    fig2 = plt.figure('Displacement Model Relative Error')

    # Plot relative errors for the stress model
    for i in range(len(rel_errors_sigma_runs)):
        plt.figure(fig1.number)  # Switch to the stress model figure
        plt.semilogy(range(epochs), rel_errors_sigma_runs[i], label=name_runs[i])

    # Finalize the stress model figure
    plt.figure(fig1.number)
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error Stress Model')
    plt.legend()
    plt.title('Model Relative Error Stress over Epochs')
    plt.grid(True)
    plt.savefig(f"{savepath}/rel_errors_sigma.png", dpi=300)

    # Plot relative errors for the displacement model
    for i in range(len(rel_errors_displacement_runs)):
        plt.figure(fig2.number)  # Switch to the displacement model figure
        plt.semilogy(range(epochs), rel_errors_displacement_runs[i], label=name_runs[i])

    # Finalize the displacement model figure
    plt.figure(fig2.number)
    plt.xlabel('Epochs')
    plt.ylabel('Relative Error Displacement Model')
    plt.legend()
    plt.title('Model Relative Error Displacement over Epochs')
    plt.grid(True)
    plt.savefig(f"{savepath}/rel_errors_displacement.png", dpi=300)

    # Show both figures at the end
    plt.show()