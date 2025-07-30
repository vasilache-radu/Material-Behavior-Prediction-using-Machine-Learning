from timeit import default_timer
from FNOhelper.helper import *
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.datasets.data_transforms import DefaultDataProcessor

# Define a function to train a model with given hyperparameters
def train_config(save_path, n_modes, hidden_channels, in_channels, out_channels, n_layers):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Step 1: Generate Synthetic Input-Output Data
    inputs = np.load(f'{save_path}/input_data.npy') # (num_samples, N_x, N_y, C_in)
    outputs = np.load(f'{save_path}/output_data.npy') # (num_samples, N_x, N_y, C_out) 
    
    dataset = FNODataset(inputs, outputs, device=device)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # 20% for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loaders = {"val": val_loader}

    model = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=in_channels, out_channels=out_channels, n_layers=n_layers).to(device)
    # model.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    data_processor = DefaultDataProcessor()
    data_processor = data_processor.to(device)

    # Creating the losses
    l2loss = LpLoss(d=3, p=2, reductions='sum', reduce_dims=[0])
    h1loss = H1Loss(d=3, reductions='sum', reduce_dims=[0])
    train_loss = h1loss
    eval_losses={'l2': l2loss}
    
    behaviour_callback = BehaviourCallback(
        model=model,
        val_loader=val_loader,
        data_processor=data_processor,
        save_path = save_path,
        device = device
    )
    
    trainer = Trainer(
        model=model,
        n_epochs=200,
        wandb_log=False,
        device=device,
        log_test_interval=1, 
        verbose=True,
        callbacks=[behaviour_callback]
    )
    
    try:
        start_time = default_timer()
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=None,
            training_loss=train_loss,
            eval_losses=eval_losses
        )
        
    except Exception as e:
        print(f"Exception: {e}")

    end_time = default_timer()
    print(f"Training time: {end_time - start_time}")

    plot(save_path, 'lr=0.001,bs=16,h1', behaviour_callback)

