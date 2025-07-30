import torch


def checkLoss(estimation, condition, mask = None):
    
    loss = torch.sum((estimation - condition)**2, dim=1, keepdim=True)
    print(loss)
    if mask==None:
        nr_samples = len(loss)
    else:
        nr_samples = torch.count_nonzero(mask)
        print(mask)
        loss = loss * mask.reshape(mask.shape[0], 1)
        print(loss)
    if nr_samples==0:
        return 0
    
    loss = torch.sum(loss) / nr_samples
    return loss

# Example usage and test for checkLoss function
estimation = torch.tensor([[1.0, 3.0, 2.0], [3.0, 4.0, 1.0], [2.0, 1.0, 3.0], [4.0, 2.0, 1.0]])
condition = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 1.0], [4.0, 1.0, 2.0]])
mask = torch.tensor([1, 0, 1, 0])

loss_without_mask = checkLoss(estimation, condition)
print(f"Loss without mask: {loss_without_mask}")
assert loss_without_mask == torch.tensor(7.25), f"Expected loss without mask to be 1.5 but got {loss_without_mask}"

loss_with_mask = checkLoss(estimation, condition, mask)
print(f"Loss with mask: {loss_with_mask}")
assert loss_with_mask == torch.tensor(8), f"Expected loss with mask to be 1 but got {loss_with_mask}"

mse_loss = torch.nn.functional.mse_loss(estimation, condition, reduction='mean')
print(f"MSE Loss: {mse_loss}")
assert mse_loss == torch.tensor(29/12), f"Expected MSE loss to be 0.75 but got {mse_loss}"