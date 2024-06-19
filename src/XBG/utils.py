import os
import torch

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        print('=== Fit data Scaler ===')
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        print('=== Scaling the data ===')
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
    def inverse_transform(self, values):
        return values * (self.std + self.epsilon) + self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.epsilon = self.epsilon.to(device)

def evaluate(loader, model, loss_fcn, device=0):
    print('=== Evaluating ===\n')
    val_loss = torch.tensor(0).to(device).float()
    num_batches = torch.tensor(0).to(device).float()
    model.eval()
    with torch.no_grad():
        for (rgb_imgs, depth_imgs, sensors, targets) in loader:
            rgb_imgs = rgb_imgs.to(device=device)
            depth_imgs = depth_imgs.to(device=device)
            targets = targets.to(device=device).float()
            sensors = sensors.to(device=device).float()

            scores = model(sensors, rgb_imgs, depth_imgs)
            batch_loss = loss_fcn(scores, targets)
            val_loss += batch_loss
            num_batches += 1
            # print(num_batches,targets.shape[0], batch_loss, val_loss)
    model.train()
    return val_loss/num_batches

def save_checkpoint(model, optimizer, loss, epoch, run, scaler, config):
    print('=== Creating Checkpoint ===\n')
    dir = f'../../../assets/trained/{run.project}/{run.id}/'
    print(run.config)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if epoch % 1 == 0:
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.module.state_dict(),
            'config': dict(run.config),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_mean': scaler.mean,
            'scaler_std': scaler.std
            }, f'{dir}checkpoint.pt')
    if epoch % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'config': dict(run.config),
            'scaler_mean': scaler.mean,
            'scaler_std': scaler.std
            }, f'{dir}epoch{epoch}.pt')
    # print(config)
    return True

def load_checkpoint(model, checkpoint_path, checkpoint_epoch):
    if checkpoint_path:
        print('\n=== Loading Checkpoint ===')
        checkpoint = torch.load(f'../../../assets/trained/{checkpoint_path}/epoch{checkpoint_epoch}.pt')
        scaler = StandardScaler(checkpoint['scaler_mean'] , checkpoint['scaler_std'])
        weights = checkpoint['model_state_dict']
        model.load_state_dict(weights)
        print(f'\n=== Resuming training from epoch {checkpoint_epoch+1} ===')
        return model, checkpoint_epoch+1, scaler, checkpoint['config']
    else:
        return model, 0, None, None
    

  