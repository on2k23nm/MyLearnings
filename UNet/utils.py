import torch
import torchvision

from DataSet import CaravanaDataset
from tqdm.notebook import tqdm

from torch.utils.data import DataLoader


def save_checkpoint(state, filename='checkpoints/my_checkpoint.pth'):
    print('=> Saving Checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    
    model.eval()
    
    with torch.no_grad():
        loop = tqdm(loader)
        for batch_idx, (x, y) in enumerate(loop):
            x = x.permute([0, 3, 1, 2]).to(device=device)
            y = y.to(device).unsqueeze(1).to(device=device)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += preds.numel()
            dice_score += (2*(preds * y).sum().item())/(preds + y).sum().item() + 1e-8
             
    print(f'Got {num_correct}/{num_pixels} with accuracy of {(num_correct/num_pixels)*100:.3f}%, Dice score : {dice_score:.3f}')
    
    model.train()


def save_preds_as_imgs(loader, model, save_dir='saved_images', device='cuda'):
    
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.permute([0, 3, 1, 2]).to(device=device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # print(f'x.size : {x.size()}, pred.size : {preds.size()}')
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{save_dir}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{save_dir}/{idx}.png')

    model.train()

def create_dataset(path):
    return CaravanaDataset(path)

def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle)

    


            
    