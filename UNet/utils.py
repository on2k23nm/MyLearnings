import torch
import torchvision

import CaravanaDataset

from torch.utils.data import DataLoader


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
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
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum())/(preds + y ).sum() + 1e-8
             
    print(f'Got {num_correct}/{num_pixels} with accuracy of {(num_correct/num_pixels)*100:.3f}, Dice score : {dice_score}')
    
    model.train()


def save_preds_as_imgs(loader, model, save_dir='saved_images', device='cuda'):
    
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{save_dir}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{save_dir}/{idx}.png')

    model.train()

            
    