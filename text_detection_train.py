import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.detection_dataloader import *
from models.rsmtd import *

config = {
    'data_path': r'F:\SynthText\SynthText',
    'batch_size': 8,
    'num_workers': 1,
    'lr': 1e-3,
    'epochs': 15,
    'img_size': (640, 640),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_samples': 1500
}

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f"Device = {config['device']}")

    # dataset = SynthTextDataset(config['data_path'], transform=transform, augment=True, sample_size=config['n_samples'])
    dataset = ICDAR15DetectionDataset(
        img_dir=r'D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar_og\ch4_training_images',
        gt_dir=r'D:\Projects\RPP\Sem 7\RISTE\data\raw\icdar_og\ch4_training_localization_transcription_gt',
        transform=transform,  # Add your PyTorch transforms here
        augment=True,
        sample_size=1000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model = RSMTD().to(config['device'])

    if 'checkpoint_path' in config and config['checkpoint_path']:
        checkpoint_path = config['checkpoint_path']
        model.load_state_dict(torch.load(checkpoint_path, map_location=config['device']))
        print(f"Loaded model from {checkpoint_path} for fine-tuning")

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=config['epochs'] * len(dataloader),
        pct_start=0.3
    )
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # print(f"Step: [{batch_idx}/{len(dataloader)}]")
            torch.cuda.empty_cache()
            images = batch['image'].to(config['device'])
            masks = batch['shrink_mask'].to(config['device'])
            offsets = batch['offsets'].to(config['device'])
            spw = batch['spw'].to(config['device'])

            optimizer.zero_grad()
            pred_masks, pred_offsets, pred_spw = model(images)
            loss = total_loss(pred_masks, pred_offsets, pred_spw, masks, offsets, spw)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{config["epochs"]} Loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), f'./model/text_detector/rsmtd_{epoch+1}.pth')

    torch.save(model.state_dict(), './model/text_detector/rsmtd_final.pth')

if __name__ == '__main__':
    train()