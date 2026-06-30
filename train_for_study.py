import sys
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import os
import math
from models.dcdicl_net import DCDicL_1D
from utils.metrics import DCDicLLoss


def set_deterministic_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OfflineGWDataset(Dataset):
    def __init__(self, noisy_npz_path, clean_npz_path, is_train=False):
        if not os.path.exists(noisy_npz_path) or not os.path.exists(clean_npz_path):
            raise FileNotFoundError(f"Missing dataset files! Please run the offline generator first.\nExpected:\n- {noisy_npz_path}\n- {clean_npz_path}")
            
        print(f"Loading offline data:\n -> Noisy: {noisy_npz_path}\n -> Clean: {clean_npz_path}\n -> Augmentation (Train Mode): {is_train}")
        
        self.noisy_data = np.load(noisy_npz_path)['strains']
        self.clean_data = np.load(clean_npz_path)['strains']
        self.is_train = is_train
        
        assert self.noisy_data.shape == self.clean_data.shape, "Shape mismatch between noisy and clean datasets!"

    def __len__(self):
        return self.noisy_data.shape[0]

    def __getitem__(self, idx):
        noisy_np = self.noisy_data[idx]
        clean_np = self.clean_data[idx]

        # DYNAMIC DATA AUGMENTATION
        if self.is_train:
            shift = np.random.randint(-500, 500)
            noisy_np = np.roll(noisy_np, shift)
            clean_np = np.roll(clean_np, shift)

        noisy_t = torch.from_numpy(noisy_np.copy()).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean_np.copy()).unsqueeze(0).float()
        
        return noisy_t, clean_t


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    set_deterministic_seeds(42)

    config = load_config('config.yaml')
    cfg_model = config['model']
    cfg_train = config['training']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")
    
    data_dir = "data_dcdicl_ready"
    
    train_noisy_path = os.path.join(data_dir, "train_noisy_whitened.npz")
    train_clean_path = os.path.join(data_dir, "train_clean_whitened.npz")
    
    val_noisy_path = os.path.join(data_dir, "val_noisy_whitened.npz")
    val_clean_path = os.path.join(data_dir, "val_clean_whitened.npz")

    try:
        train_dataset = OfflineGWDataset(train_noisy_path, train_clean_path, is_train=True)
        val_dataset = OfflineGWDataset(val_noisy_path, val_clean_path, is_train=False)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1) 

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg_train.get('batch_size', 16), 
        shuffle=True, 
        pin_memory=False, 
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg_train.get('batch_size', 16), 
        shuffle=False, 
        pin_memory=False, 
        num_workers=0
    )

    model = DCDicL_1D(
        iterations=cfg_model['iterations'],   
        num_filters=cfg_model['num_filters'], 
        kernel_size=cfg_model['kernel_size'],
        max_alpha=cfg_model.get('max_alpha', 1.0),
        max_beta=cfg_model.get('max_beta', 0.5)
    ).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=cfg_train.get('learning_rate', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = DCDicLLoss(peak_weight=0.5)

    best_val_loss = float('inf') 
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True) 


    print(f"\n--- Starting Training for {cfg_train.get('epochs', 20)} Epochs ---")
    
    for epoch in range(cfg_train.get('epochs', 20)):
        
        # TRAINING
        model.train()
        train_loss = 0.0
        
        for noisy_signals, clean_signals in train_dataloader:
            noisy_signals = noisy_signals.to(device, non_blocking=True)
            target_signals = clean_signals.to(device, non_blocking=True) 
            
            optimizer.zero_grad()
            
            reconstructed_output, z_maps = model(noisy_signals)
            loss, _, _ = criterion(target_signals, reconstructed_output, z_maps)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg_train.get('clip_max_norm', 1.0))
            
            optimizer.step()

            with torch.no_grad():
                model.dictionary.D.weight.clamp_(-1.0, 1.0)
                model.dictionary_T.weight.clamp_(-1.0, 1.0)
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_dataloader)


        if math.isnan(avg_train_loss):
            print(f"\n[!] WARNING: Exploding gradients detected (NaN loss) at epoch {epoch+1}.")
            print("[!] Halting this training run early and signaling failure to study.py.")
            sys.exit(1)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy_signals, clean_signals in val_dataloader:
                noisy_signals = noisy_signals.to(device, non_blocking=True)
                target_signals = clean_signals.to(device, non_blocking=True)
                
                reconstructed_output, z_maps = model(noisy_signals)
                loss, _, _ = criterion(target_signals, reconstructed_output, z_maps)
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{cfg_train.get('epochs', 20)}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving checkpoint...")
            best_val_loss = avg_val_loss
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_dcdicl_gw_model.pth'))

    print("\nTraining run finished!")

if __name__ == "__main__":
    train()