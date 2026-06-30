import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import shutil
from datetime import datetime
from models.dcdicl_net import DCDicL_1D


class OfflineGWDataset(Dataset):
    def __init__(self, noisy_npz_path, clean_npz_path):
        if not os.path.exists(noisy_npz_path) or not os.path.exists(clean_npz_path):
            raise FileNotFoundError(f"Missing dataset files! Please run the offline generator first.\nExpected:\n- {noisy_npz_path}\n- {clean_npz_path}")
            
        print(f"Loading offline test data:\n -> Noisy: {noisy_npz_path}\n -> Clean: {clean_npz_path}")
        
        self.noisy_data = np.load(noisy_npz_path)['strains']
        self.clean_data = np.load(clean_npz_path)['strains']
        
        assert self.noisy_data.shape == self.clean_data.shape, "Shape mismatch between noisy and clean datasets!"

    def __len__(self):
        return self.noisy_data.shape[0]

    def __getitem__(self, idx):
        noisy_t = torch.from_numpy(self.noisy_data[idx]).unsqueeze(0).float()
        clean_t = torch.from_numpy(self.clean_data[idx]).unsqueeze(0).float()
        return noisy_t, clean_t


def load_config(config_path='config.yaml'):
    """Loads the model and dataset configuration from the YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_batch_inference():
    config = load_config('config.yaml')
    cfg_model = config['model']
    batch_size = config.get('training', {}).get('batch_size', 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting batch inference on: {device} with batch size: {batch_size}")
    
    
    model_path = os.path.join('checkpoints', 'best_dcdicl_gw_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at '{model_path}'. Please train the model first.")
        return

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = f"inference_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    

    shutil.copy('config.yaml', os.path.join(output_dir, 'config_used.yaml'))
    print(f"Results and configuration will be saved to: {output_dir}")

    data_dir = "data_dcdicl_ready"
    test_noisy_path = os.path.join(data_dir, "test_noisy_whitened.npz")
    test_clean_path = os.path.join(data_dir, "test_clean_whitened.npz")
    
    try:
        test_dataset = OfflineGWDataset(test_noisy_path, test_clean_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=2
    )
    
    model = DCDicL_1D(
        iterations=cfg_model['iterations'], 
        num_filters=cfg_model['num_filters'], 
        kernel_size=cfg_model['kernel_size']
    )
    
    # Load the weights into the model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    print(f"Processing {len(test_dataset)} test signals...")

    global_idx = 0
    all_mses = []
    all_overlaps = [] 
    sample_rate = 16384

    # Batch Inference and Direct Saving
    with torch.no_grad():
        for noisy_batch, clean_batch in test_dataloader:
            
            # Asynchronous transfer to GPU
            noisy_batch_gpu = noisy_batch.to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                reconstructed_batch, _ = model(noisy_batch_gpu)

            # Move results back to CPU and convert to numpy all at once
            noisy_np = noisy_batch_gpu.cpu().float().numpy()
            clean_np = clean_batch.cpu().float().numpy()
            recon_np = reconstructed_batch.cpu().float().numpy()

            actual_batch_size = noisy_np.shape[0]

            # Loop through the processed batch to plot and save
            for b in range(actual_batch_size):
                noisy_raw = noisy_np[b].squeeze()
                
                
                clean_true = clean_np[b].squeeze()
                reconstructed_true = recon_np[b].squeeze()

                current_mse = np.mean((clean_true - reconstructed_true) ** 2)
                all_mses.append(float(current_mse))

                norm_clean = np.sqrt(np.sum(clean_true ** 2))
                norm_recon = np.sqrt(np.sum(reconstructed_true ** 2))
                if norm_clean > 0 and norm_recon > 0:
                    current_overlap = np.sum(clean_true * reconstructed_true) / (norm_clean * norm_recon)
                else:
                    current_overlap = 0.0
                all_overlaps.append(float(current_overlap))

                # Calculate the true maximums for visualization scaling
                max_clean_true = np.max(np.abs(clean_true)) + 1e-12

                time_axis = np.arange(len(noisy_raw)) / sample_rate


                fig = plt.figure(figsize=(12, 8))
                                
                # Subplot 1: Noisy vs Clean
                plt.subplot(2, 1, 1)
                plt.title(f"Wave ID: {global_idx} - Noisy Input")
                plt.plot(time_axis, noisy_raw, color='slateblue', alpha=0.6, label='Noisy Input')
                plt.plot(time_axis, clean_true, color='grey', alpha=0.8, linestyle='--', label='Clean Target')
                plt.xlabel("Time (seconds)")
                plt.grid(True, alpha=0.2)
                plt.legend(loc="upper right")

                # Subplot 2: Reconstruction vs Clean
                plt.subplot(2, 1, 2)
                plt.title(f"Wave ID: {global_idx} - DCDicL Recon vs Target | MSE: {current_mse:.2e} | Overlap: {current_overlap:.4f}")
                plt.plot(time_axis, reconstructed_true / max_clean_true, color='dodgerblue', label='Reconstructed')
                plt.plot(time_axis, clean_true / max_clean_true, color='grey', alpha=0.5, linestyle='--', label='Clean Target')
                plt.xlabel("Time (seconds)") 
                plt.grid(True, alpha=0.2)
                plt.legend(loc="upper right")

                plt.tight_layout()
                
                file_name = f"reconstruction_{global_idx:04d}.png"
                plt.savefig(os.path.join(output_dir, file_name), dpi=150)
                
                plt.close(fig)

                global_idx += 1
                
                if global_idx % 50 == 0:
                    print(f"Progress: {global_idx}/{len(test_dataset)} waves saved.")


    mean_mse = np.mean(all_mses)
    mean_overlap = np.mean(all_overlaps)

    print("\n" + "="*50)
    print(f"Average MSE: {mean_mse:.6e}")
    print(f"Average Overlap: {mean_overlap:.6f}")
    print("="*50)

    metrics_path = os.path.join(output_dir, "test_metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.safe_dump({
            "mean_mse": float(mean_mse),  
            "mean_overlap": float(mean_overlap), 
            "total_waves": len(all_mses)
        }, f)
        
    print(f"\nAll files and metrics have been saved to the directory: {output_dir}")

if __name__ == "__main__":
    run_batch_inference()