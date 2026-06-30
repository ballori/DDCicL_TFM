import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import yaml
import csv
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
    
    # Path to the trained model weights
    model_path = os.path.join('checkpoints', 'best_dcdicl_gw_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at '{model_path}'. Training must have failed.")

    # Prepare the Test Dataset and DataLoader
    data_dir = "data_dcdicl_ready"
    test_noisy_path = os.path.join(data_dir, "test_noisy_whitened.npz")
    test_clean_path = os.path.join(data_dir, "test_clean_whitened.npz")
    
    try:
        test_dataset = OfflineGWDataset(test_noisy_path, test_clean_path)
    except FileNotFoundError as e:
        print(e)
        import sys
        sys.exit(1)
    
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
        kernel_size=cfg_model['kernel_size'],
        max_alpha=cfg_model.get('max_alpha', 1.0), 
        max_beta=cfg_model.get('max_beta', 0.5)
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

    # 4. Batch Inference and Metric Calculation
    with torch.no_grad():
        for noisy_batch, clean_batch in test_dataloader:
            
            noisy_batch_gpu = noisy_batch.to(device, non_blocking=True)

            reconstructed_batch, _ = model(noisy_batch_gpu)

            clean_np = clean_batch.cpu().float().numpy()
            recon_np = reconstructed_batch.cpu().float().numpy()

            actual_batch_size = clean_np.shape[0]

            for b in range(actual_batch_size):
                
                # Reverse the 5.0 amplification factor
                clean_true = clean_np[b].squeeze() / 5.0
                reconstructed_true = recon_np[b].squeeze() / 5.0

                # Calculate MSE
                current_mse = np.mean((clean_true - reconstructed_true) ** 2)
                all_mses.append(float(current_mse))

                # Calculate Overlap (Inner Product)
                dot_product = np.sum(clean_true * reconstructed_true)
                norm_clean = np.sqrt(np.sum(clean_true ** 2))
                norm_recon = np.sqrt(np.sum(reconstructed_true ** 2))
                
                # Add small epsilon to avoid division by zero
                current_overlap = dot_product / (norm_clean * norm_recon + 1e-12)
                all_overlaps.append(float(current_overlap))

                global_idx += 1
                
                if global_idx % 500 == 0:
                    print(f"Progress: {global_idx}/{len(test_dataset)} waves processed.")


    mean_mse = np.mean(all_mses)
    mean_overlap = np.mean(all_overlaps)
    
    print("\n" + "="*50)
    print(f"INFERENCE COMPLETE.")
    print(f"Test Set Mean MSE:     {mean_mse:.6e}")
    print(f"Test Set Mean Overlap: {mean_overlap:.4f}")
    print("="*50)

    # Append results to a single CSV file for the parametric study
    csv_file = "parametric_results.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if the file is new
        if not file_exists:
            writer.writerow(['timestamp', 'iterations', 'num_filters', 'kernel_size', 'mean_mse', 'mean_overlap'])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg_model.get('iterations', 'N/A'),
            cfg_model.get('num_filters', 'N/A'),
            cfg_model.get('kernel_size', 'N/A'),
            f"{mean_mse:.6e}",
            f"{mean_overlap:.6f}"
        ])
        
    print(f"\nResults appended to {csv_file}.")

if __name__ == "__main__":
    run_batch_inference()