import torch
import torch.nn as nn
import numpy as np

class Dictionary1D(nn.Module):
    def __init__(self, num_filters=200, out_channels=1, kernel_size=128, seed=42):
        """
        1D Convolutional Dictionary Module for DCDicL. This module represents the dictionary operator (D) in the CLAWDIA-inspired architecture.
        
        Args:
            num_filters (int): Equivalent to 'd_size' (200 atoms).
            out_channels (int): Output channels (usually 1 for 1D signals).
            kernel_size (int): Equivalent to 'a_length' (128 samples per atom).
            seed (int): Equivalent to 'random_state' for reproducibility.
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        # Dictionary Operator (D)
        # Maps from Z (coefficients: num_filters channels) to Y (signal: 1 channel)
        self.D = nn.Conv1d(
            in_channels=num_filters, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding='same', 
            bias=False
        )

    def initialize_from_data(self, signal_pool, patch_min=32):
        """
        Data-driven initialization. Overwrites random PyTorch weights by extracting 
        real patches from the training signals, mimicking CLAWDIA's behavior.
        """
        print(f"Initializing dictionary with {self.num_filters} atoms of length {self.kernel_size}...")
        
        num_signals, signal_length = signal_pool.shape
        new_weights = np.zeros((1, self.num_filters, self.kernel_size))
        
        atoms_found = 0
        attempts = 0
        max_attempts = self.num_filters * 100 
        
        while atoms_found < self.num_filters and attempts < max_attempts:
            attempts += 1
            
            # Pick a random signal and a random starting point
            sig_idx = np.random.randint(0, num_signals)
            signal = signal_pool[sig_idx]
            start_idx = np.random.randint(0, signal_length - self.kernel_size)
            patch = signal[start_idx : start_idx + self.kernel_size]
            
            # Dynamic threshold: calculate 5% of the peak amplitude of this specific signal
            peak_val = np.max(np.abs(signal))
            dynamic_threshold = 0.05 * peak_val if peak_val > 0 else 1e-10
            
            # CLAWDIA's patch_min condition with the dynamic threshold
            if np.count_nonzero(np.abs(patch) > dynamic_threshold) >= patch_min:
                
                # L2 normalization
                norm = np.linalg.norm(patch)
                if norm > 0:
                    patch = patch / norm
                
                new_weights[0, atoms_found, :] = patch
                atoms_found += 1
                
        if atoms_found < self.num_filters:
            print(f"WARNING: Only found {atoms_found}/{self.num_filters} valid patches after {max_attempts} attempts.")
            print("The remaining atoms will be kept as zeros.")
            
        with torch.no_grad():
            self.D.weight.copy_(torch.from_numpy(new_weights).float())
            
        print(f"-> Dictionary initialized successfully in {attempts} attempts!")

    def forward(self, x):
        return self.D(x)