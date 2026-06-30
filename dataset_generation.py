import numpy as np
import os
import yaml
import math
import random
import pandas as pd
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
try:
    import ccphen
except ImportError:
    print("WARNING: ccphen module not found. 'ccphen' data_type will fail if requested.")


class hlmClass:
    h20 = np.zeros(1)
    h21 = np.zeros(1)
    h22 = np.zeros(1)

class DatasetGenerator:
    def __init__(self, clean_file_path=None, catalog_df=None, data_type="npy", num_samples=1000, 
                 allowed_keys=None, sample_rate=16384, seq_length=4096, noise_type="ligo", 
                 ligo_asd_path=None, ccphen_params=None):
        
        self.sample_rate = sample_rate
        self.L = seq_length  
        
        if data_type == "npy":
            print(f"Loading clean signals from NPY: {clean_file_path}")
            self.clean_signals = np.load(clean_file_path)
            
        elif data_type == "h5":
            print(f"Extracting physical projections from Pandas DataFrame...")
            if catalog_df is None or allowed_keys is None:
                raise ValueError("catalog_df and allowed_keys must be provided for DataFrame generation.")
            self.clean_signals = self._extract_from_dataframe(catalog_df, num_samples, allowed_keys)
            
        elif data_type == "ccphen":
            print(f"Generating {num_samples} signals using ccphen...")
            if ccphen_params is None:
                ccphen_params = {}
            self.clean_signals = self._generate_from_ccphen(num_samples, ccphen_params)
            
        else:
            raise ValueError("data_type must be 'npy', 'h5', or 'ccphen'")
        
        
        self.noise_type = noise_type.lower()
        self.asd_interp = None
        
        if self.noise_type == "ligo" and ligo_asd_path and os.path.exists(ligo_asd_path):
            asd_data = np.loadtxt(ligo_asd_path)
            freqs_ligo = asd_data[:, 0]
            asd_vals_ligo = asd_data[:, 1]
            self.asd_interp = interp1d(freqs_ligo, asd_vals_ligo, bounds_error=False, fill_value=np.inf)
        else:
            if self.noise_type == "ligo":
                print(f"WARNING: ASD file not found at '{ligo_asd_path}'. Using flat spectrum (Gaussian).")
            self.noise_type = "gaussian"

    def _generate_from_ccphen(self, num_samples, params):
        """Generates random waveforms using the phenomenological ccphen library."""
        generated_signals = np.zeros((num_samples, self.L), dtype=np.float32)
        dist = params.get('dist', 10000)
        base_seed = params.get('base_seed', 42)
        w_type = params.get('waveform_type', 'standard')
        
        # Generate 2 full seconds of signal to ensure the explosion occurs and is recorded in the array
        gen_length = int(self.sample_rate * 2.0) 
        
        for i in range(num_samples):
            # Unique deterministic seed for this specific sample
            seed = base_seed + i
            
            if w_type == 'sasi':
                wf = ccphen.GenerateRandomWaveformStandardNeutrinoDrivenSASI(
                    dist=dist, fs=self.sample_rate, N=gen_length, seed=seed)
            elif w_type == 'short':
                wf = ccphen.GenerateRandomWaveformShortNeutrinoDriven(
                    dist=dist, fs=self.sample_rate, N=gen_length, seed=seed)
            else:
                wf = ccphen.GenerateRandomWaveformStandardNeutrinoDriven(
                    dist=dist, fs=self.sample_rate, N=gen_length, seed=seed)
            
            # Extract real part (h_plus)
            strain = np.real(wf.h)
            
            
            # Center the 4096 sample window on the signal peak
            aligned_strain = np.zeros(self.L)
            
            if len(strain) > 0 and np.max(np.abs(strain)) > 0:
                # Find the "core bounce"
                peak_idx = np.argmax(np.abs(strain))
                
                # Jittering: force the peak to fall randomly between 15% and 75% of the window
                min_offset = int(self.L * 0.15)
                max_offset = int(self.L * 0.75)
                random_offset = random.randint(min_offset, max_offset)
                
                start_idx = max(0, peak_idx - random_offset)
                end_idx = start_idx + self.L
                
                # If the cut goes beyond the 2 seconds, readjust it
                if end_idx > len(strain):
                    end_idx = len(strain)
                    start_idx = max(0, end_idx - self.L)
                    
                actual_len = end_idx - start_idx
                aligned_strain[:actual_len] = strain[start_idx:end_idx]
                
            generated_signals[i] = aligned_strain
            
        return generated_signals

    def _compute_strain_from_hlm(self, sim_row, theta, phi):
        """
        Extracts the physical strain using the spherical harmonics formulas
        from the SNWaveforms catalog.
        """
        if 'modehlm' in sim_row and sim_row['modehlm'] == True and 'hlm' in sim_row:
            hlm = sim_row['hlm']
            
            # Spin-weighted spherical harmonics with s=-2 
            y20 = 1/4 * np.sqrt(15/(2*math.pi)) * np.sin(theta)
            y2p1 = 1/8 * np.sqrt(5/math.pi) * (2*np.sin(theta)+np.sin(2*theta)) * np.exp(+1j * phi)
            y2m1 = 1/8 * np.sqrt(5/math.pi) * (2*np.sin(theta)-np.sin(2*theta)) * np.exp(-1j * phi)
            y2p2 = 1/16 * np.sqrt(5/math.pi) * (3+4*np.cos(theta)+np.cos(2*theta)) * np.exp(+1j * 2 * phi)
            y2m2 = 1/16 * np.sqrt(5/math.pi) * (3-4*np.cos(theta)+np.cos(2*theta)) * np.exp(-1j * 2 * phi)
            
            # Calculate the complex h wave using the relation h_l-m = (-)^m h_lm^*
            h = hlm.h20*y20 + hlm.h21*y2p1 + hlm.h22*y2p2 - np.conj(hlm.h21)*y2m1 + np.conj(hlm.h22)*y2m2
            
            # The 1D neural network uses a single real channel, we take the "plus" polarization
            strain = np.real(h)
        else:
            # Fallback, if the model does not have hlm, extract the polar (hpol) or equatorial component
            try:
                strain = sim_row['hpol']
            except KeyError:
                strain = np.zeros(self.L)


        aligned_strain = np.zeros(self.L)        
        
        if hasattr(strain, '__len__') and len(strain) > 0:
            peak_idx = np.argmax(np.abs(strain))
            
            min_offset = int(self.L * 0.15)
            max_offset = int(self.L * 0.75)
            random_offset = random.randint(min_offset, max_offset)
            
            start_idx = max(0, peak_idx - random_offset)
            end_idx = start_idx + self.L
            
            if end_idx > len(strain):
                end_idx = len(strain)
                start_idx = max(0, end_idx - self.L)
                
            actual_len = end_idx - start_idx
            aligned_strain[:actual_len] = strain[start_idx:end_idx]

        return aligned_strain

    def _extract_from_dataframe(self, df, num_samples, allowed_keys):
        print(f"Generating {num_samples} angular projections from {len(allowed_keys)} authorized models...")
        generated_signals = np.zeros((num_samples, self.L), dtype=np.float32)
        
        for i in range(num_samples):
            sim_idx = random.choice(allowed_keys)
            sim_row = df.loc[sim_idx]
            
            # Data augmentation, uniform random observation angle over the sphere
            theta = math.acos(random.uniform(-1, 1))
            phi = random.uniform(0, 2 * math.pi)
            
            hp = self._compute_strain_from_hlm(sim_row, theta, phi)
            generated_signals[i] = hp
                
        return generated_signals

    def get_psd_asd(self, freqs):
        if self.noise_type == "ligo" and self.asd_interp is not None:
            asd = self.asd_interp(freqs)
            asd[asd <= 0] = 1e-20 
            return asd
        else:
            asd = np.ones_like(freqs)
            asd[freqs < 20] = 1e5
            asd[freqs > 1000] = 1e2
            return asd

    def process_and_inject(self, snr_range, lambda_base=0.001):
        N_signals, L = self.clean_signals.shape
        
        processed_noisy = np.zeros((N_signals, L), dtype=np.float32)
        processed_clean = np.zeros((N_signals, L), dtype=np.float32)
        
        # New arrays to store the variance, hyperparameter, and injected SNR for each signal
        noise_variances = np.zeros(N_signals, dtype=np.float32)
        lambda_regs = np.zeros(N_signals, dtype=np.float32)
        injected_snrs = np.zeros(N_signals, dtype=np.float32)
        
        dt = 1.0 / self.sample_rate
        freqs = np.fft.rfftfreq(L, d=dt)
        asd = self.get_psd_asd(freqs)
        window = tukey(L, alpha=0.1)

        print(f"Injecting noise into {N_signals} signals with random SNR between {snr_range[0]} and {snr_range[1]}...")
        
        for i in range(N_signals):
            clean_signal = self.clean_signals[i]

            # Whitening Process
            signal_windowed = clean_signal * window
            fft_signal = np.fft.rfft(signal_windowed)
            
            whitened_fft = fft_signal / asd
            whitened_time = np.fft.irfft(whitened_fft, n=L)
            
            # SNR Scaling
            current_snr = np.sqrt(np.sum(whitened_time**2))
            target_snr = np.random.uniform(snr_range[0], snr_range[1])
            
            # Track the injected SNR
            injected_snrs[i] = target_snr
            
            scale_factor = target_snr / (current_snr + 1e-20)
            
            clean_whitened_scaled = whitened_time * scale_factor
            
            # Inject Noise
            noise = np.random.normal(0, 1, L)
            noisy_signal = clean_whitened_scaled + noise
            
            # Normalization
            max_val = np.max(np.abs(noisy_signal)) + 1e-10
            noisy_norm = noisy_signal / max_val
            clean_norm = clean_whitened_scaled / max_val

            # Calculate noise variance and corresponding lambda for this specific signal
            effective_noise = noise / max_val 
            var_noise = np.var(effective_noise)
            
            noise_variances[i] = var_noise
            lambda_regs[i] = lambda_base * var_noise
            
            #clean_amplified = clean_norm * 1.0
            
            processed_noisy[i] = noisy_norm
            processed_clean[i] = clean_norm

        return processed_noisy, processed_clean, noise_variances, lambda_regs, injected_snrs


if __name__ == "__main__":
    config_path = 'config.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
    else:
        print("Loading configuration from YAML...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        cfg_data = config['dataset']
        snr_range = (cfg_data['snr_min'], cfg_data['snr_max'])
        noise_type = cfg_data.get('noise_type', 'gaussian')
        ligo_asd_path = cfg_data.get('ligo_asd_path', '')
        data_type = cfg_data.get('data_type', 'npy')
        
        lambda_base = cfg_data.get('lambda_base', 0.001)
        
        fs = cfg_data.get('fs', 4096)
        seq_length = cfg_data.get('seq_length', 4096)
        
        output_dir = "data_dcdicl_ready"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Generating Offline Datasets for DCDicL (SNR: {snr_range}, Data Type: {data_type}) ---")

        # CASE 2: GENERATE FROM PANDAS HDF5 CATALOG
        if data_type == "h5":
            h5_path = cfg_data.get('h5_catalog_path')
            print(f"Reading full DataFrame catalog from: {h5_path}")
            df_catalog = pd.read_hdf(h5_path)
            all_sims = df_catalog.index.tolist()
            
            random.seed(42)
            random.shuffle(all_sims)
            
            total_sims = len(all_sims)
            train_idx = int(total_sims * cfg_data['splits']['train'])
            val_idx = train_idx + int(total_sims * cfg_data['splits']['val'])
            
            split_keys = {
                "train": all_sims[:train_idx],
                "val": all_sims[train_idx:val_idx],
                "test": all_sims[val_idx:]
            }
            
            print(f"Catalog split -> Train: {len(split_keys['train'])}, Val: {len(split_keys['val'])}, Test: {len(split_keys['test'])} models.")
            
            for split_name in ["train", "val", "test"]:
                print(f"\n[{split_name.upper()} SPLIT]")
                num_samples = cfg_data['samples_per_split'][split_name]
                
                generator = DatasetGenerator(
                    catalog_df=df_catalog,
                    data_type="h5",
                    num_samples=num_samples,
                    allowed_keys=split_keys[split_name],
                    sample_rate=fs,
                    seq_length=seq_length,
                    noise_type=noise_type,
                    ligo_asd_path=ligo_asd_path
                )
                
                noisy, clean, noise_vars, lambda_regs, injected_snrs = generator.process_and_inject(
                    snr_range=snr_range, lambda_base=lambda_base
                )
                
                noisy_out_path = os.path.join(output_dir, f"{split_name}_noisy_whitened.npz")
                clean_out_path = os.path.join(output_dir, f"{split_name}_clean_whitened.npz")
                
                # Include the new arrays in the .npz archive
                np.savez(noisy_out_path, strains=noisy, noise_vars=noise_vars, lambda_regs=lambda_regs, snrs=injected_snrs)
                np.savez(clean_out_path, strains=clean)
                print(f"-> Saved dataset to {output_dir}")

        # CASE 1: ORIGINAL NPY FILES BEHAVIOR
        elif data_type == "npy":
            datasets_to_process = {
                "train": cfg_data.get('train_path'),
                "val": cfg_data.get('val_path'),
                "test": cfg_data.get('test_path')
            }
            
            for split_name, input_path in datasets_to_process.items():
                if input_path and os.path.exists(input_path):
                    print(f"\n[{split_name.upper()} SPLIT]")
                    generator = DatasetGenerator(
                        clean_file_path=input_path,
                        data_type="npy",
                        sample_rate=fs,
                        seq_length=seq_length,
                        noise_type=noise_type,
                        ligo_asd_path=ligo_asd_path
                    )
                    
                    noisy, clean, noise_vars, lambda_regs, injected_snrs = generator.process_and_inject(
                        snr_range=snr_range, lambda_base=lambda_base
                    )
                    
                    noisy_out_path = os.path.join(output_dir, f"{split_name}_noisy_whitened.npz")
                    clean_out_path = os.path.join(output_dir, f"{split_name}_clean_whitened.npz")
                    
                    # Include the new arrays in the .npz archive
                    np.savez(noisy_out_path, strains=noisy, noise_vars=noise_vars, lambda_regs=lambda_regs, snrs=injected_snrs)
                    np.savez(clean_out_path, strains=clean)
                    print(f"-> Saved dataset to {output_dir}")

        # CASE 3: CCPHEN PHENOMENOLOGICAL GENERATOR
        elif data_type == "ccphen":
            ccphen_params = cfg_data.get('ccphen_params', {})
            base_seed = ccphen_params.get('base_seed', 42)
            
            for split_name in ["train", "val", "test"]:
                print(f"\n[{split_name.upper()} SPLIT]")
                num_samples = cfg_data['samples_per_split'][split_name]
                
                # Offset to ensure the seeds for train, val, and test are completely different, preventing data leakage between the datasets.
                seed_offset = {"train": 0, "val": 100000, "test": 200000}[split_name]
                
                split_params = ccphen_params.copy()
                split_params['base_seed'] = base_seed + seed_offset
                
                generator = DatasetGenerator(
                    data_type="ccphen",
                    num_samples=num_samples,
                    sample_rate=fs,
                    seq_length=seq_length,
                    noise_type=noise_type,
                    ligo_asd_path=ligo_asd_path,
                    ccphen_params=split_params
                )
                
                noisy, clean, noise_vars, lambda_regs, injected_snrs = generator.process_and_inject(
                    snr_range=snr_range, lambda_base=lambda_base
                )
                
                noisy_out_path = os.path.join(output_dir, f"{split_name}_noisy_whitened.npz")
                clean_out_path = os.path.join(output_dir, f"{split_name}_clean_whitened.npz")
                
                # Include the new arrays in the .npz archive
                np.savez(noisy_out_path, strains=noisy, noise_vars=noise_vars, lambda_regs=lambda_regs, snrs=injected_snrs)
                np.savez(clean_out_path, strains=clean)
                print(f"-> Saved dataset to {output_dir}")