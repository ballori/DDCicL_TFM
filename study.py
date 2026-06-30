import os
import yaml
import subprocess
import itertools
import csv
import sys


max_alpha_values = [ 10.0] 
max_beta_values = [ 1.5, 5.0, 10.0] 
kernel_size_values = [57, 127, 257]
num_filters_values = [200]
iterations_values = [8]
batch_size_values = [8, 16, 32]
config_path = 'config.yaml'
inference_tracker = 'parametric_results.csv' 
results_file = 'full_parametric_study_results.csv' 
checkpoint_path = os.path.join('checkpoints', 'best_dcdicl_gw_model.pth')

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Max_Alpha', 'Max_Beta', 
        'Kernel_Size', 'Num_Filters', 'Iterations', 'Batch_Size', 
        'Mean_MSE', 'Mean_Overlap'
    ])

combinations = list(itertools.product(
    max_alpha_values, max_beta_values,
    kernel_size_values, num_filters_values, iterations_values, batch_size_values
))
total_runs = len(combinations)

print(f"Starting parametric study. Total combinations to test: {total_runs}\n")

for i, (m_alpha, m_beta, k_size, n_filters, iters, b_size) in enumerate(combinations):
    print("="*80)
    print(f"Run {i+1}/{total_runs} | m_alpha={m_alpha}, m_beta={m_beta}, "
          f"k_size={k_size}, n_filters={n_filters}, iters={iters}, b_size={b_size}")
    print("="*80)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("-> [Clean] Previous checkpoint removed.")

    # Modify the config.yaml with the current combination of parameters  
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Assign parameters to the "model" block
    if "model" not in config:
        config["model"] = {}
        
    config['model']['max_alpha'] = m_alpha
    config['model']['max_beta'] = m_beta
    config['model']['kernel_size'] = k_size
    config['model']['num_filters'] = n_filters
    config['model']['iterations'] = iters

    # Assign batch_size
    if "training" not in config:
        config["training"] = {}
    config['training']['batch_size'] = b_size

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    # Variables to hold the metrics for this run (default to "NaN" in case of failure)
    mean_mse, mean_overlap = "NaN", "NaN"

    # Security block to catch NaN or gradient explosion during training or inference
    try:
        print("-> Training the model...")
        subprocess.run([sys.executable, "train_for_study.py"], check=True)

        print("-> Running inference to calculate metrics...")
        subprocess.run([sys.executable, "inference_for_study.py"], check=True)

        if os.path.exists(inference_tracker):
            with open(inference_tracker, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1: # Ensure there is data beyond the header
                    last_line = lines[-1].strip().split(',')
                    # timestamp, iterations, num_filters, kernel_size, mean_mse, mean_overlap
                    mean_mse = last_line[4]
                    mean_overlap = last_line[5]
        else:
            print(f"Error: {inference_tracker} not found after running inference.")
            
    except subprocess.CalledProcessError:
        print("\n[!] Failure detected (NaN or gradient explosion). It will be registered as NaN in the CSV.\n")
    except Exception as e:
        print(f"\n[!] Unexpected system error: {e}\n")


    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            m_alpha, m_beta, 
            k_size, n_filters, iters, b_size, 
            mean_mse, mean_overlap
        ])
        
    print(f"-> Master record saved: MSE = {mean_mse} | Overlap = {mean_overlap}\n")

print("\nStudy completed.")
print(f"Check the file '{results_file}' to analyze your full DCDicL parameter grid.")