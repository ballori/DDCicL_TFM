import torch
import torch.nn as nn
from modules.dictionary import Dictionary1D

class HypaNet1D(nn.Module):
    """
    Predicts dynamic hyperparameters (alpha, beta) based on the noise level.
    Now accepts max_alpha and max_beta to control the scaling bounds from config.json.
    """
    def __init__(self, iterations, max_alpha=2.0, max_beta=0.15):
        super().__init__()
        self.iterations = iterations
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, iterations * 2)
        )
    def forward(self, sigma):
        raw_params = self.mlp(sigma)

        raw_params = raw_params.view(-1, self.iterations, 2)

        # Alpha (Gradient Step): Allow it to rise up to max_alpha so Z can grow and escape zero.
        alpha = torch.sigmoid(raw_params[:, :, 0]) * self.max_alpha
        # Beta (Noise Threshold): Keep it small (up to max_beta).
        # It will only kill the low-level static, letting the strong 'chirp' peaks pass.
        beta = torch.sigmoid(raw_params[:, :, 1]) * self.max_beta
        return torch.stack([alpha, beta], dim=-1)

class NetX_1D(nn.Module):
    """
    Deep Prior with explicit Soft-Thresholding.
    Forces sparsity to mathematically kill the noise before refining the signal.
    """
    def __init__(self, num_filters):
        super().__init__()
        # Refinement network (learns the complex shapes and structures of the GW)
        self.refinement = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        )

    def forward(self, z, beta):
        # Soft-Thresholding
        # Physically kills small latent coefficients (which belong to pure noise)
        # guided by the dynamic 'beta' parameter for each specific iteration.
        z_sparse = torch.sign(z) * torch.relu(torch.abs(z) - beta)

        residual = self.refinement(z_sparse)

        # Return the purified sparse signal + a touch of the refinement
        return z_sparse + residual * 0.1

class DCDicL_1D(nn.Module):
    """Main DCDicL Architecture for 1D Signals."""

    def __init__(self, iterations=8, num_filters=200, kernel_size=127, lambda_reg=0.1, max_alpha=2.0, max_beta=0.15):
        super().__init__()
        self.iterations = iterations
        self.num_filters = num_filters
        self.lambda_reg = lambda_reg

        # Dictionary Modules (from modules/dictionary.py)
        self.dictionary = Dictionary1D(num_filters=num_filters, out_channels=1, kernel_size=kernel_size)
        self.dictionary_T = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding='same', bias=False)

        # DCDicL Sub-networks: Pass the max values to HypaNet
        self.hypanet = HypaNet1D(iterations, max_alpha, max_beta)
        self.net_x_blocks = nn.ModuleList([NetX_1D(num_filters) for _ in range(iterations)])

    def init_dictionary_with_data(self, signal_pool, patch_min=32):
        """
        (Optional/Deprecated) Initializes dictionary weights using actual data patches.
        Usually skipped in favor of PyTorch's default Kaiming initialization.
        """
        print("Initializing DCDicL Dictionary and its transpose...")
        self.dictionary.initialize_from_data(signal_pool, patch_min=patch_min)

        with torch.no_grad():
            weights_D = self.dictionary.D.weight.data
            weights_DT = weights_D.transpose(0, 1)
            self.dictionary_T.weight.copy_(weights_DT)
        print("D and D_T initialized and synchronized successfully.")

    def forward(self, y, sigma_est=None):
        batch_size, _, length = y.shape
        z = torch.zeros(batch_size, self.num_filters, length, device=y.device)

        # Estimate noise standard deviation if not provided
        if sigma_est is None:
            sigma_est = torch.std(y, dim=2, keepdim=True).view(batch_size, 1)

        # Predict iteration-specific hyperparameters
        hyperparams = self.hypanet(sigma_est)

        for t in range(self.iterations):
            alpha_t = hyperparams[:, t, 0].view(batch_size, 1, 1)
            beta_t = hyperparams[:, t, 1].view(batch_size, 1, 1)

            # Data Term (Gradient step)
            # Updates the latent representation Z towards the input signal
            residual = self.dictionary(z) - y
            grad = self.dictionary_T(residual)
            z_prime = z - alpha_t * grad

            # Prior Term (NetX step) 
            # Applies sparsity and refinement to filter out the noise
            z = self.net_x_blocks[t](z_prime, beta_t)

        y_reconstructed = self.dictionary(z)
        return y_reconstructed, z 
