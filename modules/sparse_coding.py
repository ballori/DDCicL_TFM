import torch
import torch.nn as nn

class ISTABlock1D(nn.Module):
    def __init__(self, D_module, D_T_module, threshold_module, step_size=0.001):
        super().__init__()
        self.D = D_module
        self.D_T = D_T_module
        self.proximal = threshold_module
        
        # We set a smaller step size to prevent gradient explosion.
        self.step_size = nn.Parameter(torch.tensor(step_size))

    def forward(self, Y, Z_prev):
        
        # Reconstruct the signal: D * Z    
        Y_est = self.D(Z_prev)
        
        # Calculate the residual (error): Y - D*Z
        residual = Y - Y_est
        
        # Project the error: D^T * residual
        gradiente = self.D_T(residual)
        
        # Update Z using the gradient and the step size
        Z_new = Z_prev + self.step_size * gradiente

        Z_new = self.proximal(Z_new)
        
        return Z_new