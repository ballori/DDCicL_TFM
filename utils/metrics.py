import torch
import torch.nn as nn

class DCDicLLoss(nn.Module):
    """
    Custom Loss for Gravitational Waves DCDicL.
    Combines L1 Base reconstruction with an MSE penalty for peaks/chirps,
    AND a critical L1 Sparsity penalty on the activation maps to filter out noise.
    """
    def __init__(self, peak_weight=0.5, sparsity_weight=0.1):
        super(DCDicLLoss, self).__init__()
        self.recon_criterion = nn.L1Loss() 
        self.peak_criterion = nn.MSELoss()
        self.peak_weight = peak_weight
        self.sparsity_weight = sparsity_weight # ¡NUEVO: Peso para la esparsidad!

    def forward(self, target_signals, reconstructed_output, z_maps=None):
        # 1. Base L1 Loss (as per DCDicL theoretical framework)
        loss_base = self.recon_criterion(reconstructed_output, target_signals)
        
        # 2. Peak / Chirp penalty (MSE is more sensitive to large outliers/peaks)
        loss_peaks = self.peak_criterion(reconstructed_output, target_signals)
        
        # 3. Sparsity Penalty (L1 on z_maps): CRITICAL for low SNR noise removal
        # Esto obliga a la red a usar la menor cantidad de filtros posibles, matando el ruido.
        loss_sparsity = 0.0
        if z_maps is not None:
            # Sumamos el valor absoluto (L1) de las activaciones en cada etapa de la red
            for z in z_maps:
                loss_sparsity += torch.mean(torch.abs(z))
        
        # 4. Total Loss
        total_loss = loss_base + (self.peak_weight * loss_peaks) + (self.sparsity_weight * loss_sparsity)
        
        # Devolvemos 3 valores para mantener la compatibilidad con tu train.py 
        # (que hace unpack con: loss, _, _ = criterion(...))
        return total_loss, (loss_base + self.peak_weight * loss_peaks), loss_sparsity