import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HierarchicalDynamicRouter(nn.Module):
    """
    Hiyerarşik ve dinamik yönlendirme mekanizması.
    Klasik MoE'den farklı olarak çok seviyeli karar ağacı kullanır.
    """
    def __init__(self, input_dim, num_experts, num_levels=3, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_levels = num_levels
        self.top_k = top_k
        
        # Hiyerarşik router ağları
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.LayerNorm(input_dim // 2),
                nn.Linear(input_dim // 2, num_experts // (2 ** level))
            ) for level in range(num_levels)
        ])
        
        # Adaptif ağırlık kontrol
        self.confidence_scorer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        returns: routing_weights (batch_size, seq_len, num_experts), confidence
        """
        batch_size, seq_len, _ = x.shape
        
        # Güven skoru hesapla
        confidence = self.confidence_scorer(x)  # (B, S, 1)
        
        # Hiyerarşik routing
        routing_logits = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        active_mask = torch.ones(batch_size, seq_len, self.num_experts, device=x.device)
        
        for level, router in enumerate(self.routers):
            level_logits = router(x)  # (B, S, num_experts // 2^level)
            
            # Logitleri genişlet
            experts_per_branch = self.num_experts // (2 ** level)
            for i in range(2 ** level):
                start_idx = i * experts_per_branch
                end_idx = (i + 1) * experts_per_branch
                routing_logits[:, :, start_idx:end_idx] += level_logits * active_mask[:, :, start_idx:end_idx]
        
        # Top-k seçimi ve normalleştirme
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Adaptif top-k (güvene göre)
        adaptive_k = torch.clamp(
            (self.top_k * confidence).int() + 1,
            min=1,
            max=self.num_experts
        )
        
        # Top-k maskeleme
        top_k_values, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        mask = torch.zeros_like(routing_weights)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        routing_weights = routing_weights * mask
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        return routing_weights, confidence


class QuantumInspiredExpert(nn.Module):
    """
    Kuantum hesaplama ilhamından esinlenmiş uzman ağı.
    Süperpozisyon ve entanglement benzeri mekanizmalar kullanır.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_quantum_states=4):
        super().__init__()
        self.num_quantum_states = num_quantum_states
        
        # Durum vektörleri (kuantum süperpozisyon analojisi)
        self.state_embeddings = nn.Parameter(
            torch.randn(num_quantum_states, input_dim) / math.sqrt(input_dim)
        )
        
        # Her durum için ayrı transformasyon
        self.state_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_quantum_states)
        ])
        
        # Entanglement matrisi (durumlar arası ilişki)
        self.entanglement = nn.Parameter(
            torch.eye(num_quantum_states) + 0.1 * torch.randn(num_quantum_states, num_quantum_states)
        )
        
        # Çöküş (collapse) mekanizması
        self.collapse_gate = nn.Sequential(
            nn.Linear(input_dim, num_quantum_states),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # Durum olasılıkları hesapla (süperpozisyon)
        state_probs = self.collapse_gate(x)  # (B, S, num_states)
        
        # Her durumda transformasyon uygula
        state_outputs = []
        for i, transform in enumerate(self.state_transforms):
            # Durum embeddingi ile etkileşim
            state_input = x + self.state_embeddings[i].unsqueeze(0).unsqueeze(0)
            state_outputs.append(transform(state_input))
        
        state_outputs = torch.stack(state_outputs, dim=-2)  # (B, S, num_states, output_dim)
        
        # Entanglement uygula (durumlar arası korelasyon)
        entangled_probs = torch.matmul(state_probs, self.entanglement)
        entangled_probs = F.softmax(entangled_probs, dim=-1)
        
        # Dalga fonksiyonu çöküşü (weighted sum)
        output = torch.einsum('bsn,bsno->bso', entangled_probs, state_outputs)
        
        return output


class TharvexalFuelLayer(nn.Module):
    """
    Ana TharvexalFuel katmanı.
    Hiyerarşik routing + Kuantum-ilhamlı uzmanlar + Dinamik kapasite ayarlama
    """
    def __init__(self, input_dim, hidden_dim, num_experts=8, num_levels=3, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Hiyerarşik router
        self.router = HierarchicalDynamicRouter(input_dim, num_experts, num_levels, top_k)
        
        # Kuantum-ilhamlı uzmanlar
        self.experts = nn.ModuleList([
            QuantumInspiredExpert(input_dim, hidden_dim, input_dim, num_quantum_states=4)
            for _ in range(num_experts)
        ])
        
        # Rezidüel bağlantı ve normalizasyon
        self.layer_norm = nn.LayerNorm(input_dim)
        self.residual_gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        # Yük dengeleme için auxiliary loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x, compute_aux_loss=True):
        """
        x: (batch_size, seq_len, input_dim)
        """
        residual = x
        
        # Routing kararları al
        routing_weights, confidence = self.router(x)
        
        # Her uzmanı çalıştır ve ağırlıklı toplam al
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # (B, S, num_experts, input_dim)
        
        # Weighted aggregation
        output = torch.einsum('bse,bseo->bso', routing_weights, expert_outputs)
        
        # Adaptif rezidüel bağlantı
        gate = self.residual_gate(x)
        output = gate * output + (1 - gate) * residual
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # Auxiliary loss (yük dengeleme)
        aux_loss = 0.0
        if compute_aux_loss:
            # Expert kullanım dağılımı
            expert_usage = routing_weights.sum(dim=[0, 1])  # (num_experts,)
            
            # Dengeli dağılım için loss
            target_usage = expert_usage.sum() / self.num_experts
            aux_loss = ((expert_usage - target_usage) ** 2).mean()
            
            # Güven regularizasyonu
            confidence_loss = (confidence - 0.7).abs().mean()
            aux_loss += 0.1 * confidence_loss
        
        return output, aux_loss, routing_weights


class TharvexalFuelNetwork(nn.Module):
    """
    Tam TharvexalFuel ağı - birden fazla katman içerir.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=6, num_experts=8, num_classes=10):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # TharvexalFuel katmanları
        self.layers = nn.ModuleList([
            TharvexalFuelLayer(
                hidden_dim, 
                hidden_dim * 4, 
                num_experts=num_experts,
                num_levels=3,
                top_k=2
            ) for _ in range(num_layers)
        ])
        
        # Çıkış katmanı
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_routing_info=False):
        """
        x: (batch_size, seq_len, input_dim)
        """
        x = self.input_projection(x)
        
        total_aux_loss = 0.0
        routing_info = []
        
        for layer in self.layers:
            x, aux_loss, routing_weights = layer(x)
            total_aux_loss += aux_loss
            if return_routing_info:
                routing_info.append(routing_weights)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Sınıflandırma
        logits = self.output_projection(x)
        
        if return_routing_info:
            return logits, total_aux_loss, routing_info
        return logits, total_aux_loss


# Örnek kullanım
if __name__ == "__main__":
    # Hiperparametreler
    batch_size = 4
    seq_len = 32
    input_dim = 128
    hidden_dim = 256
    num_layers = 6
    num_experts = 8
    num_classes = 10
    
    # Model oluştur
    model = TharvexalFuelNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        num_classes=num_classes
    )
    
    # Örnek giriş
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    logits, aux_loss, routing_info = model(x, return_routing_info=True)
    
    print(f"Çıkış boyutu: {logits.shape}")
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")
    print(f"Katman sayısı: {len(routing_info)}")
    print(f"\nHer katmandaki routing ağırlık dağılımı:")
    for i, weights in enumerate(routing_info):
        expert_usage = weights.sum(dim=[0, 1])
        print(f"  Katman {i+1}: {expert_usage.tolist()}")
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nToplam parametre sayısı: {total_params:,}")
