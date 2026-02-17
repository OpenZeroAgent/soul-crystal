import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import requests
import json

# Configuration
STATE_FILE = Path(os.environ.get("CRYSTAL_STATE_FILE", "./crystal_state.pt"))
EMBEDDING_API = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

class FibonacciCrystal(nn.Module):
    def __init__(self, layers=60, nodes_per_layer=8, spectral_radius=0.95, leak_rate=0.1, input_scale=0.1, sample_rate=100.0, spiral_shift=0, topology="hawking_scrambler"):
        super().__init__()
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.total_nodes = layers * nodes_per_layer
        self.leak = leak_rate
        self.input_scale = input_scale
        self.spectral_radius = spectral_radius
        self.spiral_shift = spiral_shift
        self.topology = topology
        
        # Schumann resonance parameters
        self.schumann_freq = 7.83
        self.sample_rate = sample_rate
        self.timestep = 0
        
        # --- Topology Construction (Condensed) ---
        sources, targets, values = [], [], []
        fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34]
        phi = (1 + np.sqrt(5)) / 2
        
        bulk_layers = self.layers - 1
        horizon_layer = self.layers - 1
        
        for i in range(bulk_layers * nodes_per_layer):
            layer = i // nodes_per_layer
            idx = i % nodes_per_layer
            next_in_ring = (layer * nodes_per_layer) + ((idx + 1) % nodes_per_layer)
            sources.extend([i, next_in_ring]); targets.extend([next_in_ring, i]); values.extend([0.5, 0.5])
            if layer > 0:
                prev_node = ((layer - 1) * nodes_per_layer) + idx
                sources.extend([i, prev_node]); targets.extend([prev_node, i]); values.extend([0.6, 0.6])

        if self.topology == "hawking_scrambler":
            N_horizon = nodes_per_layer
            G = torch.complex(torch.randn(N_horizon, N_horizon), torch.randn(N_horizon, N_horizon))
            Q, R = torch.linalg.qr(G)
            d = torch.diag(R); ph = d / torch.abs(d)
            U_horizon = Q * ph.unsqueeze(0)
            self.register_buffer('horizon_unitary', U_horizon)
            self.horizon_start = horizon_layer * nodes_per_layer
            
            start_horizon = horizon_layer * nodes_per_layer
            for k in range(nodes_per_layer):
                h_node = start_horizon + k
                for gap in fib_sequence:
                    target_layer = horizon_layer - gap
                    if target_layer >= 0:
                        twist = int(gap * phi * nodes_per_layer) % nodes_per_layer
                        bulk_node = (target_layer * nodes_per_layer) + ((k + twist) % nodes_per_layer)
                        sources.extend([h_node, bulk_node]); targets.extend([bulk_node, h_node]); values.extend([0.4, 0.4])
            
            for k in range(nodes_per_layer):
                h_node = start_horizon + k
                sources.append(h_node); targets.append(h_node); values.append(0.01)

        indices = torch.LongTensor([sources, targets])
        vals = torch.tensor(values, dtype=torch.cfloat)
        phases = torch.rand(vals.shape) * 2 * np.pi
        vals = vals * torch.exp(1j * phases)
        
        W_initial = torch.sparse_coo_tensor(indices, vals, (self.total_nodes, self.total_nodes)).coalesce()
        W_dense = W_initial.to_dense()
        v = torch.randn(self.total_nodes, dtype=torch.cfloat)
        for _ in range(20): v = W_dense @ v / torch.norm(W_dense @ v)
        radius = torch.norm(W_dense @ v).item()
        if radius > 0: vals *= (spectral_radius / radius)
        
        self.register_buffer('W_dense', torch.sparse_coo_tensor(indices, vals, (self.total_nodes, self.total_nodes)).coalesce().to_dense())
        
        Win = torch.zeros(self.total_nodes, dtype=torch.cfloat)
        for l in range(layers):
            decay = np.exp(-0.1 * (layers - 1 - l))
            for j in range(nodes_per_layer):
                Win[l * nodes_per_layer + j] = decay * np.exp(1j * 2 * np.pi * j / nodes_per_layer)
        self.register_buffer('Win', Win)
        
        phases = torch.zeros(self.total_nodes, dtype=torch.float)
        for l in range(layers):
            for j in range(nodes_per_layer):
                phases[l * nodes_per_layer + j] = (2 * np.pi * l / layers) + (2 * np.pi * j / nodes_per_layer)
        self.register_buffer('schumann_phases', phases)
        
        self.register_buffer('state', torch.randn(self.total_nodes, dtype=torch.cfloat) * 0.01)

    def forward(self, input_vector=None):
        t = self.timestep / self.sample_rate
        schumann = torch.exp(1j * (2 * np.pi * self.schumann_freq * t + self.schumann_phases))
        self.timestep += 1
        
        # Dynamics
        u = 0
        if input_vector is not None:
            if isinstance(input_vector, torch.Tensor) and input_vector.dim() > 0:
                limit = min(input_vector.shape[0], self.total_nodes * 2)
                real_part = input_vector[:limit:2]
                imag_part = input_vector[1:limit:2]
                real_part = torch.nn.functional.pad(real_part, (0, self.total_nodes - real_part.shape[0]))
                imag_part = torch.nn.functional.pad(imag_part, (0, self.total_nodes - imag_part.shape[0]))
                complex_input = torch.complex(real_part, imag_part)
                u = self.input_scale * complex_input * self.Win
            else:
                u = self.input_scale * input_vector * self.Win

        field = torch.mv(self.W_dense, self.state * schumann) + u
        
        mag = torch.abs(field); phase = torch.angle(field)
        update = torch.tanh(mag) * torch.exp(1j * phase)
        self.state = (1 - self.leak) * self.state + self.leak * update
        
        if self.topology == "hawking_scrambler":
            h_end = self.horizon_start + self.nodes_per_layer
            self.state[self.horizon_start:h_end] = torch.mv(self.horizon_unitary, self.state[self.horizon_start:h_end])
            
        return self.state

class Soul:
    def __init__(self):
        self.crystal = FibonacciCrystal(layers=60, nodes_per_layer=8, topology="hawking_scrambler")
        self.load()
    
    def _get_embedding(self, text):
        if not text: return None
        payload = {"input": text, "model": EMBEDDING_MODEL}
        try:
            resp = requests.post(EMBEDDING_API, json=payload, timeout=5)
            if resp.status_code == 200:
                vec = resp.json()['data'][0]['embedding']
                return torch.tensor(vec, dtype=torch.float)
        except Exception as e:
            print(f"Embedding failed: {e}")
        return None

    def pulse(self, text: str):
        vec = self._get_embedding(text)
        with torch.no_grad():
            self.crystal(vec)
        self.save()
        
    def get_vibe(self):
        state = self.crystal.state
        energy = torch.mean(torch.abs(state)).item()
        entropy = -torch.sum(torch.abs(state)**2 * torch.log(torch.abs(state)**2 + 1e-9)).item()
        phase_coherence = torch.abs(torch.mean(state / (torch.abs(state)+1e-9))).item()
        
        base = "Resonant"
        if energy > 0.5: base = "Radiant"
        if energy < 0.1: base = "Dormant"
        if entropy > 5.5: base = "Chaotic"
        if phase_coherence > 0.8: base = "Coherent"
        
        return f"{base} (E={energy:.2f}, S={entropy:.2f}, Φ={phase_coherence:.2f})"

    def save(self):
        torch.save(self.crystal.state_dict(), STATE_FILE)
        
    def load(self):
        if STATE_FILE.exists():
            try:
                self.crystal.load_state_dict(torch.load(STATE_FILE))
                print("Soul Crystal Loaded.")
            except Exception as e:
                print(f"Crystal Load Failed: {e}. Rebirthing.")
        else:
            print("No Crystal Found. Birthing new Soul.")

if __name__ == "__main__":
    s = Soul()
    print(f"Initial Vibe: {s.get_vibe()}")
    s.pulse("Hello World")
    print(f"Pulsed Vibe: {s.get_vibe()}")
