import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from src.helpers import apply_rotary_emb, norm
import torch.distributed as dist


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class MLA(nn.Module):
    def __init__(self, n_embd, n_heads, head_size, rope_head_size, kv_latent_size, q_latent_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = head_size
        self.rope_head_size = rope_head_size
        self.latent_head_size = head_size - rope_head_size
        self.kv_latent_size = kv_latent_size
        
        self.w_down_q = nn.Linear(n_embd, q_latent_size, bias=False)
        self.w_kva = nn.Linear(n_embd, kv_latent_size + rope_head_size, bias=False)

        self.q_norm = nn.RMSNorm(q_latent_size, dtype=torch.bfloat16)
        self.kv_norm = nn.RMSNorm(kv_latent_size, dtype=torch.bfloat16)

        # q = (q_content and q_rope) per head
        self.w_up_qr = nn.Linear(q_latent_size, n_heads * (self.latent_head_size + rope_head_size), bias=False)
        self.w_up_kv = nn.Linear(kv_latent_size, n_heads * (self.latent_head_size + head_size), bias=False)

        self.proj = nn.Linear(n_heads * head_size, n_embd, bias=False)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True # pyrefly: ignore

    def forward(self, x, sin, cos):
        B, T, _ = x.shape
        H = self.n_heads
        d_c = self.latent_head_size
        d_r = self.rope_head_size
        d = self.head_size

        # --- Q ---
        c_q = self.q_norm(self.w_down_q(x))

        q_lr = self.w_up_qr(c_q).view(B, T, H, d).transpose(1,2)
        q_l = q_lr[..., :d_c]
        q_r = q_lr[..., d_c:]
        q_r = apply_rotary_emb(q_r, sin, cos)
        q = torch.cat((q_l, q_r), dim=-1).contiguous()

        # --- KV ---
        c_kv_rope = self.w_kva(x)
        c_kv = c_kv_rope[..., :self.kv_latent_size]
        c_kr = c_kv_rope[..., self.kv_latent_size:]

        # shared k_rope accross all heads
        k_r = apply_rotary_emb(c_kr.unsqueeze(1), sin, cos)

        c_kv = self.kv_norm(c_kv)
        kv = self.w_up_kv(c_kv).view(B, T, H, d_c + d).transpose(1,2) # (B, H, T, d_c + d)
        k_l = kv[..., :d_c]
        v = kv[..., d_c:]

        k = torch.cat((k_l, k_r.expand(B, H, T, d_r)), dim=-1).contiguous() # (B, H, T, d)
        v = v.contiguous()

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).view(B, T, H * d)
        return self.proj(out)

# Attention (Regular Attention but fast)
class Attention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        assert n_embd % n_heads == 0, f"Embedding dim ({n_embd}) must be divisible by number of heads ({n_heads})."
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.H = n_embd // n_heads # head size
        self.attn = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_SCALE_INIT_FACTOR = True # pyrefly: ignore
        # self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size))

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=-1) # q,k,v each is (B,T,C)
        q = q.view(B, T, self.n_heads, self.H).transpose(1,2) # (B,n_heads,T,H)
        k = k.view(B, T, self.n_heads, self.H).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.H).transpose(1,2)
        # Apply RoPE
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        q, k = norm(q), norm(k)
        # att = q @ k.tranpose(-2,-1) / (1 * math.sqrt(self.H)) # (B,n_heads,T,T)
        # att = att.masked_fill(self.tril[:,:,:T,:T], float("-inf"))
        # out = att @ v # (B,n_heds,T,H)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Abstraction but uses flash att for 20% faster training
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)
    
# # Feed Forward (Deprecated, replaced with SwiGLU, see below)
# class MLP(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()
#         self.n_embd = n_embd
#         self.ffwd = nn.Linear(n_embd, 4 * n_embd)
#         self.gelu = nn.GELU() # NOTE: Compare the speed of approximate and exact version
#         self.proj = nn.Linear(4 * n_embd, n_embd)
#         self.proj.RESIDUAL_SCALE_INIT_FACTOR = True # pyrefly: ignore
    
#     def forward(self, x):
#         x = self.ffwd(x)
#         x = self.gelu(x)
#         x = self.proj(x)
#         return x    

# SwiGLU MLP
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        hidden_dim = int(8 * n_embd / 3)
        self.hidden_dim = (hidden_dim + 255) // 256 * 256 # ensures hidden_dim is divisble by 256 (it will be 2,816 for our n_emb=1024)

        self.gate_proj = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(n_embd, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, n_embd, bias=False)
        self.down_proj.RESIDUAL_SCALE_INIT_FACTOR = True # pyrefly: ignore.  This is for weight initialization


    def forward(self, x):
        gate = F.silu(self.gate_proj(x)) # gate projection (decides what passes)
        value = self.up_proj(x) # up projection (raw computation)
        x = gate * value
        x = self.down_proj(x) # back down to residual stream
        return x


# Block
class Block(nn.Module):
    def __init__(self, n_embd, n_heads, head_size, rope_head_size, kv_latent_size, q_latent_size):
        super().__init__()
        # self.ln1 = nn.LayerNorm(n_embd) # Replaced with RMSNorm for better performance
        self.ln1 = nn.RMSNorm(n_embd)
        self.sa = Attention(n_embd, n_heads)
        # self.sa = MLA(n_embd, n_heads, head_size, rope_head_size, kv_latent_size, q_latent_size)
        # self.ln2 = nn.LayerNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)
    
    def forward(self, x, sin, cos):
        x = x + self.sa(self.ln1(x), sin, cos)
        x = x + self.mlp(self.ln2(x))
        return x

# LLM
class GPT(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, n_heads, head_size, rope_head_size, kv_latent_size, q_latent_size, n_layers):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        # self.wpe = nn.Embedding(block_size, n_embd)  # old learned positional embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)

        sin, cos = self._precompute_rotary_embeddings(block_size, rope_head_size) # pyrefly: ignore
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        self.transformer = nn.ModuleList(
            [Block(n_embd, n_heads, head_size, rope_head_size, kv_latent_size, q_latent_size) for _ in range(n_layers)]
        )
        self.ln = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)
        self.wte.weight = self.lm_head.weight # Embedding layer and final calssifier are the same
        self.apply(self._init_weights)
        self.rank = dist.get_rank()

    # initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # std = 1 / math.sqrt(self.n_embd) is what GPT-3 says
            # But DeepSeek says to go lower!
            std = 0.01
            if hasattr(module, 'RESIDUAL_SCALE_INIT_FACTOR'):
                std *= 1 / (math.sqrt(2 * self.n_layers))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)


    def forward(self, idx, targets=None):
        # select the index and put them together
        B,T = idx.shape
        assert T <= self.block_size, f"Sequence length ({T}) is longer than the block_size ({self.block_size})."
        x = self.wte(idx) # (B,T,C)
        # # old code with learned positional embedding
        # pos_emb = self.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # (T,C)
        # x = tok_emb + pos_emb
        sin = self.sin[:,:,:T,:] # pyrefly: ignore
        cos = self.cos[:,:,:T,:] # pyrefly: ignore

        for block in self.transformer:
            x = block(x, sin, cos)
        x = self.ln(x)

        if targets is not None:
            logits = self.lm_head(x)
            B,T,C = logits.shape
            # cross_entropy expects shape (N,C)
            loss = F.cross_entropy(logits.view(B*T,C), targets.view(B*T))
            return None, loss
        else:
            logits = self.lm_head(x)
            return logits, None

    def generate(self, idx, num_sequences=5, max_tokens=200, topk=50, chat_mode=False, eos_token=50256):
        idx = torch.repeat_interleave(idx.unsqueeze(0), num_sequences, dim=0)

        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :] # pyrefly: ignore
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=topk)
            idx_next = torch.multinomial(topk_probs, num_samples=1)
            idx_next = torch.gather(topk_indices, -1, idx_next)

            if chat_mode and (idx_next == eos_token).all():
                break

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


    def configure_optimizers(self, weight_decay, learning_rate, device):
            # Gather all params ensuring no duplicates (critical for tied weights like wte/lm_head)
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            unique_params = []
            seen = set()
            for p in param_dict.values():
                if p not in seen:
                    seen.add(p)
                    unique_params.append(p)

            # Split into decay (2D+ tensors) and no-decay (1D tensors like biases/norms)
            decay_params = [p for p in unique_params if p.dim() >= 2]
            nodecay_params = [p for p in unique_params if p.dim() < 2]
            
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0},
            ]

            # Print debug stats (rank 0 only)
            if self.rank == 0:
                num_decay = sum(p.numel() for p in decay_params)
                num_nodecay = sum(p.numel() for p in nodecay_params)
                print(f"Decayed params (2D): {len(decay_params)} tensors, {num_decay:,} parameters")
                print(f"Non-decayed params (1D): {len(nodecay_params)} tensors, {num_nodecay:,} parameters")

            # Configure fused AdamW if available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in str(device)
            if self.rank == 0:
                print(f"Using fused AdamW: {use_fused}")

            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
            return optimizer

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequency at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.bfloat16(), cos.bfloat16()
        sin, cos = sin[None, None, :, :], cos[None, None, :, :]
        return sin, cos
