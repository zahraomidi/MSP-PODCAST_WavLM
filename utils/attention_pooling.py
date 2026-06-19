import torch
import torch.nn as nn

class MultiHeadSelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling over time.

    Input:
        x    : [B, T, D]
        lens : [B] (relative 0..1 or absolute length), optional

    Output:
        pooled: [B, D]  (multi-head scores, heads averaged)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(max(1, num_heads))
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.input_dim
        self.temperature = float(temperature) if temperature is not None else 1.0
        self.dropout = nn.Dropout(p=float(dropout) if dropout is not None else 0.0)

        # score network: x -> tanh(W1 x) -> W2 -> H scores
        self.proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(self.hidden_dim, self.num_heads)

    def _build_mask(self, x: torch.Tensor, lens: torch.Tensor | None) -> torch.Tensor:
        """
        Build [B, T] boolean mask from lens.

        If lens is float in (0,1], treat as relative.
        If lens is integer, treat as absolute frame count.
        Ensures at least one valid frame per sample.
        """
        B, T, _ = x.shape
        device = x.device

        if lens is None:
            return torch.ones(B, T, dtype=torch.bool, device=device)

        if isinstance(lens, torch.Tensor):
            lens_t = lens.to(device)
        else:
            lens_t = torch.tensor(lens, device=device)

        if lens_t.dtype.is_floating_point:
            lengths = (lens_t * T).ceil().long()
        else:
            lengths = lens_t.long()

        lengths = torch.clamp(lengths, min=1, max=T)

        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = idxs < lengths.unsqueeze(1)

        # Safety: if anything ends up all-false, force first frame valid
        all_empty = ~mask.any(dim=1)
        if all_empty.any():
            mask[all_empty, 0] = True

        return mask

    def forward(self, x: torch.Tensor, lens: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : [B, T, D]
        lens : [B] (relative or absolute), optional
        """
        if x.ndim != 3:
            raise RuntimeError(f"[AttnPool] Expected x [B,T,D], got shape={x.shape}")

        B, T, D = x.shape
        if D != self.input_dim:
            # If this fires, your ssl_hidden_dim vs encoder feature dim are inconsistent.
            raise RuntimeError(
                f"[AttnPool] input_dim mismatch: expected {self.input_dim}, got {D}"
            )

        mask = self._build_mask(x, lens)  # [B, T]

        # score frames
        h = self.act(self.proj(x))       # [B, T, H]
        logits = self.out(h)              # [B, T, num_heads]

        if self.temperature is not None and self.temperature != 1.0:
            logits = logits / self.temperature

        # mask padding
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # attention over time
        attn = torch.softmax(logits, dim=1)   # [B, T, num_heads]
        attn = self.dropout(attn)

        # weighted sum per head: [B, num_heads, D]
        pooled = torch.einsum("bth,btd->bhd", attn, x)

        # average heads -> [B, D]
        pooled = pooled.mean(dim=1)

        # LayerNorm for stability
        pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))

        return pooled