#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-Stage Molecular Transformer Model
Stage 1: Unconditional pretraining
Stage 2: Conditional generation with SMILES + RMSF
"""

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer


# Special token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


# ============================================================================
# Basic Components
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """Token embedding with scaling"""

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# ============================================================================
# Encoders
# ============================================================================

class RMSFEncoder(nn.Module):
    """Encode RMSF values to vector representation"""

    def __init__(self, input_dim=30, hidden_dim=256, output_dim=512, dropout=0.1):
        super(RMSFEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, rmsf_values):
        """
        Args:
            rmsf_values: [batch_size, input_dim] RMSF values
        Returns:
            encoded: [batch_size, output_dim] encoded vector
        """
        x = self.fc1(rmsf_values)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


class SMILESEncoder(nn.Module):
    """Encode reference SMILES using Transformer Encoder"""

    def __init__(self, vocab_size, emb_size=512, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1):
        super(SMILESEncoder, self).__init__()
        self.emb_size = emb_size
        self.tok_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: [seq_len, batch_size] token IDs
            src_padding_mask: [batch_size, seq_len] padding mask
        Returns:
            memory: [seq_len, batch_size, emb_size] encoded representation
        """
        src_emb = self.tok_emb(src) * math.sqrt(self.emb_size)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        return memory


# ============================================================================
# Fusion Mechanisms
# ============================================================================

class SimpleConcatFusion(nn.Module):
    """Simple concatenation fusion"""

    def __init__(self, emb_size=512, dropout=0.1):
        super(SimpleConcatFusion, self).__init__()
        self.fusion_layer = nn.Linear(emb_size * 2, emb_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_memory, rmsf_encoded):
        seq_len, batch_size, emb_size = smiles_memory.shape
        rmsf_expanded = rmsf_encoded.unsqueeze(0).expand(seq_len, -1, -1)
        concat = torch.cat([smiles_memory, rmsf_expanded], dim=-1)
        fused = self.fusion_layer(concat)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class AdditionFusion(nn.Module):
    """Addition fusion with projection"""

    def __init__(self, emb_size=512, dropout=0.1):
        super(AdditionFusion, self).__init__()
        self.rmsf_proj = nn.Linear(emb_size, emb_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_memory, rmsf_encoded):
        seq_len, batch_size, emb_size = smiles_memory.shape
        rmsf_proj = self.rmsf_proj(rmsf_encoded)
        rmsf_expanded = rmsf_proj.unsqueeze(0).expand(seq_len, -1, -1)
        fused = smiles_memory + rmsf_expanded
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class AttentionFusion(nn.Module):
    """Multi-head attention fusion"""

    def __init__(self, emb_size=512, nhead=8, dropout=0.1):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=False)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_memory, rmsf_encoded):
        seq_len, batch_size, emb_size = smiles_memory.shape
        rmsf_expanded = rmsf_encoded.unsqueeze(0).expand(seq_len, -1, -1)
        fused, _ = self.attention(smiles_memory, rmsf_expanded, rmsf_expanded)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused + smiles_memory)
        return fused


class GatedFusion(nn.Module):
    """Gated fusion with learnable weights"""

    def __init__(self, emb_size=512, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.gate_layer = nn.Linear(emb_size * 2, emb_size)
        self.rmsf_proj = nn.Linear(emb_size, emb_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_memory, rmsf_encoded):
        seq_len, batch_size, emb_size = smiles_memory.shape
        rmsf_proj = self.rmsf_proj(rmsf_encoded)
        rmsf_expanded = rmsf_proj.unsqueeze(0).expand(seq_len, -1, -1)
        concat = torch.cat([smiles_memory, rmsf_expanded], dim=-1)
        gate = torch.sigmoid(self.gate_layer(concat))
        fused = gate * smiles_memory + (1 - gate) * rmsf_expanded
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        return fused


class BaselineFusion(nn.Module):
    """Baseline fusion with residual connection"""

    def __init__(self, emb_size=512, dropout=0.1):
        super(BaselineFusion, self).__init__()
        self.rmsf_proj = nn.Linear(emb_size, emb_size)
        self.fusion_layer = nn.Linear(emb_size * 2, emb_size)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, smiles_memory, rmsf_encoded):
        seq_len, batch_size, emb_size = smiles_memory.shape
        rmsf_proj = self.rmsf_proj(rmsf_encoded)
        rmsf_expanded = rmsf_proj.unsqueeze(0).expand(seq_len, -1, -1)
        concat = torch.cat([smiles_memory, rmsf_expanded], dim=-1)
        fused = self.fusion_layer(concat)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused + smiles_memory)
        return fused


def get_fusion_module(fusion_type, emb_size=512, nhead=8, dropout=0.1):
    """Get fusion module by type"""
    fusion_map = {
        'simple_concat': SimpleConcatFusion,
        'addition': AdditionFusion,
        'attention': AttentionFusion,
        'gated': GatedFusion,
        'baseline': BaselineFusion
    }

    if fusion_type not in fusion_map:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Available: {list(fusion_map.keys())}")

    fusion_class = fusion_map[fusion_type]

    if fusion_type == 'attention':
        return fusion_class(emb_size=emb_size, nhead=nhead, dropout=dropout)
    else:
        return fusion_class(emb_size=emb_size, dropout=dropout)


# ============================================================================
# Main Model
# ============================================================================

class TwoStageMolecularTransformer(nn.Module):
    """
    Two-Stage Molecular Generation Transformer
    Stage 1: Unconditional generation with learnable memory
    Stage 2: Conditional generation with reference SMILES + RMSF
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
        rmsf_input_dim: int = 30,
        fusion_type: str = 'simple_concat'
    ):
        super(TwoStageMolecularTransformer, self).__init__()

        self.emb_size = emb_size

        # Token embedding for decoder
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # SMILES encoder (Stage 2)
        self.smiles_encoder = SMILESEncoder(
            vocab_size=src_vocab_size,
            emb_size=emb_size,
            nhead=nhead,
            num_layers=num_encoder_layers if num_encoder_layers > 0 else 3,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # RMSF encoder (Stage 2)
        self.rmsf_encoder = RMSFEncoder(
            input_dim=rmsf_input_dim,
            hidden_dim=256,
            output_dim=emb_size,
            dropout=dropout
        )

        # Condition fusion module
        self.condition_fusion = get_fusion_module(
            fusion_type, emb_size=emb_size, nhead=nhead, dropout=dropout
        )

        # Unconditional memory (Stage 1 - learnable)
        self.unconditional_memory = nn.Parameter(torch.randn(1, 1, emb_size))

        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output layer
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, trg: torch.Tensor, tgt_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                src: torch.Tensor = None, src_padding_mask: torch.Tensor = None,
                rmsf_values: torch.Tensor = None,
                use_conditional: bool = False):
        """
        Forward pass

        Args:
            trg: [seq_len, batch_size] target sequence
            tgt_mask: [seq_len, seq_len] causal mask
            tgt_padding_mask: [batch_size, seq_len] padding mask
            src: [seq_len, batch_size] reference SMILES (for conditional)
            src_padding_mask: [batch_size, seq_len] source padding mask
            rmsf_values: [batch_size, rmsf_dim] RMSF values (for conditional)
            use_conditional: whether to use conditional generation

        Returns:
            logits: [seq_len, batch_size, vocab_size]
        """
        # Token embedding + positional encoding
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        s, b = trg.size()

        if use_conditional and src is not None and rmsf_values is not None:
            # Stage 2: Conditional generation
            smiles_memory = self.smiles_encoder(src, src_padding_mask)
            rmsf_encoded = self.rmsf_encoder(rmsf_values)
            memory = self.condition_fusion(smiles_memory, rmsf_encoded)
        else:
            # Stage 1: Unconditional generation
            memory = self.unconditional_memory.repeat(s, b, 1)

        # Decoder
        outs = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask,
            None,
            tgt_padding_mask
        )

        return self.generator(outs)

    def encode_condition(self, src: torch.Tensor, src_padding_mask: torch.Tensor,
                         rmsf_values: torch.Tensor):
        """
        Encode condition for inference

        Args:
            src: [seq_len, batch_size] reference SMILES
            src_padding_mask: [batch_size, seq_len] padding mask
            rmsf_values: [batch_size, rmsf_dim] RMSF values

        Returns:
            memory: [seq_len, batch_size, emb_size] fused representation
        """
        smiles_memory = self.smiles_encoder(src, src_padding_mask)
        rmsf_encoded = self.rmsf_encoder(rmsf_values)
        memory = self.condition_fusion(smiles_memory, rmsf_encoded)
        return memory

    def decode(self, tgt: torch.Tensor, tgt_mask: torch.Tensor,
               memory: torch.Tensor, tgt_padding_mask: torch.Tensor = None):
        """
        Decode step for inference

        Args:
            tgt: [seq_len, batch_size] target tokens
            tgt_mask: [seq_len, seq_len] causal mask
            memory: [memory_len, batch_size, emb_size] encoded condition
            tgt_padding_mask: [batch_size, seq_len] padding mask

        Returns:
            outputs: [seq_len, batch_size, emb_size]
        """
        return self.transformer_decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            None,
            tgt_padding_mask
        )


# ============================================================================
# Utility Functions
# ============================================================================

def generate_square_subsequent_mask(sz: int, device='cuda'):
    """Generate causal mask for decoder"""
    mask = torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)
    return mask


def create_mask(tgt, pad_idx=PAD_IDX, device='cuda'):
    """Create target mask and padding mask"""
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return tgt_mask, tgt_padding_mask


if __name__ == '__main__':
    print("Two-Stage Molecular Transformer Model")
    print("Import this module to use the model")
