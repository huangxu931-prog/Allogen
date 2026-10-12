#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conditional Molecular Generation with Two-Stage Transformer
Main script for training and inference
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from torch.utils.data import DataLoader

import dataset_two_stage as ds
import model_two_stage as model
import vocabulary as mv
from utils import idx2word

# ============================================================================
# Configuration
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Two-Stage Conditional Molecular Generation')

    # Mode selection
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['pretrain', 'finetune', 'infer_uncond', 'infer_cond'],
                        help='Operation mode')

    # Model architecture
    parser.add_argument('--layer', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model embedding dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Feedforward network dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Conditional encoding
    parser.add_argument('--fusion_type', type=str, default='simple_concat',
                        choices=['baseline', 'simple_concat', 'addition', 'attention', 'gated'],
                        help='Condition fusion mechanism')
    parser.add_argument('--rmsf_input_dim', type=int, default=30,
                        help='RMSF feature dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')

    # Decoding strategy
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Top-k sampling parameter')

    # Data paths
    parser.add_argument('--train_path', type=str, default='train.smi',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='test.smi',
                        help='Test data path')
    parser.add_argument('--data_path', type=str, default='example_conditions.csv',
                        help='Conditional data path for inference')

    # Model paths
    parser.add_argument('--path', type=str, default='model_epoch_50.h5',
                        help='Model checkpoint path')
    parser.add_argument('--loadmodel', action='store_true',
                        help='Load existing model')

    # Inference parameters
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of conditions to use')
    parser.add_argument('--samples_per_condition', type=int, default=10,
                        help='Number of molecules to generate per condition')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


# ============================================================================
# Data Loading
# ============================================================================

def load_vocabulary():
    """Load or create vocabulary"""
    voc_path = "Voc"
    if os.path.exists(voc_path):
        voc = mv.Vocabulary(init_from_file=voc_path)
        print(f"✓ Loaded vocabulary: {len(voc)} tokens")
    else:
        print("✗ Vocabulary file not found. Please run pretraining first.")
        sys.exit(1)
    return voc


def load_data(args, voc):
    """Load training and test data"""
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)

    train_data = ds.get_tensor_dataset(args.train_path)
    test_data = ds.get_tensor_dataset(args.test_path)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ds.SequenceCollator(voc)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ds.SequenceCollator(voc)
    )

    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    print(f"✓ Batch size: {args.batch_size}")

    return train_loader, test_loader


def load_conditional_data(args):
    """Load conditional generation data"""
    if not os.path.exists(args.data_path):
        print(f"✗ Data file not found: {args.data_path}")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    print(f"✓ Loaded {len(df)} conditions from {args.data_path}")

    # Use subset if specified
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.head(args.num_samples)
        print(f"✓ Using first {args.num_samples} conditions")

    return df


# ============================================================================
# Model Creation
# ============================================================================

def create_model(args, voc):
    """Create model based on configuration"""
    print("\n" + "="*70)
    print("Model Configuration")
    print("="*70)

    net = model.TransformerVAE(
        voc,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layer,
        num_decoder_layers=args.layer,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=128,
        device=args.device,
        fusion_type=args.fusion_type,
        rmsf_input_dim=args.rmsf_input_dim
    ).to(args.device)

    # Load pretrained weights if specified
    if args.loadmodel and os.path.exists(args.path):
        net.load_state_dict(torch.load(args.path, map_location=args.device))
        print(f"✓ Loaded model from: {args.path}")
    elif args.loadmodel:
        print(f"✗ Model file not found: {args.path}")
        sys.exit(1)
    else:
        print("✓ Created new model")

    # Print model info
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Layers: {args.layer}")
    print(f"✓ d_model: {args.d_model}")
    print(f"✓ Attention heads: {args.nhead}")
    print(f"✓ FFN dimension: {args.dim_feedforward}")
    print(f"✓ Fusion type: {args.fusion_type}")

    return net


# ============================================================================
# Training
# ============================================================================

def train_epoch(net, train_loader, optimizer, criterion, args):
    """Train for one epoch"""
    net.train()
    total_loss = 0

    for batch_idx, (src, tgt, lengths, rmsf) in enumerate(train_loader):
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        rmsf = rmsf.to(args.device)

        optimizer.zero_grad()

        # Forward pass
        output = net(src, tgt[:, :-1], rmsf)

        # Calculate loss
        output = output.reshape(-1, output.size(-1))
        target = tgt[:, 1:].reshape(-1)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(net, test_loader, criterion, args):
    """Evaluate on test set"""
    net.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt, lengths, rmsf in test_loader:
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            rmsf = rmsf.to(args.device)

            output = net(src, tgt[:, :-1], rmsf)
            output = output.reshape(-1, output.size(-1))
            target = tgt[:, 1:].reshape(-1)
            loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_model(args):
    """Main training loop"""
    print("\n" + "="*70)
    print(f"Starting Training - {args.mode.upper()}")
    print("="*70)

    # Load data and model
    voc = load_vocabulary()
    train_loader, test_loader = load_data(args, voc)
    net = create_model(args, voc)

    # Setup training
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=voc.tk2ix['_'])

    print(f"\n✓ Optimizer: Adam (lr={args.lr})")
    print(f"✓ Loss function: CrossEntropyLoss")
    print(f"✓ Training epochs: {args.epoch}")
    print(f"✓ Device: {args.device}")

    # Training loop
    print("\n" + "="*70)
    print("Training Progress")
    print("="*70)

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epoch + 1):
        epoch_start = time.time()

        # Train and evaluate
        train_loss = train_epoch(net, train_loader, optimizer, criterion, args)
        test_loss = evaluate(net, test_loader, criterion, args)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch:3d}/{args.epoch} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            save_path = f"model_epoch_{epoch}.h5"
            torch.save(net.state_dict(), save_path)
            print(f"  → Saved best model: {save_path}")

    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best test loss: {best_loss:.4f}")
    print("="*70)


# ============================================================================
# Molecular Property Calculation
# ============================================================================

def calculate_properties(smiles_list):
    """Calculate molecular properties"""
    results = {
        'valid': [],
        'qed': [],
        'mw': [],
        'logp': []
    }

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            results['valid'].append(True)
            results['qed'].append(QED.qed(mol))
            results['mw'].append(Descriptors.MolWt(mol))
            results['logp'].append(Descriptors.MolLogP(mol))
        else:
            results['valid'].append(False)
            results['qed'].append(None)
            results['mw'].append(None)
            results['logp'].append(None)

    return results


# ============================================================================
# Inference - Unconditional
# ============================================================================

def infer_unconditional(args):
    """Generate molecules without conditions"""
    print("\n" + "="*70)
    print("Unconditional Generation")
    print("="*70)

    voc = load_vocabulary()
    net = create_model(args, voc)
    net.eval()

    num_samples = args.num_samples
    print(f"✓ Generating {num_samples} molecules")
    print(f"✓ Temperature: {args.temperature}")
    print(f"✓ Top-k: {args.top_k}")

    generated = []

    with torch.no_grad():
        for i in range(num_samples):
            # Random condition (not used in baseline)
            rmsf = torch.randn(1, args.rmsf_input_dim).to(args.device)

            # Generate
            smiles = net.sample(
                1,
                rmsf,
                temp=args.temperature,
                top_k=args.top_k
            )

            generated.extend(smiles)

            if (i + 1) % 10 == 0:
                print(f"  Generated: {i + 1}/{num_samples}")

    # Calculate properties
    print("\nCalculating molecular properties...")
    props = calculate_properties(generated)

    # Create results dataframe
    df = pd.DataFrame({
        'SMILES': generated,
        'Valid': props['valid'],
        'QED': props['qed'],
        'MW': props['mw'],
        'LogP': props['logp']
    })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_uncond_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    # Print statistics
    valid_count = sum(props['valid'])
    valid_rate = valid_count / len(generated) * 100

    print("\n" + "="*70)
    print("Generation Statistics")
    print("="*70)
    print(f"Total generated: {len(generated)}")
    print(f"Valid molecules: {valid_count} ({valid_rate:.1f}%)")

    if valid_count > 0:
        valid_qed = [q for q, v in zip(props['qed'], props['valid']) if v]
        valid_mw = [m for m, v in zip(props['mw'], props['valid']) if v]
        valid_logp = [l for l, v in zip(props['logp'], props['valid']) if v]

        print(f"Average QED: {np.mean(valid_qed):.3f}")
        print(f"Average MW: {np.mean(valid_mw):.1f}")
        print(f"Average LogP: {np.mean(valid_logp):.2f}")

    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)


# ============================================================================
# Inference - Conditional
# ============================================================================

def infer_conditional(args):
    """Generate molecules with conditions"""
    print("\n" + "="*70)
    print("Conditional Generation")
    print("="*70)

    voc = load_vocabulary()
    net = create_model(args, voc)
    net.eval()

    # Load conditions
    df_cond = load_conditional_data(args)

    print(f"✓ Samples per condition: {args.samples_per_condition}")
    print(f"✓ Temperature: {args.temperature}")
    print(f"✓ Top-k: {args.top_k}")

    # Generate molecules
    all_results = []

    print("\n" + "="*70)
    print("Generating Molecules")
    print("="*70)

    for idx, row in df_cond.iterrows():
        print(f"\nCondition {idx + 1}/{len(df_cond)}")
        print(f"  LogP: {row['logP']:.2f}, QED: {row['QED']:.3f}, MW: {row['MW']:.1f}")

        # Prepare RMSF condition
        rmsf_values = row.iloc[1:].values.astype(np.float32)
        rmsf = torch.from_numpy(rmsf_values).unsqueeze(0).to(args.device)

        # Generate molecules
        generated = []
        with torch.no_grad():
            for _ in range(args.samples_per_condition):
                smiles = net.sample(
                    1,
                    rmsf,
                    temp=args.temperature,
                    top_k=args.top_k
                )
                generated.extend(smiles)

        # Calculate properties
        props = calculate_properties(generated)

        # Store results
        for i, smi in enumerate(generated):
            all_results.append({
                'Condition_ID': idx,
                'Target_LogP': row['logP'],
                'Target_QED': row['QED'],
                'Target_MW': row['MW'],
                'SMILES': smi,
                'Valid': props['valid'][i],
                'Generated_QED': props['qed'][i],
                'Generated_MW': props['mw'][i],
                'Generated_LogP': props['logp'][i]
            })

        valid_count = sum(props['valid'])
        print(f"  Generated: {len(generated)} molecules ({valid_count} valid, {valid_count/len(generated)*100:.1f}%)")

    # Save results
    df_results = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_cond_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)

    # Print statistics
    print("\n" + "="*70)
    print("Generation Statistics")
    print("="*70)

    total = len(df_results)
    valid_df = df_results[df_results['Valid'] == True]
    valid_count = len(valid_df)

    print(f"Total generated: {total}")
    print(f"Valid molecules: {valid_count} ({valid_count/total*100:.1f}%)")

    if valid_count > 0:
        print(f"Average QED: {valid_df['Generated_QED'].mean():.3f}")
        print(f"Average MW: {valid_df['Generated_MW'].mean():.1f}")
        print(f"Average LogP: {valid_df['Generated_LogP'].mean():.2f}")

        # Calculate diversity
        unique_smiles = valid_df['SMILES'].nunique()
        diversity = unique_smiles / valid_count * 100
        print(f"Unique molecules: {unique_smiles} ({diversity:.1f}%)")

    print(f"\n✓ Results saved to: {output_file}")
    print("="*70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function"""
    args = parse_arguments()

    print("\n" + "="*70)
    print("Conditional Molecular Generation")
    print("Two-Stage Transformer Model")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")

    if args.mode == 'pretrain':
        print("\nNote: Pretraining mode requires large dataset")
        train_model(args)

    elif args.mode == 'finetune':
        train_model(args)

    elif args.mode == 'infer_uncond':
        infer_unconditional(args)

    elif args.mode == 'infer_cond':
        infer_conditional(args)

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
