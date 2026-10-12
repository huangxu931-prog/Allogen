#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset utilities for two-stage molecular generation
Handles SMILES tokenization, RMSF features, and batch collation
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from torch.utils.data import Dataset


# ============================================================================
# RMSF Calculation
# ============================================================================

def calculate_rmsf(smiles, num_points=30):
    """
    Calculate RMSF (Root Mean Square Fluctuation) profile for a molecule

    Args:
        smiles: SMILES string
        num_points: Number of points in RMSF profile

    Returns:
        numpy array of RMSF values
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(num_points, dtype=np.float32)

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return np.zeros(num_points, dtype=np.float32)

    # Generate synthetic RMSF based on molecular properties
    # In practice, this would come from MD simulations
    base_rmsf = np.random.randn(num_atoms) * 0.5 + 1.0
    base_rmsf = np.abs(base_rmsf)

    # Interpolate to fixed number of points
    if num_atoms < num_points:
        # Pad with zeros
        rmsf = np.zeros(num_points, dtype=np.float32)
        rmsf[:num_atoms] = base_rmsf
    else:
        # Downsample
        indices = np.linspace(0, num_atoms - 1, num_points, dtype=int)
        rmsf = base_rmsf[indices]

    return rmsf.astype(np.float32)


def calculate_properties(smiles):
    """
    Calculate molecular properties

    Args:
        smiles: SMILES string

    Returns:
        dict with logP, QED, MW
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'logP': 0.0, 'QED': 0.0, 'MW': 0.0}

    return {
        'logP': Descriptors.MolLogP(mol),
        'QED': QED.qed(mol),
        'MW': Descriptors.MolWt(mol)
    }


# ============================================================================
# Dataset Classes
# ============================================================================

class SMILESDataset(Dataset):
    """Dataset for SMILES strings with RMSF features"""

    def __init__(self, smiles_list, rmsf_dim=30):
        """
        Args:
            smiles_list: List of SMILES strings
            rmsf_dim: Dimension of RMSF features
        """
        self.smiles_list = smiles_list
        self.rmsf_dim = rmsf_dim

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        rmsf = calculate_rmsf(smiles, self.rmsf_dim)
        return smiles, rmsf


class SequenceCollator:
    """Collate function for batching SMILES sequences"""

    def __init__(self, vocabulary, rmsf_dim=30):
        """
        Args:
            vocabulary: Vocabulary object for tokenization
            rmsf_dim: Dimension of RMSF features
        """
        self.vocabulary = vocabulary
        self.rmsf_dim = rmsf_dim

    def __call__(self, batch):
        """
        Collate a batch of SMILES strings

        Args:
            batch: List of (smiles, rmsf) tuples

        Returns:
            Tuple of (src_tensor, tgt_tensor, lengths, rmsf_tensor)
        """
        smiles_list = [item[0] for item in batch]
        rmsf_list = [item[1] for item in batch]

        # Tokenize SMILES
        tokenized = [self.vocabulary.tokenize(smi) for smi in smiles_list]

        # Get lengths
        lengths = [len(tokens) for tokens in tokenized]
        max_len = max(lengths)

        # Convert to indices and pad
        src_batch = []
        tgt_batch = []

        for tokens in tokenized:
            # Add start and end tokens
            src_seq = [self.vocabulary.tk2ix['GO']] + \
                     [self.vocabulary.tk2ix.get(tk, self.vocabulary.tk2ix['_']) for tk in tokens]
            tgt_seq = [self.vocabulary.tk2ix.get(tk, self.vocabulary.tk2ix['_']) for tk in tokens] + \
                     [self.vocabulary.tk2ix['EOS']]

            # Pad to max length
            pad_idx = self.vocabulary.tk2ix['_']
            src_seq += [pad_idx] * (max_len + 1 - len(src_seq))
            tgt_seq += [pad_idx] * (max_len + 1 - len(tgt_seq))

            src_batch.append(src_seq)
            tgt_batch.append(tgt_seq)

        # Convert to tensors
        src_tensor = torch.LongTensor(src_batch)
        tgt_tensor = torch.LongTensor(tgt_batch)
        lengths_tensor = torch.LongTensor(lengths)
        rmsf_tensor = torch.FloatTensor(np.array(rmsf_list))

        return src_tensor, tgt_tensor, lengths_tensor, rmsf_tensor


# ============================================================================
# Data Loading Functions
# ============================================================================

def read_smiles_file(file_path, max_samples=None):
    """
    Read SMILES from file

    Args:
        file_path: Path to SMILES file (.smi)
        max_samples: Maximum number of samples to read

    Returns:
        List of SMILES strings
    """
    smiles_list = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            line = line.strip()
            if line:
                # Extract SMILES (first column)
                smiles = line.split()[0] if ' ' in line or '\t' in line else line

                # Validate
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_list.append(smiles)

    return smiles_list


def get_tensor_dataset(file_path, rmsf_dim=30, max_samples=None):
    """
    Create dataset from SMILES file

    Args:
        file_path: Path to SMILES file
        rmsf_dim: Dimension of RMSF features
        max_samples: Maximum number of samples

    Returns:
        SMILESDataset object
    """
    smiles_list = read_smiles_file(file_path, max_samples)
    return SMILESDataset(smiles_list, rmsf_dim)


def load_conditional_data(file_path, num_samples=None):
    """
    Load conditional generation data

    Args:
        file_path: Path to CSV file with conditions
        num_samples: Number of samples to load

    Returns:
        DataFrame with conditions
    """
    df = pd.read_csv(file_path)

    if num_samples and num_samples < len(df):
        df = df.head(num_samples)

    return df


def prepare_training_data(smiles_file, output_file, rmsf_dim=30):
    """
    Prepare training data with RMSF and properties

    Args:
        smiles_file: Input SMILES file
        output_file: Output CSV file
        rmsf_dim: RMSF dimension
    """
    print(f"Reading SMILES from {smiles_file}...")
    smiles_list = read_smiles_file(smiles_file)
    print(f"Loaded {len(smiles_list)} valid SMILES")

    data = []
    for i, smiles in enumerate(smiles_list):
        if (i + 1) % 1000 == 0:
            print(f"Processing: {i + 1}/{len(smiles_list)}")

        # Calculate properties
        props = calculate_properties(smiles)

        # Calculate RMSF
        rmsf = calculate_rmsf(smiles, rmsf_dim)

        # Create row
        row = {
            'SMILES': smiles,
            'logP': props['logP'],
            'QED': props['QED'],
            'MW': props['MW']
        }

        # Add RMSF values
        for j, val in enumerate(rmsf):
            row[f'RMSF_{j}'] = val

        data.append(row)

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved training data to {output_file}")

    return df


# ============================================================================
# Utility Functions
# ============================================================================

def filter_by_properties(smiles_list, min_mw=0, max_mw=600, min_logp=-5, max_logp=5):
    """
    Filter SMILES by molecular properties

    Args:
        smiles_list: List of SMILES strings
        min_mw, max_mw: Molecular weight range
        min_logp, max_logp: LogP range

    Returns:
        Filtered list of SMILES
    """
    filtered = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        if min_mw <= mw <= max_mw and min_logp <= logp <= max_logp:
            filtered.append(smiles)

    return filtered


def split_dataset(smiles_list, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    np.random.seed(seed)

    n = len(smiles_list)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_list = [smiles_list[i] for i in train_idx]
    val_list = [smiles_list[i] for i in val_idx]
    test_list = [smiles_list[i] for i in test_idx]

    return train_list, val_list, test_list


def save_smiles_file(smiles_list, file_path):
    """
    Save SMILES to file

    Args:
        smiles_list: List of SMILES strings
        file_path: Output file path
    """
    with open(file_path, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")

    print(f"Saved {len(smiles_list)} SMILES to {file_path}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    # Example usage
    print("Dataset utilities for molecular generation")
    print("Import this module to use dataset functions")
