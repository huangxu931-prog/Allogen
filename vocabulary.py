#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vocabulary and tokenization utilities for SMILES strings
"""

import re
import numpy as np


# ============================================================================
# Vocabulary Class
# ============================================================================

class Vocabulary:
    """Stores tokens and their conversion to vocabulary indices"""

    def __init__(self, tokens=None, starting_id=0):
        """
        Initialize vocabulary

        Args:
            tokens: Dict mapping tokens to indices
            starting_id: Starting index for new tokens
        """
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        """Get token by ID or ID by token"""
        return self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        """Check if token/ID exists in vocabulary"""
        return token_or_id in self._tokens

    def __len__(self):
        """Get vocabulary size"""
        return len(self._tokens) // 2

    def __eq__(self, other_vocabulary):
        """Check equality with another vocabulary"""
        return self._tokens == other_vocabulary._tokens

    def add(self, token):
        """
        Add a single token to vocabulary

        Args:
            token: String token to add

        Returns:
            Index of the token
        """
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """
        Add multiple tokens to vocabulary

        Args:
            tokens: List of tokens to add

        Returns:
            List of token indices
        """
        return [self.add(token) for token in tokens]

    def encode(self, tokens):
        """
        Encode tokens as vocabulary indices

        Args:
            tokens: List of token strings

        Returns:
            numpy array of indices
        """
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """
        Decode vocabulary indices to tokens

        Args:
            vocab_index: Array of indices

        Returns:
            List of token strings
        """
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def tokens(self):
        """
        Get all tokens in vocabulary

        Returns:
            List of token strings
        """
        return [t for t in self._tokens if isinstance(t, str)]

    def _add(self, token, idx):
        """Internal method to add token with specific index"""
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("Index already present in vocabulary")


# ============================================================================
# SMILES Tokenizer
# ============================================================================

class SMILESTokenizer:
    """Tokenize and untokenize SMILES strings"""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """
        Tokenize a SMILES string

        Args:
            data: SMILES string
            with_begin_and_end: Whether to add start/end tokens

        Returns:
            List of tokens
        """
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["$"] + tokens + ["^"]
        return tokens

    def untokenize(self, tokens):
        """
        Untokenize back to SMILES string

        Args:
            tokens: List of tokens

        Returns:
            SMILES string
        """
        smi = ""
        for token in tokens:
            if token == "^":
                break
            if token != "$":
                smi += token
        return smi


# ============================================================================
# Vocabulary Creation
# ============================================================================

def create_vocabulary(smiles_list, tokenizer=None):
    """
    Create vocabulary from SMILES list

    Args:
        smiles_list: List of SMILES strings
        tokenizer: SMILESTokenizer instance (creates new if None)

    Returns:
        Vocabulary object
    """
    if tokenizer is None:
        tokenizer = SMILESTokenizer()

    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["<pad>", "$", "^"] + sorted(tokens))
    return vocabulary


def save_vocabulary(vocabulary, file_path):
    """
    Save vocabulary to file

    Args:
        vocabulary: Vocabulary object
        file_path: Output file path
    """
    with open(file_path, 'w') as f:
        for token in vocabulary.tokens():
            idx = vocabulary[token]
            f.write(f"{token}\t{idx}\n")


def load_vocabulary(file_path):
    """
    Load vocabulary from file

    Args:
        file_path: Input file path

    Returns:
        Vocabulary object
    """
    tokens = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                token, idx = parts
                tokens[token] = int(idx)

    return Vocabulary(tokens)


if __name__ == '__main__':
    print("Vocabulary and tokenization utilities for SMILES")
    print("Import this module to use vocabulary functions")
