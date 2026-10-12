# 🧬 Allogen: A Protein Dynamics-Conditioned Cross-Modal Transformer Framework for Allosteric Drug Design

A molecular generation model adapted from cMolGPT. It supports conditional molecular generation using protein-dynamics features represented by root-mean-square fluctuation (RMSF).

## Key Features

| Feature | cMolGPT | This two-stage model |
| Conditioning | Discrete target ID (0–3) | Continuous 30-dimensional RMSF vector |
| Condition encoder | Embedding layer | MLP encoder |
| Training strategy | End-to-end training | Two stages: pre-training and fine-tuning |
| Inference | Conditional generation | Unconditional and conditional generation |

- **Two-stage training:** unconditional pre-training followed by RMSF-conditioned fine-tuning.
- **Continuous conditioning:** accepts a 30-dimensional RMSF feature vector.
- **Flexible inference:** supports both unconditional and conditional generation.
- **Utilities included:** data preparation, training, evaluation, and demonstration scripts.

## Model Architecture

```text
Stage 1: Unconditional pre-training
SMILES → Tokenization → Embedding → Decoder → Learnable Memory
       → Generator → Token probabilities

Stage 2: RMSF-conditioned generation
30-dimensional RMSF → MLP Encoder → Condition/Memory representation
SMILES → Tokenization → Embedding → Decoder
       → Generator → Token probabilities
```

**Linux/macOS**

### Option : Run each step manually

```bash
cd two_stage_model

# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare example RMSF data
python prepare_rmsf_data.py --type realistic --num_samples 1000

# 3. Run unconditional pre-training
python main_two_stage.py --mode pretrain --path model_pretrain.h5 --epoch 100

# 4. Fine-tune using conditional training data
python main_two_stage.py --mode finetune --path model_pretrain.h5 \
    --data_path conditional_train_data.csv --loadmodel --epoch 50

# 5. Generate molecules with the fine-tuned model
# Use the first three conditions and generate five molecules per condition
python main_two_stage.py --mode infer_cond --path model_pretrain_finetuned.h5 \
    --data_path conditional_train_data.csv --num_samples 3 --samples_per_condition 5
```

## Usage

### 1. Unconditional Pre-training

```bash
python main_two_stage.py \
    --mode pretrain \
    --device cuda \
    --epoch 100 \
    --batch_size 512 \
    --layer 3 \
    --d_model 512 \
    --nhead 8 \
    --path model_pretrain.h5
```

**Output:** `model_pretrain.h5`, the pre-trained model checkpoint.

### 2. RMSF-Conditioned Fine-tuning

Prepare a CSV file named `conditional_train_data.csv`, for example:

```csv
smiles,rmsf_features
CCOc1ccccc1,"1.2,1.5,1.8,2.1,2.5,1.9,2.3,2.7,3.1,3.5,2.8,3.2,2.4,2.9,3.3,2.6,3.0,2.2,2.6,3.0,3.4,2.8,3.2,2.5,2.9,3.3,3.7,3.1,3.5,3.9"
CC(=O)Oc1ccccc1,"1.1,1.4,1.7,2.0,2.4,1.8,2.2,2.6,3.0,3.4,2.7,3.1,2.3,2.8,3.2,2.5,2.9,2.1,2.5,2.9,3.3,2.7,3.1,2.4,2.8,3.2,3.6,3.0,3.4,3.8"
```

### 3. Unconditional Generation

```bash
python main_two_stage.py \
    --mode infer_uncond \
    --device cuda \
    --path model_pretrain.h5 \
    --num_samples 1
```

This runs generation using the pre-trained model without an RMSF condition.

### 4. Conditional Generation

#### Generate multiple molecules per condition

```bash
python main_two_stage.py \
    --mode infer_cond \
    --device cuda \
    --path model_pretrain_finetuned.h5 \
    --data_path example_conditions.csv \
    --num_samples 1 \
    --samples_per_condition 1000 \
    --loadmodel
```
