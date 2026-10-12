# 🧬 Two-Stage Molecular Generator

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

```bash
cd two_stage_model
chmod +x train_two_stage.sh
./train_two_stage.sh
```
### Option 2: Run each step manually

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

> **Note:** The example commands assume that the scripts, data files, and model checkpoints are in the locations shown. Check each script's `--help` output if an argument differs in your local version.

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

#### Standard Fine-tuning

Suitable as a starting point for datasets containing approximately 100–1,000 samples.

```bash
python main_two_stage.py \
    --mode finetune \
    --device cuda \
    --epoch 50 \
    --batch_size 256 \
    --path model_pretrain.h5 \
    --data_path conditional_train_data.csv \
    --loadmodel \
    --freeze_strategy partial
```

#### Conservative Fine-tuning

For small conditional datasets (fewer than 100 samples), the `aggressive` freezing strategy is intended to preserve more of the pre-trained parameters.

```bash
python main_two_stage.py \
    --mode finetune \
    --device cuda \
    --epoch 30 \
    --batch_size 64 \
    --path model_pretrain.h5 \
    --data_path conditional_train_data.csv \
    --loadmodel \
    --freeze_strategy aggressive
```

**Output:** `model_pretrain_finetuned.h5`. The original pre-trained checkpoint is not intended to be overwritten.

#### Parameter-Freezing Strategies

| Strategy | Approximate frozen fraction | Description | Suggested data size |
|---|---:|---|---:|
| `none` | 0% | Train all model parameters | More than 10,000 samples |
| `partial` | 40–50% | Freeze the embedding layer and lower decoder layers | 100–1,000 samples |
| `aggressive` | 80–90% | Primarily train the condition encoder | Fewer than 100 samples |

The percentages and recommendations are guidance, not guaranteed outcomes. Actual parameter counts depend on the implementation.

Example training log:

```text
Applying freezing strategy: aggressive

Parameter statistics:
  Frozen parameters: 156
  Trainable parameters: 28
  Frozen fraction: 84.8%

Optimizer: learning rate = 1e-5, trainable parameter tensors = 28
```

Important notes:

- Fine-tuning may save a new checkpoint with `_finetuned` appended to the original filename.
- The original pre-trained checkpoint can be retained for unconditional generation or further fine-tuning.
- A lower learning rate, such as `1e-5`, can help limit disruption to pre-trained weights.
- With very small datasets, stronger freezing may reduce overfitting, although it can also limit responsiveness to the conditioning features.

### 3. Unconditional Generation

```bash
python main_two_stage.py \
    --mode infer_uncond \
    --device cuda \
    --path model_pretrain.h5 \
    --num_samples 10
```

This runs generation using the pre-trained model without an RMSF condition.

Depending on the implementation, the script may also load the conditional data file to keep the vocabulary consistent. If a warning appears because the file is missing, check the script's behavior before proceeding.

Example output:

```text
Generated molecules (unconditional):
1. CCOc1ccccc1
2. CC(=O)NCc1ccc(O)cc1
3. CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

### 4. Conditional Generation

#### Generate one molecule per condition

```bash
python main_two_stage.py \
    --mode infer_cond \
    --device cuda \
    --path model_pretrain_finetuned.h5 \
    --data_path example_conditions.csv \
    --num_samples 5
```

- `--path` specifies the fine-tuned checkpoint.
- `--data_path` points to the file containing reference SMILES and RMSF features.
- `--num_samples 5` requests the first five conditions, with one generated molecule per condition under the documented interface.
- Output may be saved to a timestamped CSV file such as `generated_cond_YYYYMMDD_HHMMSS.csv`.

Illustrative output:

```text
Condition 1:
  Reference SMILES: O=S(C1=CC=CC2=CC=CC(NC3=CC=CC=C3)=C21)(O)=O
  First five RMSF values: [2.080, 1.347, 1.034, 1.259, 1.613] ...
  Generated SMILES: CCOc1ccc(O)cc1
```

#### Generate multiple molecules per condition

```bash
python main_two_stage.py \
    --mode infer_cond \
    --device cuda \
    --path model_pretrain_finetuned.h5 \
    --data_path example_conditions.csv \
    --num_samples 3 \
    --samples_per_condition 5
```

This example requests three conditions and five molecules per condition, for up to 15 generated samples.

The generated CSV may use the following columns:

```csv
id,condition_id,sample_id,reference_smiles,rmsf_values,generated_smiles
1,1,1,REFERENCE_SMILES,"2.080,1.347,...",CCOc1ccc(O)cc1
2,1,2,REFERENCE_SMILES,"2.080,1.347,...",CC(=O)Oc1ccccc1
```

| Column | Description |
|---|---|
| `id` | Global sample identifier |
| `condition_id` | Index of the input condition |
| `sample_id` | Generated sample index within a condition |
| `reference_smiles` | Reference molecule SMILES |
| `rmsf_values` | Comma-separated RMSF condition values |
| `generated_smiles` | SMILES string generated by the model |

### Example: One Condition, Ten Molecules

```bash
python main_two_stage.py \
    --mode infer_cond \
    --path model_epoch_50.h5 \
    --data_path example_conditions.csv \
    --num_samples 1 \
    --samples_per_condition 10 \
    --loadmodel
```

### Example: Five Conditions, Twenty Molecules Each

```bash
python main_two_stage.py \
    --mode infer_cond \
    --path model_epoch_50.h5 \
    --data_path example_conditions.csv \
    --num_samples 5 \
    --samples_per_condition 20 \
    --loadmodel
```

The second command requests up to 100 molecules in total (5 conditions × 20 samples per condition). Confirm the argument semantics in your local script, as they can differ between code versions.

## Data Format

### RMSF Input Files

The documented input supports two delimiter styles. Each row contains a SMILES string and exactly 30 RMSF values separated by a tab.

**Format 1: comma-separated RMSF values**

```text
SMILES<TAB>rmsf1,rmsf2,...,rmsf30
```

**Format 2: space-separated RMSF values**

```text
SMILES<TAB>rmsf1 rmsf2 ... rmsf30
```

Example `rmsf_data.txt`:

```text
CCO	1.23,1.56,1.89,2.12,2.45,1.78,2.34,2.67,3.01,3.45,2.89,3.23,2.56,3.12,2.78,3.34,2.67,3.01,2.45,2.89,3.23,2.56,3.12,2.78,3.45,2.89,3.34,3.12,3.56,3.89
c1ccccc1	1.45,1.78,2.12,2.56,2.89,2.23,2.67,3.12,3.45,3.78,3.12,3.56,2.89,3.23,3.67,3.01,3.45,2.89,3.23,3.56,2.78,3.12,3.45,2.67,3.01,3.34,3.78,3.23,3.56,3.89
```

**Requirements:**

- Each row must contain exactly 30 RMSF values.
- Separate the SMILES string and the RMSF field with a tab.
- RMSF values are typically expressed in ångströms (Å). The example range of 0.5–5.0 Å is illustrative, not a strict validation rule.
- Use experimentally or computationally derived RMSF values for scientific evaluation; synthetic values are appropriate only for demonstrations or software tests.
