# Homeformer – Smart Home Action Modeling with TinyLlama

**Fine-tuned TinyLlama-1.1B-Chat on a large, realistic synthetic dataset of smart home routines.**

 **Hugging Face Model:** [rahulmadhugiri/homeformer](https://huggingface.co/rahulmadhugiri/homeformer)

---

## Project Overview

The goal of this project is to train a compact language model that can predict realistic sequences of smart home actions — e.g., when to turn on lights, brew coffee, or lock doors — based on time, context, and prior events.

We fine-tuned **TinyLlama-1.1B** to generate complete home automation routines in a causal language modeling setup.

---

##  Dataset & Preprocessing

### Data Source
- **File:** `data/Global_Realistic_Synthetic_Home_Routines__50k_.csv`
- **Size:** ~320,000 rows across ~50k unique routines

### Data Splits
- **Train:** 40k routines
- **Validation:** 5k routines  
- **Test:** 5k routines

### Preprocessing Pipeline
- Tokenized using TinyLlama tokenizer
- Stored in `processed_data/` as JSONL format

---

##  Training Configuration

| Parameter | Value |
|-----------|-------|
| **Script** | `tinyllama_training.py` |
| **Base Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Hardware** | GPU with ~24GB VRAM (e.g., RTX 4090) |
| **Training Time** | ~2–3 hours for 3 epochs |
| **Checkpoints** | Saved every 500 steps to `checkpoints/` |

###  Training Results

| Metric | Value |
|--------|-------|
| **Initial Train Loss** | ~1.5 |
| **Final Train Loss** | ~0.40 |
| **Eval Loss** | ~0.37 |

---

##  Inference & Usage

### Getting Started
Use the `test_inference.py` script for model inference.

### Features
-  **Iterative forecasting** - Predict next actions step by step
-  **Multi-step output** - Generate sequences of actions
-  **Full-day simulations** - Complete daily routine modeling

### Example Usage

#### Input Format
```
06:50 | bedroom_heater | turn_on [SEP] 06:52 | kitchen_coffee_maker | start_brew [SEP]
```

#### Output Format
```
07:00 | bedroom_1_mirror_light | turn_off [SEP] 07:11 | none | DONE
```

---

## 🗂️ Project Structure

```
homeformer/
├── checkpoints/          # Model checkpoints (6GB each)
├── data/                 # Raw dataset CSV
├── logs/                 # Training logs
├── processed_data/       # Tokenized input as JSONL
├── tinyllama_training.py # Training script
├── test_inference.py     # Inference script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run inference:**
   ```bash
   python test_inference.py
   ```

3. **Train model:**
   ```bash
   python tinyllama_training.py
   ```

