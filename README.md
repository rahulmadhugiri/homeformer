Smart Home TinyLlama Fine-Tuning

Project Summary

Fine-tuned TinyLlama 1.1B model on a large synthetic dataset of smart home automation routines.Goal: Predict realistic sequences of smart home actions (e.g., lights, locks, thermostats) based on prior events.

Dataset & Preprocessing

Source CSV: data/Global_Realistic_Synthetic_Home_Routines__50k_.csv (320k+ rows, 50k unique routines)

Split: 40k train, 5k validation, 5k test

Tokenized Format: Saved in JSONL (processed_data/) using TinyLlama tokenizer

Training

Script: tinyllama_training.py

Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Hardware: GPU w/ ~24GB VRAM (e.g., RTX 4090)

Output: Checkpoints in checkpoints/ every 500 steps (~6GB each)

Time: ~2-3 hrs for 3 epochs

Loss: Train loss drops from ~1.5 → ~0.38, Eval loss stabilizes ~0.38

Inference

Script: test_inference.py

Function: Loads checkpoint, takes partial routine, predicts smart home actions

Supports: Iterative, multi-step predictions, 24-hour simulation

Example Input:

06:50 | bedroom_heater | turn_on [SEP] 06:52 | kitchen_coffee_maker | start_brew [SEP]

Example Output:

07:00 | bedroom_1_mirror_light | turn_off [SEP] 07:11 | none | DONE

Next Steps

Export final model + tokenizer for deployment

Wrap inference in FastAPI or RESTful interface

Optimize for edge devices

Add continuous learning from user feedback

Build UI for scheduling / feedback / simulation

Folder Structure

/workspace
├── checkpoints/          # Model checkpoints
├── data/                 # Raw dataset CSV
├── logs/                 # Training logs
├── processed_data/       # Tokenized JSONL datasets
├── tinyllama_training.py # Training script
├── test_inference.py     # Inference script
├── requirements.txt      # Python dependencies
└── README.md             # This file
