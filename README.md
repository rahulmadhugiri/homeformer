🏠 Homeformer – Smart Home Action Modeling with TinyLlama

Fine-tuned TinyLlama-1.1B-Chat on a large, realistic synthetic dataset of smart home routines.

Hugging Face model:🔗 https://huggingface.co/rahulmadhugiri/homeformer

🧠 Project Summary

The goal of this project is to train a compact language model that can predict realistic sequences of smart home actions — e.g., when to turn on lights, brew coffee, or lock doors — based on time, context, and prior events.

We fine-tuned TinyLlama-1.1B to generate complete home automation routines in a causal language modeling setup.

📊 Dataset & Preprocessing

Source file: data/Global_Realistic_Synthetic_Home_Routines__50k_.csv

Size: ~320,000 rows across ~50k unique routines

Splits:

40k train

5k validation

5k test

Preprocessing:

Tokenized using TinyLlama tokenizer

Stored in processed_data/ as JSONL

🏋️ Training

Script: tinyllama_training.py

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Hardware: GPU with ~24GB VRAM (e.g., RTX 4090)

Training time: ~2–3 hours for 3 epochs

📉 Loss

Metric

Value

Initial train

~1.5

Final train

~0.40

Eval loss

~0.37

Checkpoints: saved every 500 steps to checkpoints/

🤖 Inference

Script: test_inference.py

Function: Given partial routine history, predicts next actions

Supports:

Iterative forecasting

Multi-step output

Full-day simulations

🔢 Example Input

06:50 | bedroom_heater | turn_on [SEP] 06:52 | kitchen_coffee_maker | start_brew [SEP]

🪄 Example Output

07:00 | bedroom_1_mirror_light | turn_off [SEP] 07:11 | none | DONE

🛠️ Next Steps

✅ Export final model + tokenizer → Hugging Face

🔜 Wrap inference in FastAPI or REST API

🔜 Optimize for edge devices

🔜 Build feedback/simulation UI

🔜 Add online learning from user behavior

🗂️ Folder Structure

homeformer/
├── checkpoints/          # Model checkpoints (6GB each)
├── data/                 # Raw dataset CSV
├── logs/                 # Training logs
├── processed_data/       # Tokenized input as JSONL
├── tinyllama_training.py # Training script
├── test_inference.py     # Inference script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

