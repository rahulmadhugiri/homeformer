#!/usr/bin/env python3
"""
Complete pipeline for fine-tuning TinyLlama on smart home routines dataset
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import Dataset as HFDataset, load_from_disk
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartHomeDataProcessor:
    """Processes raw CSV into tokenized sequences for training"""
    
    def __init__(
        self,
        csv_path: str = "/workspace/data/Global_Realistic_Synthetic_Home_Routines__50k_.csv",
        output_dir: str = "/workspace/processed_data",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ):
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
    def read_and_group_routines(self) -> Dict[int, List[Dict]]:
        """Read CSV and group by routine_id"""
        logger.info(f"Reading CSV file: {self.csv_path}")
        
        # Read CSV in chunks for memory efficiency
        df = pd.read_csv(
            self.csv_path,
            dtype={'routine_id': str, 'timestamp': str, 'device': str, 'action': str}
        )
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Extract numeric routine ID and group
        routines = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping routines"):
            # Extract number from routine_id (e.g., "routine_000001" -> 1)
            routine_id = int(row['routine_id'].split('_')[-1])
            
            if routine_id not in routines:
                routines[routine_id] = []
                
            routines[routine_id].append({
                "timestamp": row['timestamp'],
                "device": row['device'],
                "action": row['action']
            })
        
        logger.info(f"Found {len(routines)} unique routines")
        return routines
    
    def format_routine_as_string(self, routine: List[Dict]) -> str:
        """Convert routine sequence to formatted string"""
        parts = []
        for action in routine:
            part = f"{action['timestamp']} | {action['device']} | {action['action']}"
            parts.append(part)
        
        return " [SEP] ".join(parts)
    
    def create_train_val_test_splits(self, routines: Dict[int, List[Dict]]):
        """Create 80/10/10 splits and save as JSONL"""
        routine_ids = list(routines.keys())
        np.random.shuffle(routine_ids)
        
        n_routines = len(routine_ids)
        train_idx = int(n_routines * 0.8)
        val_idx = int(n_routines * 0.9)
        
        splits = {
            'train': routine_ids[:train_idx],
            'val': routine_ids[train_idx:val_idx],
            'test': routine_ids[val_idx:]
        }
        
        for split_name, ids in splits.items():
            output_file = self.output_dir / f"{split_name}.jsonl"
            logger.info(f"Creating {split_name} split with {len(ids)} routines")
            
            with open(output_file, 'w') as f:
                for routine_id in ids:
                    sequence_str = self.format_routine_as_string(routines[routine_id])
                    data = {
                        "routine_id": routine_id,
                        "text": sequence_str
                    }
                    f.write(json.dumps(data) + '\n')
    
    def process_dataset(self):
        """Main processing pipeline"""
        logger.info("Starting dataset processing...")
        
        # Read and group routines
        routines = self.read_and_group_routines()
        
        # Create splits
        self.create_train_val_test_splits(routines)
        
        logger.info("Dataset processing complete!")

class SmartHomeTokenizer:
    """Tokenizes processed sequences for TinyLlama training"""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_dir: str = "/workspace/processed_data",
        output_dir: str = "/workspace/tokenized_data",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_file(self, input_file: str, split_name: str) -> HFDataset:
        """Tokenize a JSONL file"""
        logger.info(f"Tokenizing {input_file}")
        
        # Read JSONL
        data = []
        with open(input_file) as f:
            for line in f:
                data.append(json.loads(line))
        
        # Tokenize each sequence
        tokenized_data = []
        for item in tqdm(data, desc=f"Tokenizing {split_name}"):
            text = item['text']
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create labels for causal LM (shifted input_ids)
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            
            # Labels are input_ids shifted by 1 (next token prediction)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Mask padding tokens
            
            # Only include tensor fields needed for training
            tokenized_data.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "labels": labels.tolist()
            })
        
        # Convert to HuggingFace Dataset
        dataset = HFDataset.from_list(tokenized_data)
        return dataset
    
    def tokenize_all_splits(self):
        """Tokenize train/val/test splits"""
        for split in ['train', 'val', 'test']:
            input_file = self.data_dir / f"{split}.jsonl"
            if input_file.exists():
                dataset = self.tokenize_file(input_file, split)
                
                # Save tokenized dataset
                output_path = self.output_dir / split
                dataset.save_to_disk(str(output_path))
                logger.info(f"Saved {split} dataset: {len(dataset)} examples")
            else:
                logger.warning(f"File not found: {input_file}")

class TinyLlamaTrainer:
    """Fine-tune TinyLlama on smart home data"""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_dir: str = "/workspace/tokenized_data",
        output_dir: str = "/workspace/checkpoints",
        logs_dir: str = "/workspace/logs"
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_seed(42)
        
        # Detect optimal batch size for RTX 4090 (24GB VRAM)
        self.batch_size = self._get_optimal_batch_size()
        
    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available VRAM"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)
            logger.info(f"Detected {memory_gb:.1f}GB GPU memory")
            
            # Conservative batch size for 24GB RTX 4090
            if memory_gb >= 20:
                return 4
            elif memory_gb >= 12:
                return 2
            else:
                return 1
        else:
            logger.warning("No CUDA device detected, using CPU")
            return 1
    
    def load_datasets(self):
        """Load tokenized datasets"""
        logger.info("Loading datasets...")
        
        train_dataset = load_from_disk(str(self.data_dir / "train"))
        val_dataset = load_from_disk(str(self.data_dir / "val"))
        
        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Val size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Use bfloat16
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize embeddings if tokenizer was modified
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            logging_dir=str(self.logs_dir),
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # FP16/Mixed precision settings
            bf16=True if torch.cuda.is_available() else False,  # Use bfloat16 instead of fp16
            fp16=False,  # Disable fp16
            dataloader_pin_memory=False,
            report_to="tensorboard"
        )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        train_dataset, val_dataset = self.load_datasets()
        
        # Setup model
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Training arguments
        training_args = self.create_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Beginning training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        trainer.save_state()
        
        # Log final metrics
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        
        # Final evaluation
        eval_result = trainer.evaluate()
        logger.info(f"Final eval loss: {eval_result['eval_loss']:.4f}")
        
        return trainer

class SmartHomeInference:
    """Test inference with trained model"""
    
    def __init__(self, checkpoint_dir: str = "/workspace/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        
    def load_model(self):
        """Load trained model and tokenizer"""
        logger.info(f"Loading model from {self.checkpoint_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return model, tokenizer
    
    def predict_next_action(self, partial_routine: str, max_new_tokens: int = 50):
        """Predict next action(s) given partial routine"""
        model, tokenizer = self.load_model()
        
        # Tokenize input
        inputs = tokenizer(partial_routine, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part
        new_part = generated_text[len(partial_routine):].strip()
        
        return new_part

def main():
    """Main pipeline"""
    logger.info("=== Smart Home TinyLlama Fine-tuning Pipeline ===")
    
    # Step 1: Process raw data
    logger.info("Step 1: Processing dataset...")
    processor = SmartHomeDataProcessor()
    processor.process_dataset()
    
    # Step 2: Tokenize data
    logger.info("Step 2: Tokenizing data...")
    tokenizer = SmartHomeTokenizer()
    tokenizer.tokenize_all_splits()
    
    # Step 3: Train model
    logger.info("Step 3: Training model...")
    trainer_obj = TinyLlamaTrainer()
    trainer = trainer_obj.train()
    
    # Step 4: Test inference
    logger.info("Step 4: Testing inference...")
    inference = SmartHomeInference()
    
    # Test with sample input
    test_input = "07:00 | bedroom_light | turn_on [SEP] 07:05 | kitchen_light | turn_on [SEP]"
    predicted = inference.predict_next_action(test_input)
    
    logger.info(f"Input: {test_input}")
    logger.info(f"Predicted next: {predicted}")
    
    logger.info("=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()