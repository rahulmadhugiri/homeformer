import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_24h_simulation(checkpoint_dir="/workspace/checkpoints", max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Initialize an empty or starting prompt (you can customize this)
    prompt = ""
    full_routine = ""

    for step in range(48):  # e.g., 48 steps of 30 minutes each = 24 hours
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated_text[len(prompt):].strip()

        logger.info(f"Step {step+1} continuation:\n{continuation}")
        
        full_routine += continuation + " "
        prompt = full_routine  # feed back the full generated routine for next iteration
    
    return full_routine

if __name__ == "__main__":
    routine = generate_24h_simulation()
    print("\nFull 24-hour simulated routine:\n", routine)