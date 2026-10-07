import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

def parse_llm_output(raw_text):
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
        
    text = text.strip()
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx:end_idx+1]
    
    try:
        data = json.loads(text)
        return data, True
    except json.JSONDecodeError:
        return None, False

class JsonStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that halts generation as soon as a valid JSON object is fully formed.
    This safely prevents the model from generating garbage text after the closing brace.
    """
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Quick heuristic before attempting JSON parse
        if '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                try:
                    json.loads(text[start_idx:end_idx+1])
                    return True  # Valid JSON found, STOP generation immediately
                except json.JSONDecodeError:
                    pass
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter_path = os.path.join(base_dir, "adapter")
    model_name = "Qwen/Qwen2.5-3B"
    
    print("="*50)
    print("Initializing Experiment 1 Interactive Inference (FIXED)...")
    print("="*50)
    
    # Load Tokenizer
    print(f"Loading tokenizer {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    print(f"Loading base model in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    
    # Load Adapter
    print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\nModel successfully loaded! Ready for inference.")
    print("Type 'exit' to quit.\n")
    
    system_prompt = "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan."
    
    while True:
        try:
            cmd = input("\nEnter Bangla command: ").strip()
            if cmd.lower() == 'exit':
                break
            if not cmd:
                continue
                
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cmd}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            prompt_length = inputs.input_ids.shape[1]
            
            # Create custom stopping criteria for this specific prompt
            stopping_criteria = StoppingCriteriaList([JsonStoppingCriteria(tokenizer, prompt_length)])
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria # <--- FIX: Stop dynamically when JSON is complete
                )
                
            generated_tokens = outputs[0][prompt_length:]
            prediction_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            parsed_json, is_valid = parse_llm_output(prediction_str)
            
            print("\n--- INFERENCE RESULTS ---")
            print(f"1. Input: {cmd}")
            print(f"2. Raw model output:\n{prediction_str}")
            print(f"3. Parsed JSON output:")
            if is_valid:
                json_str = json.dumps(parsed_json, ensure_ascii=False)
                print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
                
                # BRIDGE: Send valid JSON to the ROS 2 location_navigation node
                print("\n[Bridge] Sending intent to ROS 2 Nav2...")
                import subprocess
                try:
                    subprocess.run([
                        "ros2", "run", "location_navigation", "llm_to_nav", json_str
                    ])
                except Exception as e:
                    print(f"[Bridge] Failed to call ros2 command: {e}")
                    
            else:
                print("None")
            print(f"4. JSON validity status: {'Valid' if is_valid else 'Invalid'}")
            print("-" * 25)
            
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    main()
