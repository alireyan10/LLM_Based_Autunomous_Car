import os
import sys
import json

def parse_llm_output(raw_text):
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
        
    text = text.strip()
    
    start_idx = text.find('{')
    if start_idx == -1:
        return None, False

    end_idx = text.rfind('}')
    if end_idx != -1 and end_idx >= start_idx:
        candidate = text[start_idx:end_idx+1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data, True
        except json.JSONDecodeError:
            pass

    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text[start_idx:])
        if isinstance(data, dict):
            return data, True
    except (json.JSONDecodeError, ValueError):
        pass

    return None, False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 qwen_inference.py '<command>'", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
        from peft import PeftModel
    except ImportError as e:
        print(f"Failed to import LLM libraries: {e}", file=sys.stderr)
        sys.exit(1)

    class JsonStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length):
            self.tokenizer = tokenizer
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores=None, **kwargs):
            generated_ids = input_ids[0][self.prompt_length:]
            if len(generated_ids) == 0:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            is_done = False
            if '}' in text:
                start_idx = text.find('{')
                if start_idx != -1:
                    sub = text[start_idx:].strip()
                    try:
                        decoder = json.JSONDecoder()
                        obj, _ = decoder.raw_decode(sub)
                        if isinstance(obj, dict):
                            is_done = True
                    except (json.JSONDecodeError, ValueError):
                        pass

            return torch.full((input_ids.shape[0],), is_done, dtype=torch.bool, device=input_ids.device)

    base_dir = os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER")
    adapter_path = os.path.join(base_dir, "adapter")
    model_name = "Qwen/Qwen2.5-3B"

    print("Loading LLM...", file=sys.stderr, flush=True)

    tokenizer_source = adapter_path if os.path.isdir(adapter_path) and os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")) else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

    # Resolve EOS tokens: include both <|endoftext|> and ChatML <|im_end|>
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.extend(tokenizer.eos_token_id)
        else:
            eos_token_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)
        
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
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    system_prompt = "You are a robot command mapping assistant. Convert the given Bangla voice command into a structured JSON action plan."
    try:
        sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
        from interactive_inference_fixed import format_prompt
        text = format_prompt(command, system_prompt=system_prompt, tokenizer=tokenizer)
    except Exception:
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{command}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    stopping_criteria = StoppingCriteriaList([JsonStoppingCriteria(tokenizer, prompt_length)])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            eos_token_id=eos_token_ids if eos_token_ids else tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        
    generated_tokens = outputs[0][prompt_length:]
    prediction_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    parsed_json, is_valid = parse_llm_output(prediction_str)

    if not is_valid or not parsed_json:
        final_json = {"intent": "UNKNOWN_TARGET", "target": ""}
    else:
        # Import plan normalizer and validator from interactive_inference_fixed
        try:
            sys.path.insert(0, os.path.expanduser("~/case_study/llm/QWEN_LINUX_TRANSFER/inference"))
            from interactive_inference_fixed import normalize_and_validate_plan
            val_ok, norm_plan, val_err = normalize_and_validate_plan(parsed_json, command=command)
            if val_ok:
                final_json = norm_plan
            else:
                print(f"Validation error: {val_err}", file=sys.stderr)
                final_json = {"intent": "UNKNOWN_TARGET", "target": ""}
        except Exception as e:
            print(f"Error validating plan: {e}", file=sys.stderr)
            final_json = {"intent": "UNKNOWN_TARGET", "target": ""}

    # ONLY JSON TO STDOUT
    print(json.dumps(final_json))

if __name__ == '__main__':
    main()
