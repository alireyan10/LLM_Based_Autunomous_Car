#!/usr/bin/env python3
"""
Bangla Voice-to-Action Interactive Inference for car_bot.

Integrates:
1. Live Microphone Speech-to-Text (Google API bn-BD)
2. Domain-Aware Phonetic & Fuzzy Noise Auto-Correction
3. In-Memory Qwen2.5-3B LLM Inference
4. Strict Schema Normalization and Validation
5. Automatic Dispatch to ROS 2 Nav2 (location_navigation llm_to_nav)
"""

import os
import sys
import json
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import existing LLM inference components
from interactive_inference_fixed import (
    DEFAULT_SYSTEM_PROMPT,
    configure_chat_template,
    format_prompt,
    parse_llm_output,
    normalize_and_validate_plan,
    JsonStoppingCriteria,
    HAS_LLM_DEPS,
)

if HAS_LLM_DEPS:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList
    from peft import PeftModel
else:
    torch = None

# Import voice processor from location_navigation
SRC_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "src", "location_navigation"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from location_navigation.voice_processor import (
        clean_and_correct_stt_text,
        convert_to_wav,
        speech_to_bangla_text,
        listen_from_microphone,
    )
except ImportError:
    # Fallback to local import if needed
    clean_and_correct_stt_text = None
    listen_from_microphone = None
    speech_to_bangla_text = None
    convert_to_wav = None


def run_pipeline(cmd_text, model, tokenizer, eos_token_ids, system_prompt):
    """Generates JSON plan with Qwen and dispatches to ROS 2 Nav2."""
    if not cmd_text:
        return

    print(f"\n[Processing Command]: \"{cmd_text}\"")
    text = format_prompt(cmd_text, system_prompt=system_prompt, tokenizer=tokenizer)
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
            stopping_criteria=stopping_criteria,
        )

    generated_tokens = outputs[0][prompt_length:]
    prediction_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    raw_parsed_json, json_decoded = parse_llm_output(prediction_str)

    print("\n--- INFERENCE RESULTS ---")
    print(f"1. Spoken Input: {cmd_text}")
    print(f"2. Raw Model Output:\n{prediction_str}")

    if not json_decoded:
        print("Error: Could not decode JSON from model output.")
        print("-" * 25)
        return

    # Strict schema validation and normalization
    is_valid, normalized_plan, val_err = normalize_and_validate_plan(raw_parsed_json, command=cmd_text)

    if is_valid:
        json_str = json.dumps(normalized_plan, ensure_ascii=False)
        print("3. Validated JSON Plan:")
        print(json.dumps(normalized_plan, indent=4, ensure_ascii=False))

        # BRIDGE: Send valid JSON to the ROS 2 location_navigation node
        print("\n[Bridge] Sending intent to ROS 2 Nav2...")
        nav_cmd = ["ros2", "run", "location_navigation", "llm_to_nav", json_str]

        if os.environ.get("USE_SIM_TIME", "").strip().lower() in ("1", "true", "yes"):
            nav_cmd.extend(["--ros-args", "-p", "use_sim_time:=true"])

        try:
            subprocess.run(nav_cmd, check=True)
        except Exception as e:
            print(f"[Bridge] Error calling ros2 command: {e}")
    else:
        print(f"Plan Rejected: {val_err}")
    print("-" * 25)


def main():
    parser = argparse.ArgumentParser(description="Bangla Voice Navigation with In-Memory Qwen LLM")
    parser.add_argument("--audio", type=str, default=None, help="Input audio file path (.wav, .mp3, .mp4)")
    parser.add_argument("--continuous", action="store_true", help="Continuously listen to microphone without pressing Enter")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter_path = os.path.join(base_dir, "adapter")
    model_name = "Qwen/Qwen2.5-3B"

    print("=" * 65)
    print("Initializing Qwen2.5-3B Bangla Voice Navigation...")
    print("=" * 65)

    if not HAS_LLM_DEPS:
        print("Error: PyTorch / Transformers / PEFT libraries not installed.", file=sys.stderr)
        sys.exit(1)

    # Load Tokenizer
    tokenizer_source = adapter_path if os.path.isdir(adapter_path) and os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")) else model_name
    print(f"Loading tokenizer from {tokenizer_source}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_chat_template(tokenizer)

    # Resolve EOS tokens
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.extend(tokenizer.eos_token_id)
        else:
            eos_token_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    # Load Model
    print("Loading base model in 4-bit...", flush=True)
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

    print("\nModel successfully loaded! Ready for Voice Navigation.")
    system_prompt = DEFAULT_SYSTEM_PROMPT

    # Audio file mode
    if args.audio:
        print(f"Processing audio file: {args.audio}")
        wav_path = convert_to_wav(args.audio)
        cmd_text = speech_to_bangla_text(wav_path)
        if cmd_text:
            run_pipeline(cmd_text, model, tokenizer, eos_token_ids, system_prompt)
        else:
            print("No speech detected in audio file.")
        return

    # Microphone interactive mode
    print("Commands: Speak into microphone or type command. Type 'exit' to quit.\n")
    while True:
        try:
            if not args.continuous:
                user_input = input("\n[Press Enter to Speak, or type command, 'exit' to quit]: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if user_input:
                    # User typed command directly
                    run_pipeline(user_input, model, tokenizer, eos_token_ids, system_prompt)
                    continue

            # Listen from mic
            cmd_text = listen_from_microphone(timeout=8.0, phrase_time_limit=12.0)
            if cmd_text:
                run_pipeline(cmd_text, model, tokenizer, eos_token_ids, system_prompt)
            else:
                print("No clear speech detected. Please try again.")

            if args.continuous:
                import time
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nExiting voice navigation.")
            break


if __name__ == "__main__":
    main()
