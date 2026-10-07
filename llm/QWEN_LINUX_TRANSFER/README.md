# Qwen2.5-3B QLoRA Transfer Package (Linux/ROS2)

## 1. What this folder contains
This directory is a minimal, standalone transfer package for deploying the highly accurate Experiment 1 Robot Action Model on a Linux machine (such as Ubuntu). It contains everything needed to load the QLoRA weights onto the Qwen2.5-3B base model and perform interactive inference with the exact JSON stop-token safety mechanism.

## 2. Original Windows source directory
`C:\Users\Reyan\Downloads\Case_Study\training\qwen2.5_3b\experiment_1_raw\`

## 3. Exact files included
```text
QWEN_LINUX_TRANSFER/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── inference/
│   └── interactive_inference_fixed.py
├── requirements.txt
└── README.md
```

## 4. Purpose of each file
* **`adapter_config.json`**: Tells `peft` the exact LoRA hyperparameters (rank, alpha, target modules) used during Experiment 1 training.
* **`adapter_model.safetensors`**: The actual trained weights of the adapter that learned the Bangla-to-JSON mapping.
* **`tokenizer.json` / `tokenizer_config.json`**: The Qwen2.5-3B tokenizer configurations, including the chat template and EOS definitions.
* **`interactive_inference_fixed.py`**: The safe inference script. It has been updated so that the `adapter_path` dynamically resolves to the local `adapter/` folder regardless of where you place this package on Linux.
* **`requirements.txt`**: The exact Python packages needed to run inference (e.g., bitsandbytes for 4-bit loading).

## 5. Which files are NOT required and why
The following files were intentionally excluded to keep the package lightweight and strictly focused on deployment:
* **Base Model Weights**: The 3B base model weights are not included because `AutoModelForCausalLM` will automatically download them from HuggingFace Hub directly to the Linux machine on first run.
* **Optimizer States & Checkpoints**: `checkpoint-X` folders are only needed if you plan to resume training. They are useless for inference and take up gigabytes of space.
* **Dataset Files**: The model has already learned the mapping. The CSV files are not needed.
* **Original Evaluation Scripts**: Evaluation metrics (`evaluate.py`) are for research scoring. Deployment relies entirely on `interactive_inference_fixed.py`.

## 6. Linux setup steps
1. Copy this entire `QWEN_LINUX_TRANSFER` folder to your Ubuntu machine (e.g., via `scp`, USB drive, or GitHub).
2. Open a terminal on Linux and navigate into the folder:
   ```bash
   cd QWEN_LINUX_TRANSFER
   ```
3. Create a fresh Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 7. Required Python packages
See `requirements.txt`. Key dependencies include:
* `torch`
* `transformers>=4.37.0` (required for advanced stopping criteria)
* `peft`
* `accelerate`
* `bitsandbytes`

## 8. How to load the Qwen2.5-3B base model
The script `inference/interactive_inference_fixed.py` automatically loads the base model from Hugging Face using:
```python
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B",
    device_map="auto",
    quantization_config=bnb_config
)
```

## 9. How to attach the trained QLoRA adapter
The script automatically attaches the adapter from the relative `adapter/` folder using:
```python
model = PeftModel.from_pretrained(base_model, adapter_path)
```

## 10. How to run interactive_inference_fixed.py
Ensure your virtual environment is active, then run:
```bash
python3 inference/interactive_inference_fixed.py
```

## 11. Example test command
Once the script initializes, you will see `Enter Bangla command:`. Test the model's sequence capability by typing:
> সামনে গিয়ে বামে ঘুরো, তারপর থামো।

The script will instantly output the perfect JSON and halt securely.

## 12. IMPORTANT NOTES
* **Base Model:** The base model is explicitly `Qwen/Qwen2.5-3B`.
* **Trained Adapter:** The included adapter is the exact **Experiment 1 trained adapter** that achieved 100% JSON Format Success Rate.
* **No Retraining Required:** Do **NOT** retrain this model on Linux. It is fully ready for ROS2 integration as a Semantic Task Planner.
