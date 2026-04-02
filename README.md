# Cosmos-Reason2-2B-Retail-Grocery-EgoExo

BF16 LoRA adapter for [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) fine-tuned on the [PRISM dataset](https://huggingface.co/datasets/DreamVu/Prism) for embodied video understanding in retail grocery environments.

## Quick Start

```bash
pip install -r requirements.txt
python inference.py --video path/to/clip.mp4 --question "What is the person doing?"
```

## Model Details

| | |
|---|---|
| **Base Model** | nvidia/Cosmos-Reason2-2B (Qwen2.5-VL, 2.49B params) |
| **Adapter** | LoRA rank=32, alpha=64 (49.3M params, 1.98% of base) |
| **Training Data** | PRISM — 270K video SFT samples, 20+ tasks, ego+exo+360 views |
| **Precision** | BF16 (no quantization) |
| **Hardware** | 4x NVIDIA RTX PRO 6000 Blackwell (96GB each) |
| **Training Time** | ~35 hours (7,942 steps) |
| **Adapter Size** | 67MB |

## Performance

Average **+23.8pp** improvement over zero-shot baseline across 20+ evaluated tasks.

### Links
- **Dataset:** [DreamVu/PRISM-100K](https://huggingface.co/datasets/DreamVu/Prism)
- **HuggingFace Model:** [DreamVu/Cosmos-Reason2-2B-Retail-Grocery-EgoExo](https://huggingface.co/DreamVu/Cosmos-Reason2-2B-Retail-Grocery-EgoExo)
- **Paper:** [PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied VLMs](https://arxiv.org/abs/2603.29281)

| Domain | Baseline | PRISM | Delta |
|--------|:---:|:---:|:---:|
| Embodied Reasoning (9 tasks) | 54.5% | 90.9% | +36.4 |
| Common Sense (6 tasks) | 80.9% | 91.4% | +10.5 |
| Spatial Perception (2 tasks) | 57.4% | 74.5% | +17.1 |
| Intuitive Physics (3 tasks) | 51.7% | 69.3% | +17.6 |
| **Overall** | **62.8%** | **86.6%** | **+23.8** |

## Qualitative Examples

See 17 side-by-side video comparisons between the zero-shot baseline and PRISM fine-tuned model across counting, hand interaction, goal reasoning, scene understanding, domain knowledge, and spatial reasoning tasks:

**[View Demo Gallery](https://dreamvu.github.io/Cosmos-Reason2-2B-Retail-Grocery-EgoExo/)**

## Files

```
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights (67MB)
├── inference.py                 # Inference script
├── patch_qwen3vl.py             # Required video processing patch
├── requirements.txt             # Python dependencies
└── README.md
```

## Usage (Python API)

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from patch_qwen3vl import patch_qwen3vl_rope

# Load base + adapter
model = AutoModelForVision2Seq.from_pretrained(
    "nvidia/Cosmos-Reason2-2B", torch_dtype=torch.bfloat16, device_map="cuda")
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")
patch_qwen3vl_rope(model)
model = PeftModel.from_pretrained(model, ".")

# Inference
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [
        {"type": "video", "video": "clip.mp4", "fps": 4},
        {"type": "text", "text": "What is the person doing?"}
    ]}
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## License

Derivative Model of nvidia/Cosmos-Reason2-2B under [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Commercially usable.

## Citation

```bibtex
@misc{dreamvu2026prism,
    title={PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models},
    author={DreamVu AI},
    year={2026},
    url={https://arxiv.org/abs/2603.29281}
}
```

## Contact

[sales@dreamvu.ai](mailto:sales@dreamvu.ai)
