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

Below are comparisons between Cosmos-Reason2-2B (zero-shot) and PRISM (fine-tuned) on real retail video clips.

### Object Counting & Tracking

<video src="examples/counting_basket.mp4" controls muted loop width="400"></video>

> **Q:** How many products did the person put into the basket?
>
> **Cosmos-Reason2-2B:** Two products.
>
> **PRISM:** The person put one product, a small red container, into the basket.

<video src="examples/counting_before_after.mp4" controls muted loop width="400"></video>

> **Q:** Count the products in the basket at the beginning and at the end.
>
> **Cosmos-Reason2-2B:** There are 3 products in the basket at the beginning and 4 products at the end.
>
> **PRISM:** The basket contains 2 items at the start, and 3 items at the end.

<video src="examples/counting_last_item.mp4" controls muted loop width="400"></video>

> **Q:** Count the number of products in the basket by the end. What was the last item?
>
> **Cosmos-Reason2-2B:** There is one product in the basket by the end, which is a blue bottle.
>
> **PRISM:** The shopping basket contains 2 items. The last item added was the Hershey's Cocoa container.

### Hand Interaction Recognition

<video src="examples/hand_interaction.mp4" controls muted loop width="400"></video>

> **Q:** What actions are the hands performing?
>
> **Cosmos-Reason2-2B:** The hands are holding a blue basket and placing a jar into it.
>
> **PRISM:** The left hand is holding the basket, and the right hand is grasping an item.

### Goal-Conditioned Reasoning (Chain-of-Thought)

<video src="examples/goal_reasoning.mp4" controls muted loop width="400"></video>

> **Q:** Is the person still evaluating the product or ready to place it in the basket?
>
> **Cosmos-Reason2-2B:** The person is still evaluating the product. Next, he should place the guava in the basket.
>
> **PRISM:** *\<think\>The person is currently holding and inspecting the guava. The current sub-goal is "Evaluate Product Quality," part of the larger goal "Select Produce Item."\</think\>* The person is still evaluating the guava's quality. The next action would be to place it in the basket if it meets their standards, or return it to the display.

### Scene Understanding & Localization

<video src="examples/action_recognition.mp4" controls muted loop width="400"></video>

> **Q:** What is he doing in the scene?
>
> **Cosmos-Reason2-2B:** He is reaching into a fruit display case.
>
> **PRISM:** The person is inspecting a blackberry. They hold it in their hand and examine it closely.

<video src="examples/localization.mp4" controls muted loop width="400"></video>

> **Q:** Where is the person located?
>
> **Cosmos-Reason2-2B:** The person is located in the aisle of a store.
>
> **PRISM:** The person is in the Aisle (Personal Care/Flour).

<video src="examples/activity_reasoning.mp4" controls muted loop width="400"></video>

> **Q:** What activity is this person performing and why?
>
> **Cosmos-Reason2-2B:** The person is carefully reading the label on the shampoo bottle.
>
> **PRISM:** The person is holding the Vatika shampoo bottle and examining the back label, likely checking ingredients, usage instructions, or expiration date.

### Factual Accuracy & Domain Knowledge

<video src="examples/product_counting.mp4" controls muted loop width="400"></video>

> **Q:** How many products did the person check? Respond with a number only.
>
> **Cosmos-Reason2-2B:** 1
>
> **PRISM:** 2

<video src="examples/domain_knowledge.mp4" controls muted loop width="400"></video>

> **Q:** Would you expect to find beef in this aisle?
>
> **Cosmos-Reason2-2B:** Yes.
>
> **PRISM:** No, the visible items are tea boxes, indicating this is a tea aisle, not a grocery aisle for beef.

### Spatial Reasoning & Affordances

<video src="examples/affordance_fridge.mp4" controls muted loop width="400"></video>

> **Q:** Can we open the refrigerator from where we are standing?
>
> **Cosmos-Reason2-2B:** Yes, it is possible to open it from where you are standing.
>
> **PRISM:** *\<think\>The person's current position is at a distance from the refrigerator, indicating they cannot interact directly.\</think\>* No, the refrigerator is too far away to be opened from this position.

<video src="examples/navigation_fridge.mp4" controls muted loop width="400"></video>

> **Q:** How can we get to the refrigerator from where we are standing?
>
> **Cosmos-Reason2-2B:** Walk straight ahead to the refrigerator on the left side of the aisle.
>
> **PRISM:** *\<think\>The refrigerator units are positioned along the left side of the aisle, adjacent to the bulk item display.\</think\>* Walk forward along the aisle, past the bulk item display, to reach the refrigerator on the left side.

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
