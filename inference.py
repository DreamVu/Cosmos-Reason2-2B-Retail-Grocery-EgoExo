"""
Inference script for Cosmos-Reason2-2B-Retail-Grocery-EgoExo

Usage:
    python inference.py --video path/to/clip.mp4 --question "What is the person doing?"
    python inference.py --video path/to/clip.mp4  # uses default question
"""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from patch_qwen3vl import patch_qwen3vl_rope


def load_model(adapter_path=".", base_model="nvidia/Cosmos-Reason2-2B", device="cuda"):
    """Load base model + LoRA adapter."""
    print("Loading base model: %s" % base_model)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(base_model)

    # Apply Qwen3-VL video patch
    patch_qwen3vl_rope(model)

    # Load LoRA adapter
    print("Loading LoRA adapter: %s" % adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, processor


def run_inference(model, processor, video_path, question, max_new_tokens=512):
    """Run inference on a single video clip."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 4},
            {"type": "text", "text": question},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="PRISM Retail Video Inference")
    parser.add_argument("--video", required=True, help="Path to video clip (.mp4)")
    parser.add_argument("--question", default="Describe what is happening in this video.",
                        help="Question to ask about the video")
    parser.add_argument("--adapter", default=".", help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", default="nvidia/Cosmos-Reason2-2B",
                        help="Base model name or path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate")
    args = parser.parse_args()

    model, processor = load_model(args.adapter, args.base_model)
    response = run_inference(model, processor, args.video, args.question, args.max_tokens)

    print("\nQuestion: %s" % args.question)
    print("Answer: %s" % response)


if __name__ == "__main__":
    main()
