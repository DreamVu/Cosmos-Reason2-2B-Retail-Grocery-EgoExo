"""
Qwen3-VL Temporal Frame Handling Patch

Required for video inference with Cosmos-Reason2-2B. The Qwen3-VL processor
creates per-frame token groups but video_grid_thw has a single [T,H,W] entry,
causing a StopIteration error in get_rope_index.

This patch splits each [T,H,W] entry into T rows of [1,H,W] to match
the per-frame token groups. Must be applied before inference.
"""

import torch


def patch_qwen3vl_rope(model):
    """Monkey-patch get_rope_index on Qwen3VLModel for video frame handling."""

    base_model = model.model if hasattr(model, "model") else model
    if not hasattr(base_model, "get_rope_index"):
        # Try deeper
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model

    if not hasattr(base_model, "get_rope_index"):
        print("Warning: Could not find get_rope_index to patch")
        return

    original_fn = base_model.get_rope_index

    def patched_get_rope_index(
        self, input_ids, image_grid_thw=None, video_grid_thw=None,
        second_per_grid_ts=None, attention_mask=None,
    ):
        if video_grid_thw is not None and video_grid_thw.ndim == 2:
            split_rows = []
            for row in video_grid_thw:
                t, h, w = row[0].item(), row[1].item(), row[2].item()
                for _ in range(t):
                    split_rows.append(torch.tensor([1, h, w], device=row.device, dtype=row.dtype))
            video_grid_thw = torch.stack(split_rows)

        return original_fn(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

    import types
    base_model.get_rope_index = types.MethodType(patched_get_rope_index, base_model)
    print("Patched get_rope_index for Qwen3-VL video frame handling")
