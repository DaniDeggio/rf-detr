# ------------------------------------------------------------------------
# RF-DETR
# Helper to wrap a model so ONNX export returns post-processed top-K outputs.
# ------------------------------------------------------------------------
from typing import Tuple

import torch
import torch.nn as nn

from rfdetr.models.postprocess import PostProcess


class PostprocessedExportModule(nn.Module):
    """Wraps an RF-DETR nn.Module (with forward_export) and a PostProcess
    instance to return fixed tensors (boxes, labels, scores, masks) suitable
    for ONNX export.

    Forward input: tensors (B,3,H,W)
    Forward output: tuple of tensors:
      - boxes: float32 [B, K, 4] (xyxy absolute coordinates)
      - labels: int64 [B, K]
      - scores: float32 [B, K]
      - masks: float32 [B, K, Hm, Wm] (optional; 0/1 float)
    """

    def __init__(self, model: nn.Module, postprocess: PostProcess, output_mask: bool = True, shape: Tuple[int, int] = (432, 432)):
        super().__init__()
        self.model = model
        self.postprocess = postprocess
        self.output_mask = output_mask
        self.shape = shape

    def export(self):
        """Prepare wrapped model for ONNX export by forwarding export() to submodules.

        This ensures submodules that implement `export()` (e.g., LWDETR, Joiner,
        Backbone, SegmentationHead) switch to their `forward_export` paths.
        """
        # forward export to underlying model if supported
        if hasattr(self.model, "export"):
            try:
                self.model.export()
            except Exception:
                # swallow exceptions to not break the exporter; tracing may still work
                pass

    def forward(self, tensors: torch.Tensor):
        # Call model.forward_export which returns (boxes, logits) or (boxes, logits, masks)
        out = self.model.forward_export(tensors)
        if isinstance(out, tuple) and len(out) == 3:
            boxes, logits, masks = out
            outputs = {"pred_boxes": boxes, "pred_logits": logits, "pred_masks": masks}
        else:
            boxes, logits = out
            outputs = {"pred_boxes": boxes, "pred_logits": logits}

        B = tensors.shape[0]
        h, w = self.shape
        # PostProcess expects target_sizes as [B,2] tensor (h,w)
        target_sizes = torch.tensor([[h, w]] * B, dtype=torch.float32, device=tensors.device)

        results = self.postprocess(outputs, target_sizes)

        # results is a list of dicts with keys: 'boxes','labels','scores' and optionally 'masks'
        # Convert to stacked tensors with fixed K = self.postprocess.num_select
        K = self.postprocess.num_select

        # Prepare tensors
        boxes_t = torch.zeros((B, K, 4), dtype=results[0]["boxes"].dtype, device=tensors.device)
        labels_t = torch.zeros((B, K), dtype=torch.long, device=tensors.device)
        scores_t = torch.zeros((B, K), dtype=results[0]["scores"].dtype, device=tensors.device)

        if self.output_mask and ("masks" in results[0]):
            # results[0]["masks"] may have shape [K, 1, H, W] (bool) or [K, H, W]
            mask_example = results[0]["masks"]
            if mask_example.dim() == 4:
                _, _, hm, wm = mask_example.shape
            else:
                hm, wm = mask_example.shape[-2:]
            masks_t = torch.zeros((B, K, hm, wm), dtype=torch.float32, device=tensors.device)
        else:
            masks_t = None

        for i, r in enumerate(results):
            n = r["boxes"].shape[0]
            take = min(n, K)
            boxes_t[i, :take] = r["boxes"][:take]
            labels_val = r["labels"][:take]
            # ensure labels are long
            labels_t[i, :take] = labels_val.to(torch.long)
            scores_t[i, :take] = r["scores"][:take]
            if masks_t is not None and "masks" in r:
                m = r["masks"][:take]
                # squeeze possible channel dimension [K,1,H,W] -> [K,H,W]
                if m.dim() == 4 and m.shape[1] == 1:
                    m = m.squeeze(1)
                masks_t[i, :take] = m.to(torch.float32)

        if masks_t is not None:
            return boxes_t, labels_t, scores_t, masks_t
        return boxes_t, labels_t, scores_t
