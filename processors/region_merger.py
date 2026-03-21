
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def calculate_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


COMPATIBLE_TYPES = {
    "Figure": ["FigureCaption", "paragraph"],
    "Table": ["heading", "paragraph"],
}
IOU_GATE = 0.2


def merge_regions(layout_regions: List[Dict[str, Any]], text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Link captions/headings/paragraphs to their parent Figure/Table using IoU and type-compat gates.
    - Only merge text into a layout region if:
        * parent (layout) type in COMPATIBLE_TYPES
        * child (text) type is allowed
        * IoU(layout, text) > IOU_GATE
    - When merging: append with newline and keep a 'linked_regions' list for traceability.
    """
    if not layout_regions:
        return text_regions

    merged = list(layout_regions)  # start with layout regions

    for tb in text_regions:
        tb_type = tb.get("type") or tb.get("region_type") or "paragraph"
        tb_bbox = tb.get("bbox")
        if not tb_bbox:
            continue

        best_lr = None
        best_iou = 0.0

        for lr in merged:
            lr_type = lr.get("type") or lr.get("region_type") or ""
            if lr_type not in COMPATIBLE_TYPES:
                continue
            if tb_type not in COMPATIBLE_TYPES[lr_type]:
                continue
            lr_bbox = lr.get("bbox")
            if not lr_bbox:
                continue
            iou = calculate_iou(tb_bbox, lr_bbox)
            if iou > IOU_GATE and iou > best_iou:
                best_iou = iou
                best_lr = lr

        if best_lr is not None:
            tb_text = tb.get("text", "")
            if tb_text:
                prev = best_lr.get("text", "")
                best_lr["text"] = (prev + ("\n" if prev else "") + tb_text).strip()
            best_lr.setdefault("linked_regions", []).append(tb.get("region_id", tb))
        else:
            merged.append(tb)

    return merged
