
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


def save_image(img, out_path: Path, msg: str = "") -> None:
    if img is None or img.size == 0:
        logger.warning("save_image: empty image for %s", out_path)
        return
    if cv2 is None:
        logger.warning("OpenCV not available; cannot save %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    if msg:
        logger.debug(msg)


class PageProcessor:
    """
    Minimal wiring to save region snapshots (Figure/Table).
    Call `attach_region_snapshots(page_image, regions)` after layout detection.
    """

    def __init__(self, output_paths: Dict[str, Path]):
        self.output_paths = output_paths

    def attach_region_snapshots(self, page_image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Use separate folders for figures and tables
        figures_dir = self.output_paths.get("figures", Path("Output/extracted_figures"))
        tables_dir = self.output_paths.get("tables", Path("Output/extracted_tables"))
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        figure_count = 1
        for r in regions:
            r_type = str(r.get("type", "")).lower()
            if r_type in ["figure", "table"]:
                bbox = r.get("bbox")
                if not bbox or page_image is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                crop = page_image[y1:y2, x1:x2]
                
                # Save to appropriate folder
                if r_type == "figure":
                    fname = f"figure_{figure_count:03d}.png"
                    fpath = figures_dir / fname
                    figure_count += 1
                else:  # table
                    # Tables already saved by table extractor, just reference
                    fname = f"region_{r.get('region_id', 'unknown')}.png"
                    fpath = tables_dir / fname
                
                save_image(crop, fpath, f"{r_type} snapshot: {fname}")
                r["snapshot_image"] = str(fpath)
        return regions

    def verify_layout(self, page_image: np.ndarray, detected_regions: List[Dict]) -> List[Dict]:
        """
        Use VLM Agent to verify and filter detected regions.
        """
        # --- NEW: VLM AGENTIC VERIFICATION ---
        from models.vlm_client import VLMClient
        
        try:
            vlm = VLMClient()
            if vlm.check_availability():
                logger.info("🤖 VLM Agent: Verifying layout regions...")
                
                # OPTIMIZATION: Resize image to max 512px for faster VLM inference
                h, w = page_image.shape[:2]
                max_dim = 512
                scale = 1.0
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    page_image_resized = cv2.resize(page_image, (new_w, new_h))
                else:
                    page_image_resized = page_image
                
                # Scale regions for VLM prompt
                vlm_regions = []
                for r in detected_regions:
                    # Create a copy with scaled bbox
                    scaled_r = r.copy()
                    bbox = r['bbox']
                    scaled_r['bbox'] = [b * scale for b in bbox]
                    vlm_regions.append(scaled_r)

                # Convert cv2 image (BGR) to RGB bytes for VLM
                img_rgb = cv2.cvtColor(page_image_resized, cv2.COLOR_BGR2RGB)
                is_success, buffer = cv2.imencode(".jpg", img_rgb)
                
                if is_success:
                    # Pass SCALED regions to VLM
                    keep_ids = vlm.verify_layout_smart(buffer.tobytes(), vlm_regions)
                    
                    # Filter regions
                    original_count = len(detected_regions)
                    filtered_regions = [r for r in detected_regions if r['region_id'] in keep_ids]
                    new_count = len(filtered_regions)
                    
                    if new_count < original_count:
                        logger.info(f"🤖 VLM Agent: Removed {original_count - new_count} duplicate/noise regions.")
                        return filtered_regions
                    else:
                        logger.info("🤖 VLM Agent: No changes made to layout.")
                        return detected_regions
                        
        except Exception as e:
            logger.warning(f"VLM Agent failed (skipping verification): {e}")
            
        return detected_regions
