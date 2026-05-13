"""
Page Processor Module
Handles processing of a single PDF page, orchestrating layout detection, 
OCR, and region processing.
"""

import logging
import fitz
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from config import OUTPUT_CONFIG
from processors.region_merger import merge_regions
from processors.page_processor import PageProcessor as SnapshotProcessor
from utils.pipeline_utils import save_image, assess_text_quality

from pipeline.coordinate_converter import (
    convert_page_to_image,
    convert_regions_to_pdf_coords,
    convert_regions_to_image_coords
)
from pipeline.region_cleaner import cleanup_regions
from pipeline.visualization_manager import create_debug_visualizations, create_bounding_box_visualization_wrapper
from common.types import LayoutRegion, BBox

logger = logging.getLogger(__name__)

class PageProcessor:
    """
    Handles processing logic for a single page.
    Encapsulates the 'process_page' logic from the original pipeline.
    """
    
    def __init__(self, components: Dict[str, Any], output_paths: Dict[str, Any], debug_mode: bool = False):
        """
        Initialize PageProcessor with shared components and output paths.
        
        Args:
            components: Dictionary of initialized pipeline components (models, processors)
            output_paths: Dictionary of output paths
            debug_mode: Whether to enable debug mode
        """
        self.components = components
        self.output_paths = output_paths
        self.debug_mode = debug_mode
        
        # Unpack directories for convenience
        self.thumbnails_dir = output_paths['thumbnails']
        self.tables_dir = output_paths['tables']
        self.figures_dir = output_paths['figures']
        self.debug_dir = output_paths['debug']
        
        # Unpack components for convenience
        self.document_analyzer = components.get('document_analyzer')
        self.layout_detector = components.get('layout_detector')
        self.ocr_engine = components.get('ocr_engine')
        self.reading_order_resolver = components.get('reading_order_resolver')
        self.semantic_grouper = components.get('semantic_grouper')
        self.figure_caption_processor = components.get('figure_caption_processor')
        self.region_processor = components.get('region_processor')
        self.table_extractor = components.get('table_extractor')
        self.markdown_renderer = components.get('markdown_renderer')
        self.snapshot_processor = components.get('page_processor') # This is the SnapshotProcessor
        self.reading_order_planner = components.get('reading_order_planner')
        self.reading_order_referee = components.get('reading_order_referee')
        self.font_analyzer = components.get('font_analyzer')
        self.layout_refiner = components.get('layout_refiner')
        self.table_anchor_associator = components.get('table_anchor_associator')
        self.vlm_client = components.get('vlm_client')

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU for two [x1,y1,x2,y2] bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = max(0.0, (bbox1[2] - bbox1[0])) * max(0.0, (bbox1[3] - bbox1[1]))
        area2 = max(0.0, (bbox2[2] - bbox2[0])) * max(0.0, (bbox2[3] - bbox2[1]))
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _is_contained(self, inner_bbox: List[float], outer_bbox: List[float], threshold: float = 0.8) -> bool:
        """Check if inner_bbox is contained within outer_bbox by at least threshold area."""
        x1 = max(inner_bbox[0], outer_bbox[0])
        y1 = max(inner_bbox[1], outer_bbox[1])
        x2 = min(inner_bbox[2], outer_bbox[2])
        y2 = min(inner_bbox[3], outer_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        inter_area = (x2 - x1) * (y2 - y1)
        inner_area = max(0.0, inner_bbox[2] - inner_bbox[0]) * max(0.0, inner_bbox[3] - inner_bbox[1])
        
        if inner_area <= 0:
            return False
            
        return (inter_area / inner_area) >= threshold

    def _dedupe_table_regions(self, table_regions: List[Dict[str, Any]], iou_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Deduplicate near-identical table detections, keeping higher-confidence ones."""
        if not table_regions:
            return table_regions

        sorted_regions = sorted(
            table_regions,
            key=lambda region: float(region.get('confidence', 0.0)),
            reverse=True,
        )

        kept: List[Dict[str, Any]] = []
        for region in sorted_regions:
            bbox = region.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            duplicate = False
            for existing in kept:
                existing_bbox = existing.get('bbox')
                if not existing_bbox or len(existing_bbox) != 4:
                    continue
                if self._compute_iou(bbox, existing_bbox) >= iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(region)

        return kept
        
    def process_page(self, page: fitz.Page, page_num: int, current_pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF page using initialized components."""
        logger.info(f"Processing page {page_num}")
        
        try:
            # Extract page information
            page_info = {
                "page_num": page_num,
                "page_size": {
                    "width": page.rect.width,
                    "height": page.rect.height
                }
            }
            
            # Define standard image metadata
            pdf_id = Path(current_pdf_path).stem
            page_id = f"{pdf_id}_p{page_num:03d}"
            vlm_metadata = {
                "image_id": page_id,
                "pdf_path": current_pdf_path,
                "page_num": page_num
            }
            
            # Convert page to image for processing
            page_image, dpi_scale = convert_page_to_image(page)
            
            # Save page thumbnail
            thumbnail_path = self.thumbnails_dir / f"page_{page_num:02d}.png"
            save_image(page_image, thumbnail_path, f"page thumbnail: page_{page_num:02d}.png")
            page_info["thumbnail_path"] = f"{OUTPUT_CONFIG['thumbnails_subdir']}/page_{page_num:02d}.png"
            
            # Step 0: Analyze document to compute adaptive profile
            logger.info("Analyzing document for adaptive thresholds")
            doc_profile = self.document_analyzer.analyze(page_image)
            logger.info(f"Document type detected: {doc_profile.document_type.value}")
            
            # Step 1: Native Text Extraction (Recovering orphans missed by YOLO)
            logger.info("Extracting native text blocks for orphan recovery (Line-level aware)")
            page_dict = page.get_text("dict")
            text_blocks_pdf = []
            b_val = 0
            for block in page_dict.get("blocks", []):
                if block.get("type", 1) != 0: 
                    continue # Skip image blocks
                
                b_bbox = block.get("bbox")
                b_height = b_bbox[3] - b_bbox[1]
                
                if b_height > 75:
                    # Break into sequence of logical line groups to prevent monolithic cross-table merging
                    current_group = None
                    last_line_y = None
                    
                    for line in block.get("lines", []):
                        l_text = " ".join([s.get("text", "") for s in line.get("spans", [])]).strip()
                        if not l_text: 
                            continue
                            
                        l_y0 = line.get("bbox")[1]
                        l_y1 = line.get("bbox")[3]
                        l_x0 = line.get("bbox")[0]
                        l_x2 = line.get("bbox")[2]
                        
                        # If vertical gap to last line is small (e.g. < 15 points), merge it
                        if current_group and last_line_y is not None and (l_y0 - last_line_y) < 15:
                            current_group["lines"].append(l_text)
                            current_group["bbox"][0] = min(current_group["bbox"][0], l_x0)
                            current_group["bbox"][2] = max(current_group["bbox"][2], l_x2)
                            current_group["bbox"][3] = max(current_group["bbox"][3], l_y1)
                        else:
                            # Start a new group
                            if current_group:
                                text_blocks_pdf.append({
                                    "type": "text",
                                    "bbox": current_group["bbox"],
                                    "text": "\n".join(current_group["lines"]),
                                    "source": "native_pdf",
                                    "confidence": 1.0,
                                    "id": f"native_{page_num}_{b_val}"
                                })
                                b_val += 1
                                
                            current_group = {
                                "lines": [l_text],
                                "bbox": [l_x0, l_y0, l_x2, l_y1]
                            }
                        last_line_y = l_y1
                    
                    if current_group:
                        text_blocks_pdf.append({
                            "type": "text",
                            "bbox": current_group["bbox"],
                            "text": "\n".join(current_group["lines"]),
                            "source": "native_pdf",
                            "confidence": 1.0,
                            "id": f"native_{page_num}_{b_val}"
                        })
                        b_val += 1
                else:
                    # Keep standard block structure
                    block_lines = []
                    for line in block.get("lines", []):
                        l_text = " ".join([s.get("text", "") for s in line.get("spans", [])]).strip()
                        if l_text:
                            block_lines.append(l_text)
                    b_text = "\n".join(block_lines).strip()
                    
                    if b_text:
                        text_blocks_pdf.append({
                            "type": "text",
                            "bbox": list(b_bbox),
                            "text": b_text,
                            "source": "native_pdf",
                            "confidence": 1.0,
                            "id": f"native_{page_num}_{b_val}"
                        })
                        b_val += 1
            
            # Convert native blocks to image space for hierarchical processing
            text_blocks_image = convert_regions_to_image_coords(text_blocks_pdf, dpi_scale)
            logger.info(f"Extracted {len(text_blocks_image)} native text blocks")

            # Step 2: Detect layout regions using layout models (already in Image Space)
            logger.info("Detecting layout regions with ensemble")
            layout_regions = self.layout_detector.detect_layout_regions(
                page_image, 
                debug=self.debug_mode,
                use_ensemble=True 
            )
            logger.info(f"Detected {len(layout_regions)} layout regions")

            # Layout Refinement (Gap Analysis)
            # Find possible missed content using VLM
            if self.layout_refiner and self.vlm_client:
                logger.info("Starting SOTA Layout Refinement (Gap Analysis)...")
                pil_page = Image.fromarray(page_image)
                
                # Convert layout_regions (dicts) to LayoutRegion objects
                regions_obj = [LayoutRegion.from_dict(r, overrides={'page_num': page_num}) for r in layout_regions]
                
                logger.info("Executing Layout Refiner (Visual Gap Analysis)...")
                # Refine ensemble (Gap Analysis & Snap)
                refined_objs = self.layout_refiner.refine_layout_ensemble(
                    pil_page, 
                    regions_obj, 
                    page_num,
                    metadata=vlm_metadata
                )
                
                # Convert back to dict format
                layout_regions = [r.to_dict() for r in refined_objs]
                logger.info(f"Refinement complete: {len(layout_regions)} regions")

            # Remove over-eager figures that swallow tables BEFORE they absorb native text
            # And remove over-eager text regions that vertically span multiple tables
            filtered_regions = []
            
            table_y_spans = []
            for r in layout_regions:
                if r.get('type', '').lower() == 'table':
                    table_y_spans.append((r['bbox'][1], r['bbox'][3]))
                    
            for r in layout_regions:
                r_type = r.get('type', '').lower()
                
                if r_type == 'figure':
                    m_bbox = r.get('bbox')
                    swallows_table = False
                    for other_r in layout_regions:
                        if other_r.get('type', '').lower() == 'table':
                            o_bbox = other_r.get('bbox')
                            if m_bbox and o_bbox and self._is_contained(o_bbox, m_bbox, 0.8):
                                swallows_table = True
                                break
                    if swallows_table:
                        logger.info(f"Removing over-eager Figure {r.get('id', 'unknown')} because it swallows a Table.")
                        continue
                        
                elif r_type in ['text', 'list', 'title', 'heading', 'caption']:
                    m_bbox = r.get('bbox')
                    spanned = 0
                    for ty1, ty2 in table_y_spans:
                        inter_y1 = max(m_bbox[1], ty1)
                        inter_y2 = min(m_bbox[3], ty2)
                        inter_height = inter_y2 - inter_y1
                        table_height = ty2 - ty1
                        if inter_height > 0 and inter_height > (table_height * 0.1): 
                            spanned += 1
                    if spanned >= 2:
                        logger.info(f"Removing monolithic Text region {r.get('id', 'unknown')} because it spans {spanned} tables!")
                        continue
                        
                filtered_regions.append(r)
            layout_regions = filtered_regions

            # Step 7 (Pre-Conversion): Process regions hierarchically in IMAGE SPACE
            # This ensures LayoutLMv3 crops are accurate
            logger.info("Processing regions hierarchically with LayoutLMv3 (IMAGE SPACE)")
            layout_regions = self.region_processor.process_regions_hierarchically(
                layout_regions, 
                text_blocks_image,
                page_image=page_image
            )

            # Step 4: Associate figures with captions (before coordinate conversion)
            logger.info("Associating figures with captions")
            layout_regions = self.figure_caption_processor.associate_captions(
                layout_regions, 
                doc_profile
            )

            # Step 3: Process table regions (smart hybrid: pdfplumber + PaddleOCR)
            table_regions = [r for r in layout_regions if r.get('type') in ['Table', 'table']]
            deduped_table_regions = self._dedupe_table_regions(table_regions)
            if len(deduped_table_regions) < len(table_regions):
                logger.info(
                    f"Deduplicated table regions: {len(table_regions)} -> {len(deduped_table_regions)}"
                )
            table_regions = deduped_table_regions

            for i, table_region in enumerate(table_regions):
                logger.info(f"Processing table region {i+1} with confidence {table_region.get('confidence', 0):.3f}")
                
                # Convert table bbox from image space to PDF space for pdfplumber
                image_bbox = table_region['bbox']
                pdf_bbox = [coord / dpi_scale for coord in image_bbox]
                
                # Pass PDF path for pdfplumber extraction
                table_data = self.table_extractor.extract_table_structure(
                    page_image, 
                    image_bbox,  # Use image bbox for cropping
                    page_num, 
                    i+1,
                    doc_profile=doc_profile,
                    pdf_path=current_pdf_path,
                    pdf_page_num=page_num - 1,
                    pdf_bbox=pdf_bbox,
                    fitz_page=page,     # Pass the actual PyMuPDF page object required by tables_v2
                    vlm_metadata=vlm_metadata
                )
                table_region['table_data'] = table_data
                
                # Save table image
                table_image_name = f"table_page_{page_num:02d}_{i+1:02d}.png"
                table_region['table_image_path'] = table_image_name
                logger.info(f"Saved table image: {table_image_name}")
            
            # Step 3.5: Create visualizations BEFORE coordinate conversion
            create_bounding_box_visualization_wrapper(page_image, layout_regions, page_num, self.thumbnails_dir, self.debug_mode)
            
            # Step 3.6: Attach region snapshots for Figure/Table regions (before coordinate conversion)
            if self.snapshot_processor:
                layout_regions = self.snapshot_processor.attach_region_snapshots(page_image, layout_regions)
            
            # Step 5: Convert layout regions from image coordinates to PDF coordinates
            logger.info("Converting layout regions to PDF coordinate space")
            merged_regions = convert_regions_to_pdf_coords(layout_regions, dpi_scale)
            
            # Step 7: (Replaced by Image-Space Step 7 above)
            logger.info(f"Hierarchical processing completed: {len(merged_regions)} regions")
            
            # Merge remaining regions
            logger.info("Merging processed regions")
            # We pass empty list for text_blocks here as they are already integrated
            merged_regions = merge_regions(merged_regions, [])
            
            # Step 8.5: Final cleanup - sort by Y-coordinate and remove duplicates
            logger.info("Final cleanup: sorting and deduplication")
            merged_regions = cleanup_regions(merged_regions)
            
            # Step 8.75: Extract text content for regions (Required since we skipped native block extraction)
            logger.info("Extracting text content from PDF for detected regions (with Quality Check)")
            
            for r in merged_regions:
                # If region has no text (most won't), extract it from PDF
                region_text = r.get('text') or ''
                if not region_text.strip():
                    bbox = r.get('bbox')
                    if bbox:
                        try:
                            # Verify bbox validity
                            rect = fitz.Rect(bbox)
                            # Extract text
                            text_content = page.get_textbox(rect)
                            
                            # Fallback if get_textbox returns empty
                            if not text_content.strip():
                                text_content = page.get_text("text", clip=rect)
                            
                            # --- QUALITY CHECK & REPAIR ---
                            # Check if text is garbage (PUA characters or gibberish)
                            is_good_quality = assess_text_quality([{'text': text_content}])
                            
                            if not is_good_quality and self.ocr_engine.is_available():
                                logger.warning(f"Region {r.get('id')} has poor text quality. Attempting OCR repair.")
                                # Convert PDF bbox to Image bbox
                                x1, y1, x2, y2 = [int(c * dpi_scale) for c in bbox]
                                # Clip to image bounds
                                h, w = page_image.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    crop = page_image[y1:y2, x1:x2]
                                    ocr_result = self.ocr_engine.extract_text_from_image(crop)
                                    # ocr_result is list of blocks usually? Or text? 
                                    # Wrapper returns list of dicts. JOIN them.
                                    repaired_text = " ".join([res.get('text', '') for res in ocr_result])
                                    
                                    if repaired_text.strip():
                                        text_content = repaired_text
                                        r['source'] = 'ocr_repaired'
                                        logger.info(f"Repaired text for region {r.get('id')}")
                            
                            r['text'] = text_content
                        except Exception as e:
                            logger.warning(f"Failed to extract/repair text for region {r.get('id', 'unknown')}: {e}")

            # Apply Reading Order
            logger.info("Resolving Reading Order flow...")
            # 9a: Ask VLM Planner for layout classification
            layout_prior = None
            if self.reading_order_planner:
                logger.info("Asking VLM Planner for layout classification...")
                layout_prior = self.reading_order_planner.generate_priors(
                    page_image,
                    metadata=vlm_metadata
                )
                logger.info(f"VLM Planner: layout='{layout_prior.layout_type}', strategy='{layout_prior.suggested_strategy}'")
            
            # 9b: Apply deterministic reading order (potentially overridden by VLM strategy)
            logger.info("Applying Final Recursive XY-Cut Reading Order")
            merged_regions = self.reading_order_resolver.order_regions(
                merged_regions,
                page_image,
                doc_profile,
                strategy_override=layout_prior.suggested_strategy if layout_prior else None
            )
            # 9c: Apply Stylesheet Grounding (Physical Font Analysis)
            if self.font_analyzer:
                logger.info("Applying physical font analysis to regions...")
                merged_regions = self.font_analyzer.assign_fonts_to_regions(merged_regions, page_num)
            
            logger.info("Reading order applied successfully")
            logger.info(f"After cleanup: {len(merged_regions)} regions")
            
            # Table Anchor Association Pass (attach table titles/captions)
            if self.table_anchor_associator:
                merged_regions = self.table_anchor_associator.associate_anchors(merged_regions)
            
            # Generate clean markdown content
            logger.info("Generating markdown content...")
            self.markdown_renderer.clean_output = not self.debug_mode  # Clean output unless debug
            markdown_content = self.markdown_renderer.extract_markdown_from_regions(merged_regions)

            # Step 11: VLM Referee QA (Professional Reading Order Check)
            stats = {
                "total_regions": len(merged_regions),
                "text_blocks": len(text_blocks_pdf),
                "layout_regions": len(layout_regions),
                "tables_found": len([r for r in layout_regions if r['type'] in ['Table', 'table']]),
            }
            
            if self.reading_order_referee:
                logger.info("🤖 VLM Referee: Verifying reading order for interleaved columns...")
                qa_result = self.reading_order_referee.verify_order(
                    page_image, 
                    markdown_content,
                    metadata=vlm_metadata
                )
                
                # Check for "rerun_column_first" action (Specialized 3-Column Recovery)
                if qa_result.suggested_action == "rerun_column_first":
                    # Only retry if we haven't already used this strategy
                    already_column_first = layout_prior and layout_prior.suggested_strategy == "xy_cut_column_first"
                    
                    if not already_column_first:
                        logger.warning(f"🤖 VLM Referee: Interleaved columns detected (Score: {qa_result.score}/10). Suggestion: {qa_result.reasoning}")
                        logger.info("🔄 Retrying extraction with 'xy_cut_column_first' strategy...")
                        
                        # Rerun reading order
                        merged_regions = self.reading_order_resolver.order_regions(
                            merged_regions,
                            page_image,
                            doc_profile,
                            strategy_override="xy_cut_column_first"
                        )
                        # Re-render markdown
                        markdown_content = self.markdown_renderer.extract_markdown_from_regions(merged_regions)
                        
                        # Update stats to reflect the professional recovery
                        stats["final_reading_order_strategy"] = "xy_cut_column_first"
                        stats["referee_score"] = qa_result.score
                        stats["referee_action"] = "rerun_column_first_success"
                    else:
                        logger.info(f"🤖 VLM Referee: Suggestion 'rerun_column_first' skipped (already using column-major sorting). Score: {qa_result.score}/10")
                        stats["referee_score"] = qa_result.score
                elif qa_result.suggested_action == "accept":
                    logger.info(f"✅ VLM Referee: Accepted reading order (Score: {qa_result.score}/10)")
                    stats["referee_score"] = qa_result.score
                    stats["referee_action"] = "accept"
                else:
                    logger.info(f"⚠️ VLM Referee: Action suggested: {qa_result.suggested_action} (Score: {qa_result.score}/10)")
                    stats["referee_score"] = qa_result.score
                    stats["referee_action"] = qa_result.suggested_action

            
            # Step 7 (Viz): Create debug visualizations if enabled (using image-space coordinates)
            if self.debug_mode:
                # Convert merged regions back to image space for visualization
                image_space_regions = convert_regions_to_image_coords(merged_regions, dpi_scale)
                create_debug_visualizations(page_image, image_space_regions, page_num, self.thumbnails_dir, self.debug_dir, self.debug_mode)
            
            # Aggregate statistics (using counting functions from result_builder - implicitly here using method calls or inline)
            # Actually we can't easily import result_builder here without circular deps if result_builder uses classes from here.
            # But result_builder functions are standalone. We can duplicate the simple counting logic or strict output.
            
            # Simple inline counting to avoid dependency for now
            def count_types(regs):
                c = {}
                for reg in regs:
                    t = reg.get('type', 'unknown')
                    c[t] = c.get(t, 0) + 1
                return c
            
            def count_methods(regs):
                c = {}
                for reg in regs:
                    s = reg.get('source', 'unknown')
                    c[s] = c.get(s, 0) + 1
                return c

            # Update metrics in existing stats dict
            stats.update({
                "total_regions": len(merged_regions),
                "region_types": count_types(merged_regions),
                "processing_methods": count_methods(merged_regions)
            })
            
            # Build page result
            page_result = {
                **page_info,
                "regions": merged_regions,
                "markdown": markdown_content,
                "stats": stats
            }
            
            logger.info(f"Page {page_num} processed successfully: {len(merged_regions)} total regions")
            return page_result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "page_num": page_num,
                "error": str(e),
                "regions": [],
                "markdown": "",
                "stats": {}
            }
