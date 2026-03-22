import logging
import time
import uuid
from typing import List, Optional, Dict, Any
import re
import numpy as np

try:
    import fitz
except ImportError:
    fitz = None

from .types import (
    BBoxPDF,
    TablePrimitives,
    TableCell,
    TableResult,
    TableType,
    TableQAMetrics,
)

logger = logging.getLogger(__name__)

class TableExtractorVLM:
    """
    Extract technical tables using a Multi-VLM Specialist Chain.
    
    This extractor bypasses the native PDF text layer to avoid jumbled 
    encoding, performing direct vision-to-grid reconstruction.
    """
    
    def __init__(
        self,
        vlm_client: Any = None,
        render_dpi: int = 200,
        use_specialist_florence: bool = True,
    ):
        """
        Args:
            vlm_client: Client for VLM inference (e.g. Ollama)
            render_dpi: DPI for rendering table region
            use_specialist_florence: Whether to use Florence-2 for OCR refinement
        """
        self.vlm_client = vlm_client
        self.render_dpi = render_dpi
        self.use_specialist_florence = use_specialist_florence
        self.method_name = "specialist_vlm_ensemble"

    def extract(
        self,
        bbox: BBoxPDF,
        primitives: TablePrimitives,
        page: "fitz.Page" = None,
        table_id: Optional[str] = None,
    ) -> TableResult:
        """Extract table using Specialist VLM Chain."""
        start_time = time.time()
        table_id = table_id or str(uuid.uuid4())[:8]
        
        if page is None:
            return self._empty_result(table_id, bbox, start_time, "No page provided for VLM rendering")

        # 1. Render Table to Image
        try:
            pil_image = self._render_to_image(page, bbox)
        except Exception as e:
            logger.error(f"VLM Render failure: {e}")
            return self._empty_result(table_id, bbox, start_time, f"Render failed: {e}")

        # 2. Get Structural Markdown from VLM (MiniCPM or Custom Specialist)
        try:
            prompt = self._get_structural_prompt(table_id)
            logger.info(f"[vlm_extractor] Requesting markdown grid from VLM for table {table_id}...")
            
            # Fix: VLMClient.generate expects (image, prompt, is_complex)
            markdown_table = self.vlm_client.generate(
                image=pil_image,
                prompt=prompt,
                is_complex=True
            )
            
            if not markdown_table or "Error" in markdown_table:
                 logger.error(f"[vlm_extractor] VLM returned error or empty response: {markdown_table}")
                 return self._empty_result(table_id, bbox, start_time, f"VLM Error: {markdown_table}")
                 
            logger.info(f"[vlm_extractor] VLM Success. Received {len(markdown_table)} chars.")
            logger.debug(f"[vlm_extractor] Raw Markdown: \n{markdown_table}")
            
        except Exception as e:
            logger.error(f"[vlm_extractor] VLM Structural Extraction failure: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._empty_result(table_id, bbox, start_time, f"VLM failed: {e}")

        # 3. Parse Markdown into TableCells
        cells = self._parse_markdown_to_cells(markdown_table, bbox)
        
        if not cells:
            return self._empty_result(table_id, bbox, start_time, "VLM returned no valid table structure")

        elapsed = (time.time() - start_time) * 1000
        
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.VLM,
            method=self.method_name,
            cells=cells,
            num_rows=max((c.row for c in cells), default=0) + 1,
            num_cols=max((c.col for c in cells), default=0) + 1,
            extraction_time_ms=elapsed,
        )

    def _render_to_image(self, page: "fitz.Page", bbox: BBoxPDF) -> "Image.Image":
        """Render the bbox area of the page to a PIL Image."""
        from PIL import Image
        clip = fitz.Rect(bbox)
        zoom = self.render_dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def _get_structural_prompt(self, table_id: str) -> str:
        return (
            f"Analyze this technical table ({table_id}) and extract its 3-column data. "
            "Represent each row using these tags to ensure column alignment: "
            "<ROW><P>Parameter</P><C>Conditions</C><V>Value</V></ROW>\n"
            "Rules:\n"
            "- <P>: Parameter name (left)\n"
            "- <C>: Testing Conditions (middle). Prefix with '@@' if it's a specific rated condition (e.g., <C>@@24 V DC</C>).\n"
            "- <V>: Numeric Value (right)\n"
            "- If a row contains a GRAPH, DIAGRAM, or FIGURE, IGNORE all internal text and coordinates. Simply output: <ROW><P>Graph/Figure</P><C></C><V>[TECHNICAL_GRAPH]</V></ROW> and STOP investigating it.\n"
            "Output ONLY the <ROW> tags. No markdown pipes or preamble. Each physical horizontal line must strictly produce exactly one <ROW> tag. DO NOT REPEAT THE SAME PARAMETER. STOP IMMEDIATELY after the last row in the image."
        )

    def _parse_markdown_to_cells(self, vlm_output: str, table_bbox: BBoxPDF) -> List[TableCell]:
        """Robustly parse VLM output (supports tags or Markdown)."""
        cells = []
        import re
        
        # 1. Try Tag-based parsing (Robust for local VLMs)
        rows = re.findall(r'<ROW>(.*?)</ROW>', vlm_output, re.DOTALL | re.IGNORECASE)
        if rows:
            for i, row_content in enumerate(rows):
                p_match = re.search(r'<P>(.*?)</P>', row_content, re.DOTALL | re.IGNORECASE)
                c_match = re.search(r'<C>(.*?)</C>', row_content, re.DOTALL | re.IGNORECASE)
                v_match = re.search(r'<V>(.*?)</V>', row_content, re.DOTALL | re.IGNORECASE)
                
                texts = [
                    p_match.group(1).strip() if p_match else "",
                    c_match.group(1).strip() if c_match else "",
                    v_match.group(1).strip() if v_match else ""
                ]
                
                if not any(texts) and i > 0: continue
                    
                for j, text in enumerate(texts):
                    if text:
                        cells.append(TableCell(
                            row=i,
                            col=j,
                            text=text,
                        ))
            if cells: return cells

        # 2. Fallback to Markdown pipe parsing
        lines = [l.strip() for l in vlm_output.split('\n') if '|' in l]
        data_lines = []
        for line in lines:
            if re.match(r'^\|[\s\-\|:]+\|$', line): continue
            data_lines.append(line)

        for row_idx, line in enumerate(data_lines):
            cols = [col.strip() for col in line.split('|')]
            if cols[0] == "": cols = cols[1:]
            if cols[-1] == "": cols = cols[:-1]
            for col_idx, text in enumerate(cols):
                if text:
                    cells.append(TableCell(row=row_idx, col=col_idx, text=text))
        return cells

    def _empty_result(
        self,
        table_id: str,
        bbox: BBoxPDF,
        start_time: float,
        reason: str,
    ) -> TableResult:
        elapsed = (time.time() - start_time) * 1000
        return TableResult(
            table_id=table_id,
            bbox_pdf=bbox,
            table_type=TableType.VLM,
            method="vlm_empty",
            cells=[],
            qa=TableQAMetrics(passed=False, failure_reasons=[reason]),
            extraction_time_ms=elapsed,
        )
