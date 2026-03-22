"""
Enhanced Pipeline Module - Main Pipeline Class with Model Reuse
Exact architectural match to original pipeline.py but with modular components
"""

import logging
import fitz
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

import sys
# Add parent directory to path for imports if needed (though package structure should handle this)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, FONT_CONFIG, OUTPUT_CONFIG, setup_environment, get_output_paths

# Import components
from modules.layout_detector import LayoutDetector
from modules.ocr_engine import OCREngine
from modules.table_structure_model import TableStructureModel
from processors.table_extractor import TableExtractor
from processors.font_analyzer import FontAnalyzer
from processors.stylesheet_planner import StylesheetPlanner
from renderers.markdown_renderer import MarkdownRenderer
from processors.page_processor import PageProcessor as SnapshotProcessor
from analysis.document_analyzer import DocumentAnalyzer
from processors.reading_order import ReadingOrderResolver
from processors.reading_order_planner import ReadingOrderPlannerVLM
from processors.reading_order_referee import ReadingOrderRefereeVLM
from analysis.semantic_grouper import SemanticTextGrouper
from processors.figure_caption_processor import FigureCaptionProcessor
from processors.region_processor import RegionProcessor
from common.vlm_client import VLMClient
from processors.layout_refiner import LayoutRefiningAgent
from research.deprecated.distillation_agent import DistillationAgent

# Import new pipeline modules
from pipeline.page_processor import PageProcessor
from pipeline.result_builder import build_final_result, save_results
from pipeline.visualization_manager import generate_layout_pdf

# Setup environment
setup_environment()

logger = logging.getLogger(__name__)

class EnhancedPipeline:
    """Enhanced PDF processing pipeline with proper model integration and reuse."""
    
    def __init__(self, output_dir: str = None, debug_mode: bool = False, vlm_model: str = None, vlm_provider: str = None, strategy: str = None, distill: bool = False):
        """Initialize the enhanced PDF pipeline with model reuse, research strategies, and distillation."""
        logger.info(f"Initializing Enhanced PDF Pipeline with strategy: {strategy or 'default'}")
        
        self.debug_mode = debug_mode
        self.strategy = strategy
        self.distill = distill
        
        # Strategy-based model selection
        if strategy == 'gpt4o':
            self.vlm_config = {"model": "gpt-4o", "provider": "openai"}
        elif strategy == 'sota_os':
            # Use the absolute best OS models for research parity
            self.vlm_config = {"model": "internvl2-llama3-76b", "provider": "internvl"}
        elif strategy == 'fast_os':
            # Local efficiency focus
            self.vlm_config = {"model": "minicpm-v-2.6", "provider": "ollama"}
        else:
            self.vlm_config = {
                "model": vlm_model or "qwen2.5-vl:7b",
                "provider": vlm_provider
            }
        
        # Get output paths
        output_paths = get_output_paths(output_dir)
        
        self.output_dir = output_paths['base']
        self.thumbnails_dir = output_paths['thumbnails']
        self.tables_dir = output_paths['tables']
        self.figures_dir = output_paths['figures']
        self.debug_dir = output_paths['debug']
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        self.debug_mode = debug_mode
        if debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
        
        # Store configuration
        self.config = MODEL_CONFIG.copy()
        self.config.update(FONT_CONFIG)
        
        # Initialize output paths dictionary
        self.output_paths = {
            'main': self.output_dir,
            'thumbnails': self.thumbnails_dir,
            'tables': self.tables_dir,
            'figures': self.figures_dir,
            'debug': self.debug_dir,
            'json_file': self.output_dir / OUTPUT_CONFIG['json_filename'],
            'markdown_file': self.output_dir / OUTPUT_CONFIG['markdown_filename']
        }
        
        # Initialize components ONCE
        self.components = self._initialize_components()
        
        # Initialize PageProcessor
        self.page_processor = PageProcessor(
            components=self.components,
            output_paths=self.output_paths,
            debug_mode=self.debug_mode
        )
        
        logger.info(f"Enhanced PDF Pipeline initialized with output dir: {output_dir}")
        logger.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all pipeline components once for reuse across pages."""
        logger.info("Initializing pipeline components...")
        
        components = {}
        
        # Initialize document analyzer (computes adaptive thresholds)
        components['document_analyzer'] = DocumentAnalyzer()
        logger.info("Document analyzer initialized")
        
        # Initialize layout detector (loads model once)
        components['layout_detector'] = LayoutDetector(debug_mode=self.debug_mode)
        logger.info("Layout detector initialized")
        
        # Initialize OCR engine (loads model once)
        components['ocr_engine'] = OCREngine()
        logger.info("OCR engine initialized")
        
        # Initialize reading order resolver (deterministic fallback)
        components['reading_order_resolver'] = ReadingOrderResolver()
        logger.info("Reading order resolver initialized")
        
        # Initialize VLM Client
        vlm_client = VLMClient(config=self.vlm_config)
        
        # Initialize Distillation Agent if requested
        if self.distill:
            distillation_agent = DistillationAgent()
            vlm_client.observer = distillation_agent
            logger.info("Distillation Agent initialized and attached to VLM Client")
            
        components['vlm_client'] = vlm_client
        logger.info(f"VLM Client initialized ({vlm_client.provider_name}: {vlm_client.model})")
        
        # [STREAMLINED] Planners and Refiners are now deterministic (RT-DETR + XY-Cut)
        components['reading_order_planner'] = None
        components['reading_order_referee'] = None
        
        # Initialize semantic text grouper
        components['semantic_grouper'] = SemanticTextGrouper()
        logger.info("Semantic text grouper initialized")
        
        # Initialize figure-caption processor
        components['figure_caption_processor'] = FigureCaptionProcessor()
        logger.info("Figure-caption processor initialized")
        
        # Initialize region processor (hierarchical processing - switched to base RT-DETR)
        components['region_processor'] = RegionProcessor(use_layoutlm=False)
        logger.info("Region processor initialized (RT-DETR only)")
        
        # [STREAMLINED] Layout Refining is now handled by RT-DETR/TATR anchors
        components['layout_refiner'] = None

        
        # Initialize table structure model (TATR)
        components['table_structure_model'] = TableStructureModel()
        logger.info("Table structure model initialized")

        # Initialize table extractor with Tables v2 coordinator
        components['table_extractor'] = TableExtractor(
            self.output_paths,
            structure_model=components['table_structure_model'],
            vlm_client=vlm_client,
            strategy=self.strategy # Pass strategy for specialist routing
        )
        logger.info("Table extractor initialized")

        # Initialize Stylesheet Agent
        components['stylesheet_planner'] = StylesheetPlanner(vlm_client=vlm_client)
        logger.info("Stylesheet Planner initialized")
        
        # Initialize markdown renderer
        components['markdown_renderer'] = MarkdownRenderer(debug=self.debug_mode)
        logger.info("Markdown renderer initialized")
        
        # Initialize page processor for region snapshots
        components['page_processor'] = SnapshotProcessor(self.output_paths)
        logger.info("Snapshot processor initialized")
        
        logger.info("All pipeline components initialized successfully")
        return components
    
    def process_pdf(self, pdf_path: str, max_pages: int = None, page_range: str = None) -> Dict[str, Any]:
        """Process entire PDF document with enhanced functionality."""
        logger.info(f"Starting enhanced PDF processing: {pdf_path}")
        
        # Clear table extraction cache for new document
        if hasattr(self.components['table_extractor'], '_extracted_tables_cache'):
            self.components['table_extractor']._extracted_tables_cache.clear()
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            # Global Stylesheet Analysis
            logger.info("Starting Global Stylesheet Analysis...")
            font_analyzer = FontAnalyzer(pdf_path)
            self.components['font_analyzer'] = font_analyzer # Temporary for this run
            
            # Use the first page to hypothesize the stylesheet via VLM
            first_page = doc[0]
            from pipeline.coordinate_converter import convert_page_to_image
            page_image, _ = convert_page_to_image(first_page)
            from PIL import Image
            pil_image = Image.fromarray(page_image)
            
            hypothesized = self.components['stylesheet_planner'].generate_hypothesized_stylesheet(pil_image)
            grounded_stylesheet = self.components['stylesheet_planner'].ground_stylesheet(hypothesized, font_analyzer)
            
            self.components['markdown_renderer'].set_stylesheet(grounded_stylesheet)
            logger.info(f"Document Stylesheet grounded: H1={getattr(grounded_stylesheet.h1, 'size', 'N/A')}pt, Body={grounded_stylesheet.body.size}pt")
            
            # Update PageProcessor with the new components
            self.page_processor = PageProcessor(
                components=self.components,
                output_paths=self.output_paths,
                debug_mode=self.debug_mode
            )
            
            # Determine pages to process
            pages_to_process = list(range(total_pages)) # Default all 0-indexed
            
            if page_range:
                try:
                    selected_pages = []
                    parts = page_range.split(',')
                    for part in parts:
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            # Convert 1-based input to 0-based index
                            selected_pages.extend(range(start - 1, end))
                        else:
                            selected_pages.append(int(part) - 1)
                    
                    # Filter valid pages and unique sort
                    pages_to_process = sorted(list(set([p for p in selected_pages if 0 <= p < total_pages])))
                    logger.info(f"Processing specific pages: {[p+1 for p in pages_to_process]}")
                except ValueError as e:
                    raise ValueError(f"Invalid page range format: {page_range}") from e
            
            if max_pages:
                pages_to_process = pages_to_process[:max_pages]
                logger.info(f"Limited processing to first {len(pages_to_process)} pages")
            
            # Process each page using initialized components
            pages_data = []
            all_markdown = []
            
            for page_num in pages_to_process:
                page = doc[page_num]
                # Delegate to PageProcessor
                page_result = self.page_processor.process_page(page, page_num + 1, current_pdf_path=pdf_path)
                pages_data.append(page_result)
                
                # Collect markdown
                if page_result.get('markdown'):
                    all_markdown.append(f"## Page {page_num + 1}\\n\\n{page_result['markdown']}")
            
            # Build final result
            result = build_final_result(pdf_path, doc.page_count, pages_data, all_markdown, self.config)
            
            # Save results
            save_results(result, self.output_paths)
            
            # Generate consolidated layout PDF if requested
            generate_layout_pdf(self.output_dir, self.thumbnails_dir)
            
            logger.info(f"Enhanced PDF processing completed. Results saved to: {self.output_paths['json_file']}")
            logger.info(f"Markdown content saved to: {self.output_paths['markdown_file']}")
            logger.info(f"Summary: {result['summary']}")
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
