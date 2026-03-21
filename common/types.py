"""
Common Pydantic types shared across the PDF pipeline.
"""

from typing import List, Any, Optional
from pydantic import BaseModel, Field, model_validator, field_validator
from common.vlm_types import FontSignature


class BBox(BaseModel):
    """
    Bounding box in PDF coordinates (x0, y0, x1, y1).
    """
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode='before')
    @classmethod
    def parse_list_or_tuple(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) != 4:
                raise ValueError(f"BBox list/tuple must have 4 elements, got {len(data)}")
            return {'x0': data[0], 'y0': data[1], 'x1': data[2], 'y1': data[3]}
        return data

    @model_validator(mode='after')
    def validate_coordinates(self) -> 'BBox':
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be >= x0 ({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"y1 ({self.y1}) must be >= y0 ({self.y0})")
        return self

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]
    
    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BBox':
        if len(bbox) != 4:
            raise ValueError(f"BBox list must have 4 elements, got {len(bbox)}")
        return cls(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
        
    @property
    def width(self) -> float:
        return self.x1 - self.x0
        
    @property
    def height(self) -> float:
        return self.y1 - self.y0
        
    @property
    def area(self) -> float:
        return self.width * self.height

    def __getitem__(self, index):
        if index == 0: return self.x0
        if index == 1: return self.y0
        if index == 2: return self.x1
        if index == 3: return self.y1
        return [self.x0, self.y0, self.x1, self.y1][index]
    
    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1


class LayoutRegion(BaseModel):
    """
    A detected region on a PDF page.
    """
    region_id: str
    page_num: int
    type: str 
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    source: str 
    text: Optional[str] = None
    model_class: Optional[int] = None 
    
    # State fields used during processing
    table_data: Optional[Any] = None 
    table_image_path: Optional[str] = None
    is_figure: bool = False
    is_table: bool = False
    associated_figure: Optional[str] = None 
    
    # Caption association fields
    caption: Optional[str] = None
    caption_bbox: Optional[List[float]] = None
    caption_confidence: Optional[float] = None
    caption_source: Optional[str] = None
    
    # Merged regions tracking
    linked_regions: List[str] = Field(default_factory=list)
    
    # Text appearance (Strategy 3 DOM Grounding)
    font_signature: Optional[FontSignature] = None
    
    class Config:
        arbitrary_types_allowed = True 
        extra = 'allow' 
    
    @field_validator('bbox', mode='before')
    @classmethod
    def parse_bbox(cls, v):
        """Allow initializing with list [x0, y0, x1, y1]."""
        if isinstance(v, (list, tuple)):
            return BBox.from_list(list(v))
        return v
    
    @classmethod
    def from_dict(cls, data: dict, overrides: Optional[dict] = None) -> 'LayoutRegion':
        """Create LayoutRegion from legacy dictionary format with optional overrides."""
        mapped_data = data.copy()
        if overrides:
            mapped_data.update(overrides)
            
        # Map legacy keys
        if 'id' in data and 'region_id' not in mapped_data:
            mapped_data['region_id'] = data['id']
            
        return cls(**mapped_data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary consistent with legacy format."""
        return {
            "region_id": self.region_id,
            "page_num": self.page_num,
            "type": self.type,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "source": self.source,
            "text": self.text,
            "model_class": self.model_class
        }
