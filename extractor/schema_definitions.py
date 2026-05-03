"""
Librarian Schema Definitions: Modular Technical Taxonomy
========================================================
High-fidelity Pydantic models for industrial hardware extraction.

Design Reference:
  - IEC 61076 (Connectors)
  - ISA-95 (Industrial Data Modeling)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class SourceEvidence(BaseModel):
    """Provenance for a specific extracted fact or block."""
    text_snippet: Optional[str] = Field(None, description="The exact text from the document providing this data")
    page_number: Optional[int] = Field(None, description="Page number where the snippet was found")
    confidence: float = 1.0


class ParameterType(str, Enum):
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    ENVIRONMENTAL = "environmental"
    LOGISTICAL = "logistical"


class TechParameter(BaseModel):
    """A single technical specification (e.g., 'Operating Voltage: 24V')."""
    name: str = Field(..., description="Name of the parameter (e.g., 'Supply Voltage')")
    value: str = Field(..., description="The value as a string (e.g., '18...30')")
    unit: Optional[str] = Field(None, description="The unit (e.g., 'V DC', 'kg', 'mm')")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    param_type: ParameterType = ParameterType.ELECTRICAL
    source_evidence: Optional[SourceEvidence] = None


class PinAssignment(BaseModel):
    """Detailed pin-level mapping for a connector."""
    pin: str = Field(..., description="Pin number or ID (e.g., '1', 'A')")
    signal: str = Field(..., description="Signal name (e.g., 'V+', 'Bus A')")
    function: Optional[str] = Field(None, description="Description of use")
    wire_color: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class ConnectorSpec(BaseModel):
    """Representation of an industrial connector (M12, M8, RJ45)."""
    name: str = Field(..., description="Connector identifier (e.g., 'Port 0', 'Power In')")
    type: str = Field(..., description="Physical type (e.g., 'M12 female', 'M8 male')")
    coding: Optional[str] = Field(None, description="Coding (e.g., 'A-coded', 'D-coded')")
    pins: List[PinAssignment] = Field(default_factory=list)
    source_evidence: Optional[SourceEvidence] = None


class LEDBehavior(BaseModel):
    """Logic for diagnostic LEDs."""
    name: str = Field(..., description="LED label (e.g., 'US', 'ERR')")
    state: str = Field(..., description="Condition (e.g., 'flashing rouge', 'on green')")
    meaning: str = Field(..., description="Diagnostic meaning")
    source_evidence: Optional[SourceEvidence] = None


# --- MODULAR BLOCKS FOR DYNAMIC SYNTHESIS ---

class IdentificationBlock(BaseModel):
    """Core product and document identification."""
    product_name: Optional[str] = None
    art_no: Optional[str] = None
    manufacturer: Optional[str] = None
    gtin: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class ElectricalModule(BaseModel):
    """Module for electrical parameters."""
    parameters: List[TechParameter] = Field(default_factory=list)
    source_evidence: Optional[SourceEvidence] = None


class MechanicalModule(BaseModel):
    """Module for mechanical parameters."""
    parameters: List[TechParameter] = Field(default_factory=list)
    source_evidence: Optional[SourceEvidence] = None


class EnvironmentalModule(BaseModel):
    """Module for environmental parameters (Temp, IP Rating)."""
    parameters: List[TechParameter] = Field(default_factory=list)
    source_evidence: Optional[SourceEvidence] = None


class ConnectorModule(BaseModel):
    """Module for connector and pinout definitions."""
    connectors: List[ConnectorSpec] = Field(default_factory=list)


class DiagnosticModule(BaseModel):
    """Module for LED and error code behavior."""
    leds: List[LEDBehavior] = Field(default_factory=list)


# --- LEGACY UNIVERSAL SCHEMAS (Refactored to use Blocks) ---

class LibrarianUniversalHardware(BaseModel):
    """
    The master 'Universal' container for industrial hardware.
    Includes identity, technical parameters (split by type), connectors, and diagnostics.
    """
    parameters: List[TechParameter] = Field(
        default_factory=list, 
        max_items=60,
        description="ALL unique technical parameters. Extract parameters FIRST. Max 60 unique entries."
    )
    identity: Optional[IdentificationBlock] = Field(default_factory=IdentificationBlock)
    connectors: List[ConnectorSpec] = Field(
        default_factory=list, 
        max_items=40,
        description="Industrial connectors and pinouts. UNIQUE entries only. Max 40."
    )
    diagnostics: List[LEDBehavior] = Field(
        default_factory=list, 
        max_items=40,
        description="Diagnostic LED states and meanings. UNIQUE entries only. Max 40."
    )
    
    reasoning_thoughts: Optional[str] = Field(None, description="The internal logic/CoT used to arrive at this extraction")
    page_references: List[int] = Field(
        default_factory=list, 
        max_items=10,
        description="List ONLY primary pages where major data was found (max 10). NO LOOPS."
    )
    confidence_score: float = 1.0


class DocumentIdentity(BaseModel):
    """General identity for non-hardware documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = Field(None, description="Classification (e.g., invoice, contract, paper)")


class GeneralEntity(BaseModel):
    """Generic entity extraction (Person, Org, Location)."""
    name: str
    category: str = Field(..., description="Type (e.g. PERSON, ORG, GPE)")
    context: str = Field(..., description="Snipped from the text where found")


class TimelineEvent(BaseModel):
    """Chronological event extraction."""
    date: str
    event: str


class GeneralInfoBlock(BaseModel):
    """Broad summary and purpose of the document."""
    title: Optional[str] = None
    summary: Optional[str] = None
    purpose: Optional[str] = Field(None, description="The intended goal or audience")
    source_evidence: Optional[SourceEvidence] = None


class LibrarianGeneralClerk(BaseModel):
    """
    Exhaustive 'General' schema for diverse PDFs.
    """
    info: Optional[GeneralInfoBlock] = Field(default_factory=GeneralInfoBlock)
    entities: List[GeneralEntity] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    tables_markdown: List[str] = Field(default_factory=list, description="Raw markdown representation of all tables found")
    reasoning_thoughts: Optional[str] = Field(None, description="The internal logic/CoT used to arrive at this extraction")
    page_references: List[int] = Field(default_factory=list)
    confidence_score: float = 1.0


class InvoiceLineItem(BaseModel):
    """Structured representation of a single invoice line item."""
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    amount: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class VATBreakdown(BaseModel):
    """VAT summary block for an invoice."""
    rate: Optional[str] = None
    base_amount: Optional[str] = None
    vat_amount: Optional[str] = None


class InvoiceHeaderBlock(BaseModel):
    """Core billing metadata for an invoice."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier: Optional[str] = None
    recipient: Optional[str] = None
    currency: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class InvoiceLinesBlock(BaseModel):
    """Table-like line items in an invoice."""
    line_items: List[InvoiceLineItem] = Field(default_factory=list)


class TotalsBlock(BaseModel):
    """Summary financial data."""
    subtotal: Optional[str] = None
    vat: List[VATBreakdown] = Field(default_factory=list)
    total: Optional[str] = None
    source_evidence: Optional[SourceEvidence] = None


class LibrarianInvoiceRecord(BaseModel):
    """Invoice-specific schema for corporate billing documents."""
    header: Optional[InvoiceHeaderBlock] = Field(default_factory=InvoiceHeaderBlock)
    lines: Optional[InvoiceLinesBlock] = Field(default_factory=InvoiceLinesBlock)
    totals: Optional[TotalsBlock] = Field(default_factory=TotalsBlock)
    page_references: List[int] = Field(default_factory=list)
    confidence_score: float = 1.0


class LibrarianBusinessRecord(BaseModel):
    """General business schema for non-invoice corporate documents."""
    identity: Optional[DocumentIdentity] = Field(default_factory=DocumentIdentity)
    key_points: List[str] = Field(default_factory=list)
    entities: List[GeneralEntity] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    tables_markdown: List[str] = Field(default_factory=list)
    page_references: List[int] = Field(default_factory=list)
    confidence_score: float = 1.0


class ProductCommercialData(BaseModel):
    """Catalog and commercial metadata commonly found in product PDFs."""
    gtin: Optional[str] = None
    customs_tariff_number: Optional[str] = None
    packaging_unit: Optional[str] = None
    eclass_codes: List[str] = Field(default_factory=list)
    etim_codes: List[str] = Field(default_factory=list)


class ProductPdfRecord(BaseModel):
    """
    Focused schema for catalog-style product PDFs and cable/connector product sheets.
    This is intentionally lighter than TechnicalDatasheetRecord and avoids requiring
    complete pin maps or LED diagnostics when the document is mostly article-level specs.
    """
    product_name: Optional[str] = None
    art_no: Optional[str] = None
    manufacturer: Optional[str] = None
    connector_type: Optional[str] = None
    pole_count: Optional[str] = None
    cable_length: Optional[str] = None
    material: Optional[str] = None
    protection_rating: Optional[str] = None
    electrical_specs: List[TechParameter] = Field(default_factory=list)
    standards: List[str] = Field(default_factory=list)
    commercial_data: Optional[ProductCommercialData] = Field(default_factory=ProductCommercialData)
    page_references: List[int] = Field(default_factory=list)
    confidence_score: float = 1.0


class TechnicalDatasheetRecord(BaseModel):
    """
    Focused schema for single-product technical datasheets (IEC-style).
    Designed for compact datasheets with parameter tables, connector pinouts, and LED diagnostics.
    Preferred over LibrarianUniversalHardware when the document is a standalone product datasheet.
    """
    identity: Optional[IdentificationBlock] = Field(default_factory=IdentificationBlock)
    parameters: List[TechParameter] = Field(
        default_factory=list,
        description=(
            "ALL technical parameters extracted from parameter tables. "
            "Include electrical (supply voltage, current consumption, power dissipation), "
            "mechanical (dimensions, weight, housing material, mounting), "
            "and environmental (operating temperature range, protection class IP xx, EMC) parameters. "
            "Each row of a parameter table should become one TechParameter entry."
        ),
    )
    connectors: List[ConnectorSpec] = Field(
        default_factory=list,
        description=(
            "ALL connectors with COMPLETE pin assignments. "
            "For EACH connector: name (e.g. 'Port 0', 'Power Supply'), "
            "type (e.g. 'M12 male A-coded', 'M8 female'), "
            "and EVERY pin with signal name (e.g. 'V+', 'GND', 'Bus A') and function."
        ),
    )
    diagnostics: List[LEDBehavior] = Field(
        default_factory=list,
        description=(
            "ALL LED indicators and their states/meanings. "
            "Each LED-state combination is one entry (e.g. 'ERR LED flashing red → Bus error')."
        ),
    )
    standards: List[str] = Field(
        default_factory=list,
        description=(
            "Standards and certifications mentioned "
            "(e.g. 'IEC 61076-2-101', 'IP67', 'UL 508', 'CE', 'PROFINET', 'IO-Link')."
        ),
    )
    reasoning_thoughts: Optional[str] = Field(
        None,
        description="Step-by-step extraction reasoning (internal chain-of-thought).",
    )
    page_references: List[int] = Field(default_factory=list)
    confidence_score: float = 1.0
