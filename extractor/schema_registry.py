"""
Librarian Schema Registry: Document Domain Mapping
==================================================
Maps document domains (detected by the Scout) to specific Pydantic models.
This allows for 'Universal' extraction across diverse PDF types.
"""

from typing import Dict, Type, Any
from pydantic import BaseModel
from .schema_definitions import (
    LibrarianUniversalHardware,
    LibrarianGeneralClerk,
    LibrarianInvoiceRecord,
    LibrarianBusinessRecord,
    ProductPdfRecord,
    TechnicalDatasheetRecord,
)

# Schema family-to-model mapping
SCHEMA_FAMILY_REGISTRY: Dict[str, Type[BaseModel]] = {
    "hardware_v1": LibrarianUniversalHardware,
    "product_pdf_v1": ProductPdfRecord,
    "technical_datasheet_v1": TechnicalDatasheetRecord,
    "invoice_v1": LibrarianInvoiceRecord,
    "business_v1": LibrarianBusinessRecord,
    "general_v1": LibrarianGeneralClerk,
}

# Domain-to-Schema Mapping
DOMAIN_REGISTRY: Dict[str, Type[BaseModel]] = {
    "Industrial": LibrarianUniversalHardware,
    "Industrial_product_pdf": ProductPdfRecord,
    "Industrial_datasheet": TechnicalDatasheetRecord,
    "Industrial_manual": LibrarianUniversalHardware,
    "General": LibrarianGeneralClerk,
    "Legal": LibrarianGeneralClerk,
    "Academic": LibrarianGeneralClerk,
    "Corporate": LibrarianInvoiceRecord,
}

DOMAIN_TO_FAMILY: Dict[str, str] = {
    "Industrial": "hardware_v1",
    "Industrial_product_pdf": "product_pdf_v1",
    "Industrial_datasheet": "technical_datasheet_v1",
    "Industrial_manual": "hardware_v1",
    "General": "general_v1",
    "Legal": "general_v1",
    "Academic": "general_v1",
    "Corporate": "invoice_v1",
}

def get_schema_for_domain(domain: str) -> Type[BaseModel]:
    """Retrieves the specialist schema for a given document domain."""
    return DOMAIN_REGISTRY.get(domain, LibrarianGeneralClerk)


def get_schema_family(domain: str) -> str:
    """Returns the schema family key for a domain."""
    return DOMAIN_TO_FAMILY.get(domain, "general_v1")


def get_schema_model(schema_family: str) -> Type[BaseModel]:
    """Returns a schema model for a given schema family key."""
    return SCHEMA_FAMILY_REGISTRY.get(schema_family, LibrarianGeneralClerk)


def list_schema_families() -> list:
    """List all available schema family keys."""
    return sorted(SCHEMA_FAMILY_REGISTRY.keys())

def get_active_modules_for_domain(domain: str) -> list:
    """Returns a description of active modules for a domain."""
    if domain in ("Industrial", "Industrial_manual"):
        return ["Electrical Parameters", "Mechanical Parameters", "Connector Pin Assignments", "LED Diagnostics"]
    if domain == "Industrial_product_pdf":
        return ["Product Identity", "Connector Summary", "Electrical Parameters", "Standards & Certifications", "Commercial Data"]
    if domain == "Industrial_datasheet":
        return ["Electrical Parameters", "Mechanical Parameters", "Connector Pin Assignments", "LED Diagnostics", "Standards & Certifications"]
    if domain == "Corporate":
        return ["Invoice Header", "Line Items", "Totals", "VAT"]
    return ["Identity", "Summary", "Entities", "Timeline", "Tables"]
