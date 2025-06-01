"""
Blood Report Analyzer Utilities Module

This module contains utility functions for PDF processing, 
LLM clients, and other helper functions.
"""

from .pdf_processor import BloodReportProcessor
from .llm_clients import LLMClients

# Make utilities available at package level
__all__ = [
    'BloodReportProcessor',
    'LLMClients'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Blood Report Analyzer Team'

# Package-level convenience functions
def create_pdf_processor():
    """Create a new PDF processor instance"""
    return BloodReportProcessor()

def create_llm_clients():
    """Create a new LLM clients instance"""
    return LLMClients()

# Utility constants
SUPPORTED_FILE_TYPES = ['pdf']
MAX_FILE_SIZE_MB = 10
SUPPORTED_BLOOD_PARAMETERS = [
    'hemoglobin', 'white_blood_cells', 'red_blood_cells', 'platelets',
    'glucose', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 
    'creatinine', 'urea', 'bilirubin', 'alt', 'ast'
]

# Error classes for the utils module
class PDFProcessingError(Exception):
    """Raised when PDF processing fails"""
    pass

class LLMClientError(Exception):
    """Raised when LLM client operations fail"""
    pass

class BloodDataExtractionError(Exception):
    """Raised when blood data extraction fails"""
    pass
