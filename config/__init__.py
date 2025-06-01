"""
Blood Report Analyzer Configuration Module

This module handles all configuration settings, environment variables,
and application constants.
"""

from .settings import Settings

# Make settings available at package level
__all__ = [
    'Settings'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Blood Report Analyzer Team'

# Create global settings instance
try:
    settings = Settings()
    __all__.append('settings')
except Exception as e:
    print(f"Warning: Could not initialize settings: {e}")
    settings = None

# Application constants
APP_NAME = "AI Blood Report Analyzer"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered blood test analysis and health recommendations"

# API configuration
DEFAULT_MODEL_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.3

# File processing constants
MAX_PDF_PAGES = 50
MAX_TEXT_LENGTH = 100000  # characters
SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de']

# Blood test categories
BLOOD_TEST_CATEGORIES = {
    'basic_metabolic': ['glucose', 'sodium', 'potassium', 'chloride', 'co2', 'bun', 'creatinine'],
    'lipid_panel': ['cholesterol', 'hdl', 'ldl', 'triglycerides'],
    'complete_blood_count': ['hemoglobin', 'hematocrit', 'white_blood_cells', 'red_blood_cells', 'platelets'],
    'liver_function': ['alt', 'ast', 'bilirubin', 'albumin', 'alkaline_phosphatase'],
    'thyroid_function': ['tsh', 't3', 't4'],
    'inflammatory_markers': ['esr', 'crp'],
    'vitamins_minerals': ['vitamin_d', 'vitamin_b12', 'folate', 'iron', 'ferritin']
}

# Risk level definitions
RISK_LEVELS = {
    'normal': {'color': 'green', 'priority': 0},
    'borderline': {'color': 'yellow', 'priority': 1},
    'high': {'color': 'orange', 'priority': 2},
    'critical': {'color': 'red', 'priority': 3}
}

# Recommendation types
RECOMMENDATION_TYPES = {
    'immediate': 'Requires immediate medical attention',
    'urgent': 'Should consult doctor within 1-2 days',
    'routine': 'Discuss with doctor at next routine visit',
    'lifestyle': 'Can be addressed with lifestyle changes',
    'monitoring': 'Requires regular monitoring'
}

def get_app_info():
    """Get application information"""
    return {
        'name': APP_NAME,
        'version': APP_VERSION,
        'description': APP_DESCRIPTION
    }

def get_risk_level_info(level):
    """Get risk level information"""
    return RISK_LEVELS.get(level.lower(), {'color': 'gray', 'priority': -1})

def get_blood_category(parameter):
    """Get the category for a blood parameter"""
    for category, parameters in BLOOD_TEST_CATEGORIES.items():
        if parameter.lower() in parameters:
            return category
    return 'other'

def validate_environment():
    """Validate that all required environment variables are set"""
    if settings is None:
        return False, "Settings could not be initialized"
    
    required_vars = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not hasattr(settings, var) or not getattr(settings, var):
            missing_vars.append(var)
    
    if missing_vars:
        return False, f"Missing required environment variables: {', '.join(missing_vars)}"
    
    return True, "All required environment variables are set"
