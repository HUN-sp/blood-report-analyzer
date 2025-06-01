import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import streamlit as st
from config.settings import Settings

@dataclass
class BloodTestResult:
    parameter: str
    value: float
    unit: str
    reference_range: str
    status: str
    
@dataclass
class PatientInfo:
    name: str
    age: int
    sex: str
    patient_id: str
    test_date: str
    lab_name: str

class APIBloodReportProcessor:
    def __init__(self):
        self.settings = Settings()
        self.supported_apis = {
            "pdf_co": self.process_with_pdf_co,
            "parseur": self.process_with_parseur,
            "custom_ocr": self.process_with_custom_ocr
        }
    
    def process_blood_report(self, pdf_file, api_choice: str = "pdf_co") -> Dict:
        """Main method to process blood report using selected API"""
        try:
            # Step 1: Send PDF to API for structured extraction
            raw_data = self.supported_apis[api_choice](pdf_file)
            
            # Step 2: Validate and structure the data
            validated_data = self.validate_extracted_data(raw_data)
            
            # Step 3: Normalize blood values
            normalized_data = self.normalize_blood_values(validated_data)
            
            # Step 4: Apply medical validation rules
            medical_validation = self.apply_medical_validation(normalized_data)
            
            return {
                "success": True,
                "patient_info": normalized_data.get("patient_info", {}),
                "blood_results": normalized_data.get("blood_results", {}),
                "validation_results": medical_validation,
                "raw_api_response": raw_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_data": self.fallback_extraction(pdf_file)
            }
    
    def process_with_pdf_co(self, pdf_file) -> Dict:
        """Process using PDF.co API with blood test template"""
        
        # PDF.co Document Parser API
        api_url = "https://api.pdf.co/v1/pdf/documentparser"
        
        # Blood test template (based on the search results)
        template = {
            "templateName": "BloodTestTemplate",
            "templateVersion": 4,
            "objects": [
                {
                    "name": "PatientName",
                    "objectType": "field",
                    "fieldProperties": {
                        "fieldType": "regex",
                        "expression": r"(?:Name|Patient)[\s:]*([A-Za-z\s\.]+)",
                        "dataType": "string"
                    }
                },
                {
                    "name": "PatientAge", 
                    "objectType": "field",
                    "fieldProperties": {
                        "fieldType": "regex",
                        "expression": r"Age[\s:]*(\d+)",
                        "dataType": "integer"
                    }
                },
                {
                    "name": "PatientSex",
                    "objectType": "field", 
                    "fieldProperties": {
                        "fieldType": "regex",
                        "expression": r"Sex[\s:]*([MF]ale)",
                        "dataType": "string"
                    }
                },
                {
                    "name": "TestResults",
                    "objectType": "table",
                    "tableProperties": {
                        "columns": ["Investigation", "Result", "Reference Value", "Unit"],
                        "detectTableStructure": True
                    }
                }
            ]
        }
        
        # Prepare request
        files = {'file': pdf_file}
        data = {
            'template': json.dumps(template),
            'async': 'false'
        }
        headers = {
            'x-api-key': self.settings.PDF_CO_API_KEY  # Add this to your settings
        }
        
        try:
            response = requests.post(api_url, files=files, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"PDF.co API error: {str(e)}")
    
    def process_with_parseur(self, pdf_file) -> Dict:
        """Process using Parseur API for blood test reports"""
        
        # Parseur API endpoint (you'll need to set up a parser template)
        api_url = f"https://api.parseur.com/parser/{self.settings.PARSEUR_PARSER_ID}/upload"
        
        files = {'file': pdf_file}
        headers = {
            'Authorization': f'Bearer {self.settings.PARSEUR_API_KEY}'
        }
        
        try:
            response = requests.post(api_url, files=files, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Parseur API error: {str(e)}")
    
    def process_with_custom_ocr(self, pdf_file) -> Dict:
        """Process using custom OCR + AI extraction"""
        
        # Use OpenAI Vision API or similar for structured extraction
        prompt = """
        Extract blood test data from this medical report and return as JSON with this exact structure:
        {
            "patient_info": {
                "name": "patient name",
                "age": age_number,
                "sex": "Male/Female", 
                "patient_id": "id",
                "test_date": "date",
                "lab_name": "lab name"
            },
            "blood_results": [
                {
                    "parameter": "parameter_name",
                    "value": numeric_value,
                    "unit": "unit",
                    "reference_range": "range",
                    "status": "Normal/High/Low"
                }
            ]
        }
        
        Extract ALL visible blood test parameters and their values.
        """
        
        # Implementation would use OpenAI Vision API or similar
        # For now, return structured format
        return self.extract_with_ai_vision(pdf_file, prompt)
    
    def validate_extracted_data(self, raw_data: Dict) -> Dict:
        """Validate extracted data using JSON schema"""
        
        # Define JSON schema for blood test data
        blood_test_schema = {
            "type": "object",
            "required": ["patient_info", "blood_results"],
            "properties": {
                "patient_info": {
                    "type": "object",
                    "required": ["name", "age", "sex"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150},
                        "sex": {"type": "string", "enum": ["Male", "Female", "M", "F"]},
                        "patient_id": {"type": "string"},
                        "test_date": {"type": "string"},
                        "lab_name": {"type": "string"}
                    }
                },
                "blood_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["parameter", "value"],
                        "properties": {
                            "parameter": {"type": "string"},
                            "value": {"type": "number"},
                            "unit": {"type": "string"},
                            "reference_range": {"type": "string"},
                            "status": {"type": "string", "enum": ["Normal", "High", "Low", "Critical"]}
                        }
                    }
                }
            }
        }
        
        # Validate using jsonschema library
        try:
            import jsonschema
            jsonschema.validate(raw_data, blood_test_schema)
            return {"valid": True, "data": raw_data, "errors": []}
        except jsonschema.ValidationError as e:
            return {"valid": False, "data": raw_data, "errors": [str(e)]}
        except Exception as e:
            return {"valid": False, "data": raw_data, "errors": [f"Validation error: {str(e)}"]}
    
    def normalize_blood_values(self, validated_data: Dict) -> Dict:
        """Normalize blood values to standard units and formats"""
        
        if not validated_data.get("valid", False):
            return validated_data["data"]
        
        data = validated_data["data"]
        normalized_results = {}
        
        # Unit conversion mappings
        unit_conversions = {
            "hemoglobin": {"g/dl": 1.0, "g/l": 0.1, "mmol/l": 1.61},
            "glucose": {"mg/dl": 1.0, "mmol/l": 18.0},
            "cholesterol": {"mg/dl": 1.0, "mmol/l": 38.67},
            "creatinine": {"mg/dl": 1.0, "μmol/l": 0.0113}
        }
        
        # Parameter name standardization
        parameter_mapping = {
            "hb": "hemoglobin",
            "rbc": "red_blood_cells", 
            "wbc": "white_blood_cells",
            "plt": "platelets",
            "total cholesterol": "cholesterol",
            "blood sugar": "glucose"
        }
        
        for result in data.get("blood_results", []):
            param_name = result["parameter"].lower().strip()
            
            # Standardize parameter name
            standardized_name = parameter_mapping.get(param_name, param_name)
            
            # Convert units if needed
            value = result["value"]
            unit = result.get("unit", "").lower()
            
            if standardized_name in unit_conversions and unit in unit_conversions[standardized_name]:
                conversion_factor = unit_conversions[standardized_name][unit]
                normalized_value = value * conversion_factor
            else:
                normalized_value = value
            
            normalized_results[standardized_name] = {
                "value": normalized_value,
                "original_value": value,
                "unit": result.get("unit", ""),
                "reference_range": result.get("reference_range", ""),
                "status": result.get("status", "")
            }
        
        return {
            "patient_info": data.get("patient_info", {}),
            "blood_results": normalized_results
        }
    
    def apply_medical_validation(self, normalized_data: Dict) -> Dict:
        """Apply medical validation rules to blood values"""
        
        validation_results = {
            "critical_values": [],
            "abnormal_values": [],
            "normal_values": [],
            "missing_critical_tests": [],
            "data_quality_issues": []
        }
        
        # Critical value thresholds
        critical_thresholds = {
            "hemoglobin": {"critical_low": 7.0, "critical_high": 20.0},
            "glucose": {"critical_low": 40.0, "critical_high": 400.0},
            "white_blood_cells": {"critical_low": 2.0, "critical_high": 30.0},
            "platelets": {"critical_low": 50.0, "critical_high": 1000.0}
        }
        
        # Essential tests that should be present
        essential_tests = ["hemoglobin", "white_blood_cells", "platelets"]
        
        blood_results = normalized_data.get("blood_results", {})
        
        # Check for critical values
        for param, data in blood_results.items():
            value = data["value"]
            
            if param in critical_thresholds:
                thresholds = critical_thresholds[param]
                if value <= thresholds["critical_low"] or value >= thresholds["critical_high"]:
                    validation_results["critical_values"].append({
                        "parameter": param,
                        "value": value,
                        "threshold_type": "critical_low" if value <= thresholds["critical_low"] else "critical_high"
                    })
        
        # Check for missing essential tests
        for test in essential_tests:
            if test not in blood_results:
                validation_results["missing_critical_tests"].append(test)
        
        # Data quality checks
        for param, data in blood_results.items():
            if data["value"] <= 0:
                validation_results["data_quality_issues"].append(f"Invalid value for {param}: {data['value']}")
        
        return validation_results
    
    def extract_with_ai_vision(self, pdf_file, prompt: str) -> Dict:
        """Extract data using AI vision models"""
        # This would implement OpenAI Vision API or similar
        # For now, return a placeholder structure
        return {
            "patient_info": {
                "name": "Extracted via AI",
                "age": 0,
                "sex": "Unknown"
            },
            "blood_results": []
        }
    
    def fallback_extraction(self, pdf_file) -> Dict:
        """Fallback extraction method if APIs fail"""
        # Use the original regex-based extraction as fallback
        from utils.pdf_processor import BloodReportProcessor
        
        processor = BloodReportProcessor()
        try:
            text = processor.extract_text_from_pdf(pdf_file)
            blood_data = processor.parse_blood_values(text)
            patient_info = processor.extract_patient_info(text)
            
            return {
                "patient_info": patient_info,
                "blood_results": blood_data,
                "extraction_method": "fallback_regex"
            }
        except Exception as e:
            return {"error": f"Fallback extraction failed: {str(e)}"}
