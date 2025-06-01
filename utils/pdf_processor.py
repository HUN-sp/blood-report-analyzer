import PyPDF2
import re
import base64
from typing import Dict, List, Optional, Tuple
import json

class BloodReportProcessor:
    def __init__(self):
        self.extracted_data = {}
        self.raw_text = ""
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            self.raw_text = text
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def get_pdf_as_base64(self, pdf_file) -> str:
        """Convert PDF to base64 for direct LLM processing"""
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)  # Reset again for other operations
            return base64.b64encode(pdf_bytes).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error converting PDF to base64: {str(e)}")
    
    def extract_blood_values_with_llm(self, text: str, llm_client, model_choice: str) -> Dict:
        """
        Use LLM to extract blood values from text with structured output
        """
        # Simplified and more focused extraction prompt
        extraction_prompt = f"""
        Extract blood test values from this medical report text and return a JSON object:

        MEDICAL REPORT TEXT:
        {text}

        Extract these blood parameters if present:
        - Hemoglobin (Hb)
        - White Blood Cells (WBC) 
        - Red Blood Cells (RBC)
        - Platelets
        - Hematocrit (PCV/HCT)
        - Mean Corpuscular Volume (MCV)
        - Mean Corpuscular Hemoglobin (MCH)
        - MCHC
        - RDW
        - Neutrophils
        - Lymphocytes  
        - Eosinophils
        - Monocytes
        - Basophils

        Also extract:
        - Patient name
        - Patient age
        - Patient gender

        Return ONLY this JSON format:
        {{
            "blood_values": {{
                "hemoglobin": numeric_value_or_null,
                "white_blood_cells": numeric_value_or_null,
                "red_blood_cells": numeric_value_or_null,
                "platelets": numeric_value_or_null,
                "hematocrit": numeric_value_or_null,
                "mcv": numeric_value_or_null,
                "mch": numeric_value_or_null,
                "mchc": numeric_value_or_null,
                "rdw": numeric_value_or_null,
                "neutrophils": numeric_value_or_null,
                "lymphocytes": numeric_value_or_null,
                "eosinophils": numeric_value_or_null,
                "monocytes": numeric_value_or_null,
                "basophils": numeric_value_or_null
            }},
            "patient_info": {{
                "name": "patient_name_or_null",
                "age": numeric_age_or_null,
                "gender": "male_or_female_or_null"
            }},
            "units": {{}},
            "reference_ranges": {{}}
        }}

        IMPORTANT: 
        - Extract only numeric values for blood parameters
        - Use null for missing values
        - Look for values like "12.5", "9000", "150000" etc.
        - Don't include units in the values
        - Return ONLY the JSON, no explanation
        """
        
        try:
            if model_choice == "OpenAI GPT-3.5":
                result = llm_client.extract_structured_data_openai(extraction_prompt)
            elif model_choice == "Google Gemini":
                result = llm_client.extract_structured_data_gemini(extraction_prompt)
            else:
                result = llm_client.extract_structured_data_openai(extraction_prompt)
            
            print(f"LLM Raw Response: {result}")  # Debug output
            
            # Parse JSON response
            if isinstance(result, str):
                try:
                    # Clean the response
                    clean_result = result.strip()
                    
                    # Remove markdown formatting if present
                    if clean_result.startswith('```json'):
                        clean_result = clean_result[7:]
                    if clean_result.startswith('```'):
                        clean_result = clean_result[3:]
                    if clean_result.endswith('```'):
                        clean_result = clean_result[:-3]
                    
                    clean_result = clean_result.strip()
                    
                    # Try to find JSON object in the response
                    json_start = clean_result.find('{')
                    json_end = clean_result.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = clean_result[json_start:json_end]
                        parsed_result = json.loads(json_str)
                        
                        print(f"Successfully parsed JSON: {parsed_result}")  # Debug output
                        return parsed_result
                    else:
                        raise json.JSONDecodeError("No JSON object found", clean_result, 0)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}, trying fallback extraction")
                    # Fallback to regex extraction
                    return self._direct_extraction_fallback(text)
            else:
                return result
                
        except Exception as e:
            print(f"LLM extraction failed completely: {e}")
            # Fallback to direct extraction
            return self._direct_extraction_fallback(text)
    
    def _direct_extraction_fallback(self, text: str) -> Dict:
        """Direct extraction fallback for when LLM fails"""
        print("Using direct extraction fallback...")
        
        result = {
            "blood_values": self.parse_blood_values(text),
            "patient_info": self.extract_patient_info(text),
            "units": {},
            "reference_ranges": {}
        }
        
        print(f"Fallback extraction result: {result}")
        return result
    
    def parse_blood_values(self, text: str) -> Dict:
        """Enhanced regex parsing specifically for your PDF format"""
        patterns = {
            # Basic parameters
            'hemoglobin': [
                r'Hemoglobin\s*\(Hb\)[\s\n]*(\d+\.?\d*)',
                r'Hemoglobin.*?(\d+\.?\d*)',
                r'Hb[\s:]*(\d+\.?\d*)'
            ],
            'white_blood_cells': [
                r'Total WBC count[\s\n]*(\d+)',
                r'WBC.*?(\d+)',
                r'White Blood Cells?.*?(\d+)'
            ],
            'red_blood_cells': [
                r'Total RBC count[\s\n]*(\d+\.?\d*)',
                r'RBC.*?(\d+\.?\d*)',
                r'Red Blood Cells?.*?(\d+\.?\d*)'
            ],
            'platelets': [
                r'Platelet Count[\s\n]*(\d+)',
                r'Platelets?.*?(\d+)',
                r'PLT.*?(\d+)'
            ],
            # Blood indices
            'hematocrit': [
                r'Packed Cell Volume\s*\(PCV\)[\s\n]*(\d+\.?\d*)',
                r'PCV.*?(\d+\.?\d*)',
                r'Hematocrit.*?(\d+\.?\d*)'
            ],
            'mcv': [
                r'Mean Corpuscular Volume\s*\(MCV\)[\s\n]*(\d+\.?\d*)',
                r'MCV.*?(\d+\.?\d*)'
            ],
            'mch': [
                r'MCH[\s\n]*(\d+\.?\d*)',
                r'Mean Corpuscular Hemoglobin[\s\n]*(\d+\.?\d*)'
            ],
            'mchc': [
                r'MCHC[\s\n]*(\d+\.?\d*)',
                r'Mean Corpuscular Hemoglobin Concentration[\s\n]*(\d+\.?\d*)'
            ],
            'rdw': [
                r'RDW[\s\n]*(\d+\.?\d*)',
                r'Red Cell Distribution Width[\s\n]*(\d+\.?\d*)'
            ],
            # Differential count
            'neutrophils': [
                r'Neutrophils[\s\n]*(\d+)',
                r'Neutrophil.*?(\d+)'
            ],
            'lymphocytes': [
                r'Lymphocytes[\s\n]*(\d+)',
                r'Lymphocyte.*?(\d+)'
            ],
            'eosinophils': [
                r'Eosinophils[\s\n]*(\d+)',
                r'Eosinophil.*?(\d+)'
            ],
            'monocytes': [
                r'Monocytes[\s\n]*(\d+)',
                r'Monocyte.*?(\d+)'
            ],
            'basophils': [
                r'Basophils[\s\n]*(\d+)',
                r'Basophil.*?(\d+)'
            ]
        }
        
        results = {}
        
        # Print text for debugging
        print(f"Text being parsed: {text[:500]}...")
        
        for parameter, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        value = float(match.group(1))
                        results[parameter] = value
                        print(f"Found {parameter}: {value}")
                        break  # Found a match, move to next parameter
                    except (ValueError, IndexError):
                        continue
        
        print(f"Final parsed results: {results}")
        return results
    
    def extract_patient_info(self, text: str) -> Dict:
        """Enhanced patient information extraction"""
        info = {}
        text_lower = text.lower()
        
        # Extract age with multiple patterns
        age_patterns = [
            r'age[\s:]*(\d+)',
            r'(\d+)[\s]*years?\s*old',
            r'(\d+)\s*yrs?',
            r'age\s*[-:]\s*(\d+)'
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, text_lower)
            if age_match:
                try:
                    info['age'] = int(age_match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract gender with better patterns
        if re.search(r'\b(?:male|m)\b(?!\w)', text_lower):
            info['gender'] = 'male'
        elif re.search(r'\b(?:female|f)\b(?!\w)', text_lower):
            info['gender'] = 'female'
        
        # Extract name with multiple patterns
        name_patterns = [
            r'(?:name|patient name|patient)[\s:]*([a-zA-Z\s]{2,30})',
            r'mr\.?\s+([a-zA-Z\s]{2,30})',
            r'mrs?\.?\s+([a-zA-Z\s]{2,30})',
            r'patient:\s*([a-zA-Z\s]{2,30})'
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                if len(name) > 2 and not any(char.isdigit() for char in name):
                    info['name'] = name
                    break
        
        # Extract test date
        date_patterns = [
            r'date[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'collected on[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                info['test_date'] = date_match.group(1)
                break
        
        return info
    
    def validate_extracted_data(self, data: Dict) -> Tuple[Dict, List[str]]:
        """Validate extracted blood values and return warnings"""
        warnings = []
        validated_data = {}
        
        # Define reasonable ranges for validation
        reasonable_ranges = {
            'hemoglobin': (5, 25),
            'white_blood_cells': (1, 50),
            'red_blood_cells': (2, 8),
            'platelets': (50, 1000),
            'glucose': (30, 500),
            'cholesterol': (50, 500),
            'hdl': (10, 150),
            'ldl': (10, 300),
            'triglycerides': (20, 1000),
            'creatinine': (0.5, 15),
            'urea': (5, 150),
            'hba1c': (3, 20),
            'tsh': (0.1, 50),
            'vitamin_d': (5, 200),
            'vitamin_b12': (100, 2000)
        }
        
        for param, value in data.items():
            if value is not None and param in reasonable_ranges:
                min_val, max_val = reasonable_ranges[param]
                if min_val <= value <= max_val:
                    validated_data[param] = value
                else:
                    warnings.append(f"Unusual value for {param}: {value} (expected range: {min_val}-{max_val})")
            elif value is not None:
                validated_data[param] = value
        
        return validated_data, warnings
    
    def get_comprehensive_extraction(self, pdf_file, llm_client, model_choice: str) -> Dict:
        """
        Main method to extract all information from PDF using LLM
        """
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "No readable text found in PDF",
                    "blood_values": {},
                    "patient_info": {},
                    "warnings": ["PDF appears to be empty or contains only images"]
                }
            
            # Use LLM for structured extraction
            extraction_result = self.extract_blood_values_with_llm(text, llm_client, model_choice)
            
            # Extract components
            blood_values = extraction_result.get("blood_values", {})
            patient_info = extraction_result.get("patient_info", {})
            units = extraction_result.get("units", {})
            reference_ranges = extraction_result.get("reference_ranges", {})
            
            # Validate extracted data
            validated_blood_values, warnings = self.validate_extracted_data(blood_values)
            
            # Check if we found any blood values
            if not validated_blood_values:
                warnings.append("No blood values could be extracted from the PDF")
                warnings.append("The PDF might not contain standard blood test results")
            
            return {
                "success": len(validated_blood_values) > 0,
                "blood_values": validated_blood_values,
                "patient_info": patient_info,
                "units": units,
                "reference_ranges": reference_ranges,
                "warnings": warnings,
                "raw_text_preview": text[:500] + "..." if len(text) > 500 else text,
                "extraction_method": f"LLM-powered extraction using {model_choice}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "blood_values": {},
                "patient_info": {},
                "warnings": [f"Extraction failed: {str(e)}"]
            }