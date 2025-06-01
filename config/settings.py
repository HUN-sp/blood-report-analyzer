import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Enhanced blood test normal ranges
    NORMAL_RANGES = {
        # Basic Blood Count
        "hemoglobin": {"male": (13.0, 17.0), "female": (12.0, 15.5)},  # g/dL
        "white_blood_cells": (4000, 11000),  # /µL
        "red_blood_cells": {"male": (4.5, 5.5), "female": (4.0, 5.0)},  # mill/cumm
        "platelets": (150000, 410000),  # /µL
        "hematocrit": {"male": (40, 50), "female": (36, 46)},  # %
        
        # Blood Indices
        "mcv": (83, 101),  # fL
        "mch": (27, 32),   # pg
        "mchc": (32.5, 34.5),  # g/dL
        "rdw": (11.6, 14.0),   # %
        
        # Differential Count (percentages)
        "neutrophils": (50, 62),  # %
        "lymphocytes": (20, 40),  # %
        "eosinophils": (0, 6),    # %
        "monocytes": (0, 10),     # %
        "basophils": (0, 2),      # %
        
        # Metabolic Panel
        "glucose": (70, 100),  # mg/dL (fasting)
        "creatinine": {"male": (0.7, 1.3), "female": (0.6, 1.1)},  # mg/dL
        "urea": (7, 20),  # mg/dL
        "bilirubin": (0.3, 1.2),  # mg/dL
        
        # Lipid Panel
        "cholesterol": (0, 200),  # mg/dL (desirable)
        "hdl": {"male": (40, float('inf')), "female": (50, float('inf'))},  # mg/dL
        "ldl": (0, 100),  # mg/dL (optimal)
        "triglycerides": (0, 150),  # mg/dL (normal)
        
        # Liver Function
        "alt": {"male": (10, 40), "female": (7, 35)},  # U/L
        "ast": {"male": (10, 40), "female": (9, 32)},  # U/L
        "alkaline_phosphatase": (44, 147),  # U/L
        
        # Diabetes Markers
        "hba1c": (4.0, 5.6),  # % (normal)
        "fasting_insulin": (2.6, 24.9),  # µU/mL
        
        # Thyroid Function
        "tsh": (0.27, 4.20),  # µIU/mL
        "t3": (80, 200),  # ng/dL
        "t4": (5.1, 14.1),  # µg/dL
        
        # Vitamins and Minerals
        "vitamin_d": (20, 50),  # ng/mL
        "vitamin_b12": (160, 950),  # pg/mL
        "folate": (2.7, 17.0),  # ng/mL
        "iron": {"male": (65, 176), "female": (50, 170)},  # µg/dL
        "ferritin": {"male": (12, 300), "female": (12, 150)},  # ng/mL
        
        # Cardiac Markers
        "troponin": (0, 0.04),  # ng/mL
        "ck_mb": (0, 6.3),  # ng/mL
        
        # Inflammatory Markers
        "esr": {"male": (0, 22), "female": (0, 29)},  # mm/hr
        "crp": (0, 3.0),  # mg/L
        
        # Electrolytes
        "sodium": (136, 145),  # mmol/L
        "potassium": (3.5, 5.1),  # mmol/L
        "chloride": (98, 107),  # mmol/L
        "calcium": (8.5, 10.2),  # mg/dL
        "magnesium": (1.7, 2.2),  # mg/dL
        
        # Protein Markers
        "total_protein": (6.0, 8.3),  # g/dL
        "albumin": (3.5, 5.0),  # g/dL
        "globulin": (2.3, 3.4),  # g/dL
    }
    
    # Unit conversions
    UNIT_CONVERSIONS = {
        "glucose": {
            "mg_dl_to_mmol_l": 0.0555,
            "mmol_l_to_mg_dl": 18.0182
        },
        "cholesterol": {
            "mg_dl_to_mmol_l": 0.0259,
            "mmol_l_to_mg_dl": 38.67
        },
        "creatinine": {
            "mg_dl_to_umol_l": 88.4,
            "umol_l_to_mg_dl": 0.0113
        }
    }
    
    # Critical value thresholds that require immediate attention
    CRITICAL_VALUES = {
        "hemoglobin": {"critically_low": 7, "critically_high": 20},
        "white_blood_cells": {"critically_low": 2000, "critically_high": 20000},
        "platelets": {"critically_low": 50000, "critically_high": 1000000},
        "hematocrit": {"critically_low": 20, "critically_high": 60},
        "glucose": {"critically_low": 50, "critically_high": 400},
        "creatinine": {"critically_high": 4.0},
        "potassium": {"critically_low": 2.5, "critically_high": 6.0},
        "sodium": {"critically_low": 125, "critically_high": 160},
        "troponin": {"critically_high": 0.04}
    }
    
    def get_normal_range(self, parameter: str, gender: str = None):
        """Get normal range for a parameter, considering gender if applicable"""
        if parameter not in self.NORMAL_RANGES:
            raise ValueError(f"No normal range defined for {parameter}")
        
        range_value = self.NORMAL_RANGES[parameter]
        
        # If range is gender-specific
        if isinstance(range_value, dict):
            if gender and gender.lower() in range_value:
                return range_value[gender.lower()]
            else:
                # Return a combined range if gender not specified
                male_range = range_value.get("male", (0, 0))
                female_range = range_value.get("female", (0, 0))
                return (min(male_range[0], female_range[0]), 
                       max(male_range[1], female_range[1]))
        else:
            return range_value
    
    def is_critical_value(self, parameter: str, value: float) -> dict:
        """Check if a value is in critical range"""
        if parameter not in self.CRITICAL_VALUES:
            return {"is_critical": False}
        
        thresholds = self.CRITICAL_VALUES[parameter]
        
        result = {
            "is_critical": False,
            "level": "normal",
            "message": ""
        }
        
        if "critically_low" in thresholds and value < thresholds["critically_low"]:
            result = {
                "is_critical": True,
                "level": "critically_low",
                "message": f"Critically low {parameter}: {value}"
            }
        elif "critically_high" in thresholds and value > thresholds["critically_high"]:
            result = {
                "is_critical": True,
                "level": "critically_high", 
                "message": f"Critically high {parameter}: {value}"
            }
        
        return result
    
    def get_parameter_info(self, parameter: str) -> dict:
        """Get comprehensive information about a blood parameter"""
        parameter_info = {
            "hemoglobin": {
                "full_name": "Hemoglobin",
                "abbreviation": "Hb",
                "unit": "g/dL",
                "description": "Protein in red blood cells that carries oxygen",
                "low_causes": ["Iron deficiency", "Chronic disease", "Blood loss"],
                "high_causes": ["Dehydration", "Smoking", "Living at high altitude"]
            },
            "white_blood_cells": {
                "full_name": "White Blood Cells",
                "abbreviation": "WBC",
                "unit": "/µL",
                "description": "Cells that fight infection and disease",
                "low_causes": ["Viral infections", "Autoimmune disorders", "Medications"],
                "high_causes": ["Bacterial infections", "Stress", "Inflammatory conditions"]
            },
            "red_blood_cells": {
                "full_name": "Red Blood Cells",
                "abbreviation": "RBC",
                "unit": "mill/cumm",
                "description": "Cells that carry oxygen throughout the body",
                "low_causes": ["Anemia", "Blood loss", "Nutritional deficiency"],
                "high_causes": ["Dehydration", "Smoking", "High altitude"]
            },
            "platelets": {
                "full_name": "Platelets",
                "abbreviation": "PLT",
                "unit": "/µL",
                "description": "Blood cells that help with clotting",
                "low_causes": ["Bone marrow disorders", "Medications", "Autoimmune conditions"],
                "high_causes": ["Inflammation", "Cancer", "Blood disorders"]
            },
            "hematocrit": {
                "full_name": "Hematocrit/Packed Cell Volume",
                "abbreviation": "HCT/PCV",
                "unit": "%",
                "description": "Percentage of blood volume made up of red blood cells",
                "low_causes": ["Anemia", "Blood loss", "Overhydration"],
                "high_causes": ["Dehydration", "Polycythemia", "Smoking"]
            },
            "mcv": {
                "full_name": "Mean Corpuscular Volume",
                "abbreviation": "MCV",
                "unit": "fL",
                "description": "Average size of red blood cells",
                "low_causes": ["Iron deficiency", "Thalassemia"],
                "high_causes": ["Vitamin B12/folate deficiency", "Alcohol use"]
            },
            "mch": {
                "full_name": "Mean Corpuscular Hemoglobin",
                "abbreviation": "MCH",
                "unit": "pg",
                "description": "Average amount of hemoglobin in each red blood cell",
                "low_causes": ["Iron deficiency", "Thalassemia"],
                "high_causes": ["Vitamin B12/folate deficiency", "Liver disease"]
            },
            "mchc": {
                "full_name": "Mean Corpuscular Hemoglobin Concentration",
                "abbreviation": "MCHC",
                "unit": "g/dL",
                "description": "Concentration of hemoglobin in red blood cells",
                "low_causes": ["Iron deficiency", "Chronic disease"],
                "high_causes": ["Hereditary spherocytosis", "Dehydration"]
            },
            "rdw": {
                "full_name": "Red Cell Distribution Width",
                "abbreviation": "RDW",
                "unit": "%",
                "description": "Variation in size of red blood cells",
                "low_causes": ["Usually normal"],
                "high_causes": ["Iron deficiency", "Vitamin deficiencies", "Mixed anemias"]
            },
            "neutrophils": {
                "full_name": "Neutrophils",
                "abbreviation": "NEUT",
                "unit": "%",
                "description": "Most common type of white blood cell, fights bacterial infections",
                "low_causes": ["Viral infections", "Chemotherapy", "Autoimmune disorders"],
                "high_causes": ["Bacterial infections", "Stress", "Inflammation"]
            },
            "lymphocytes": {
                "full_name": "Lymphocytes",
                "abbreviation": "LYMPH",
                "unit": "%",
                "description": "White blood cells that fight viral infections and make antibodies",
                "low_causes": ["Immunodeficiency", "Stress", "Steroids"],
                "high_causes": ["Viral infections", "Leukemia", "Lymphoma"]
            },
            "eosinophils": {
                "full_name": "Eosinophils",
                "abbreviation": "EOS",
                "unit": "%",
                "description": "White blood cells that fight parasites and allergic reactions",
                "low_causes": ["Usually normal when low"],
                "high_causes": ["Allergies", "Parasitic infections", "Asthma"]
            },
            "monocytes": {
                "full_name": "Monocytes",
                "abbreviation": "MONO",
                "unit": "%",
                "description": "White blood cells that become macrophages and fight infections",
                "low_causes": ["Usually normal when low"],
                "high_causes": ["Chronic infections", "Autoimmune disorders", "Blood cancers"]
            },
            "basophils": {
                "full_name": "Basophils",
                "abbreviation": "BASO",
                "unit": "%",
                "description": "White blood cells involved in allergic reactions",
                "low_causes": ["Usually normal when low"],
                "high_causes": ["Allergic reactions", "Blood disorders", "Infections"]
            },
            "glucose": {
                "full_name": "Blood Glucose",
                "abbreviation": "GLU",
                "unit": "mg/dL",
                "description": "Amount of sugar in blood",
                "low_causes": ["Medication side effects", "Excessive exercise", "Poor nutrition"],
                "high_causes": ["Diabetes", "Stress", "Certain medications"]
            },
            "cholesterol": {
                "full_name": "Total Cholesterol",
                "abbreviation": "CHOL",
                "unit": "mg/dL",
                "description": "Total amount of cholesterol in blood",
                "low_causes": ["Malnutrition", "Liver disease", "Hyperthyroidism"],
                "high_causes": ["Poor diet", "Genetics", "Sedentary lifestyle"]
            }
        }
        
        return parameter_info.get(parameter, {
            "full_name": parameter.replace('_', ' ').title(),
            "abbreviation": parameter.upper(),
            "unit": "Various",
            "description": "Blood parameter",
            "low_causes": ["Various medical conditions"],
            "high_causes": ["Various medical conditions"]
        })