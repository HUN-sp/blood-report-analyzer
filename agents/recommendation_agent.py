from crewai import Agent, Task, Crew
from utils.llm_clients import LLMClients
from config.settings import Settings
from typing import Dict, List, Optional

class RecommendationAgent:
    def __init__(self):
        self.llm_clients = LLMClients()
        self.settings = Settings()
        self.setup_agents()
    
    def setup_agents(self):
        """Setup specialized recommendation agents"""
        
        self.medication_advisor = Agent(
            role='Medication Advisor',
            goal='Provide evidence-based medication recommendations for blood abnormalities',
            backstory='''You are a clinical pharmacist with expertise in interpreting blood tests 
            and recommending appropriate medications. You provide safe, evidence-based suggestions 
            while emphasizing the need for professional medical consultation.''',
            verbose=True
        )
        
        self.lifestyle_coach = Agent(
            role='Lifestyle and Nutrition Coach',
            goal='Recommend lifestyle changes, diet modifications, and natural remedies',
            backstory='''You are a certified nutritionist and lifestyle coach specializing in 
            health optimization through natural means. You focus on diet, exercise, sleep, 
            and stress management to improve blood markers.''',
            verbose=True
        )
        
        self.natural_remedy_specialist = Agent(
            role='Natural Remedy Specialist',
            goal='Suggest natural supplements and holistic approaches for health improvement',
            backstory='''You are a naturopathic doctor with extensive knowledge of herbs, 
            supplements, and natural healing methods. You provide evidence-based natural 
            solutions while respecting conventional medicine.''',
            verbose=True
        )
        
        self.follow_up_coordinator = Agent(
            role='Medical Follow-up Coordinator',
            goal='Determine appropriate medical follow-up and monitoring requirements',
            backstory='''You are a healthcare coordinator who specializes in determining 
            when patients need immediate attention, routine follow-ups, or specialist referrals 
            based on blood test results.''',
            verbose=True
        )
    
    def generate_comprehensive_recommendations(self, blood_data: Dict, analysis: str, patient_info: Dict) -> Dict:
        """Generate comprehensive recommendations using multiple agents"""
        
        # Task 1: Medication recommendations
        medication_task = Task(
            description=f"""
            Based on blood test results: {blood_data}
            Patient info: {patient_info}
            Analysis: {analysis}
            
            Provide medication recommendations for any abnormal values. Include:
            1. Specific medications that might be considered
            2. Dosage considerations
            3. Potential side effects
            4. Drug interactions to watch for
            5. Strong disclaimer about consulting healthcare professionals
            
            Focus on commonly prescribed medications for blood abnormalities.
            """,
            agent=self.medication_advisor,
            expected_output="Detailed medication recommendations with dosage guidelines, side effects, drug interactions, and strong medical consultation disclaimers for each abnormal blood parameter."
        )
        
        # Task 2: Lifestyle and diet recommendations
        lifestyle_task = Task(
            description=f"""
            Based on blood results: {blood_data}
            Patient profile: {patient_info}
            
            Provide comprehensive lifestyle recommendations:
            1. Specific dietary changes for each abnormal parameter
            2. Exercise recommendations
            3. Sleep optimization
            4. Stress management techniques
            5. Foods to avoid and foods to include
            6. Meal timing and preparation tips
            """,
            agent=self.lifestyle_coach,
            expected_output="Comprehensive lifestyle and dietary recommendations including specific food choices, exercise routines, sleep optimization strategies, and stress management techniques tailored to the blood test results."
        )
        
        # Task 3: Natural remedies
        natural_remedies_task = Task(
            description=f"""
            For blood test abnormalities in: {blood_data}
            
            Suggest natural remedies and supplements:
            1. Herbal remedies with scientific backing
            2. Vitamin and mineral supplements
            3. Dosage recommendations
            4. Safety considerations and contraindications
            5. Expected timeline for improvements
            6. Natural alternatives to conventional treatments
            
            Emphasize evidence-based natural approaches.
            """,
            agent=self.natural_remedy_specialist,
            expected_output="Evidence-based natural remedies and supplements recommendations including herbal treatments, vitamins, minerals, dosage guidelines, safety considerations, and expected timelines for improvement."
        )
        
        # Task 4: Follow-up recommendations
        followup_task = Task(
            description=f"""
            Based on blood abnormalities: {blood_data}
            Analysis: {analysis}
            
            Determine appropriate follow-up:
            1. Urgency level (immediate, routine, long-term)
            2. Recommended timeline for re-testing
            3. Specialist referrals needed
            4. Warning signs to watch for
            5. When to seek emergency care
            6. Monitoring parameters
            """,
            agent=self.follow_up_coordinator,
            expected_output="Complete medical follow-up plan including urgency assessment, re-testing timeline, specialist referral recommendations, warning signs to monitor, and emergency care indicators."
        )
        
        try:
            # Create crew and execute tasks
            recommendation_crew = Crew(
                agents=[
                    self.medication_advisor,
                    self.lifestyle_coach,
                    self.natural_remedy_specialist,
                    self.follow_up_coordinator
                ],
                tasks=[medication_task, lifestyle_task, natural_remedies_task, followup_task],
                verbose=True
            )
            
            # Execute the crew
            results = recommendation_crew.kickoff()
            
            # Parse and structure the results
            return self.structure_recommendations(results, blood_data)
            
        except Exception as e:
            print(f"CrewAI recommendation failed: {e}")
            # Fallback to simple recommendations
            return self._fallback_recommendations(blood_data, analysis, patient_info)
    
    def _fallback_recommendations(self, blood_data: Dict, analysis: str, patient_info: Dict) -> Dict:
        """Fallback recommendations when CrewAI fails"""
        try:
            # Use LLM clients for basic recommendations
            recommendations = self.llm_clients.get_comprehensive_recommendations(
                analysis, blood_data, patient_info
            )
            
            return {
                "structured_recommendations": recommendations,
                "raw_crew_output": "CrewAI failed, using LLM fallback",
                "personalized_tips": self.generate_personalized_tips(blood_data),
                "emergency_indicators": self.check_emergency_indicators(blood_data),
                "method": "LLM Fallback"
            }
        except Exception as e:
            return {
                "structured_recommendations": f"Error generating recommendations: {str(e)}",
                "raw_crew_output": "Both CrewAI and LLM failed",
                "personalized_tips": self.generate_personalized_tips(blood_data),
                "emergency_indicators": self.check_emergency_indicators(blood_data),
                "method": "Basic Fallback"
            }
    
    def structure_recommendations(self, crew_results: str, blood_data: Dict) -> Dict:
        """Structure the crew results into organized recommendations"""
        
        # Use LLM to structure the recommendations
        structuring_prompt = f"""
        Please structure these recommendation results into a clear, organized format:
        
        Crew Results: {crew_results}
        Blood Data: {blood_data}
        
        Structure as:
        1. IMMEDIATE ACTIONS (if any critical values)
        2. MEDICATION RECOMMENDATIONS
        3. LIFESTYLE CHANGES
        4. NATURAL REMEDIES
        5. FOLLOW-UP PLAN
        6. MEDICAL DISCLAIMERS
        
        Make it clear, actionable, and well-organized.
        """
        
        try:
            structured_recommendations = self.llm_clients.get_comprehensive_recommendations(
                structuring_prompt, blood_data, {}
            )
            
            return {
                "structured_recommendations": structured_recommendations,
                "raw_crew_output": str(crew_results),
                "personalized_tips": self.generate_personalized_tips(blood_data),
                "emergency_indicators": self.check_emergency_indicators(blood_data),
                "method": "CrewAI Success"
            }
        except Exception as e:
            return {
                "structured_recommendations": str(crew_results),
                "error": f"Structuring failed: {str(e)}",
                "personalized_tips": self.generate_personalized_tips(blood_data),
                "emergency_indicators": self.check_emergency_indicators(blood_data),
                "method": "CrewAI Partial"
            }
    
    def generate_personalized_tips(self, blood_data: Dict) -> List[str]:
        """Generate personalized tips based on specific blood values"""
        tips = []
        
        # Hemoglobin recommendations
        if "hemoglobin" in blood_data:
            hb_value = blood_data["hemoglobin"]
            if hb_value < 12:
                tips.append("🩸 Low Hemoglobin: Include iron-rich foods like spinach, red meat, and lentils")
                tips.append("🍊 Enhance iron absorption with vitamin C-rich foods (citrus fruits, bell peppers)")
            elif hb_value > 17:
                tips.append("⚠️ High Hemoglobin: Stay well-hydrated and consider reducing iron supplements")
        
        # Glucose recommendations (if present)
        if "glucose" in blood_data:
            glucose_value = blood_data["glucose"]
            if glucose_value > 100:
                tips.append("🍎 High Glucose: Focus on low-glycemic foods and regular physical activity")
                tips.append("⏰ Consider intermittent fasting and smaller, frequent meals")
            elif glucose_value < 70:
                tips.append("🍌 Low Glucose: Eat regular meals and include complex carbohydrates")
        
        # Cholesterol recommendations (if present)
        if "cholesterol" in blood_data:
            chol_value = blood_data["cholesterol"]
            if chol_value > 200:
                tips.append("🥑 High Cholesterol: Increase omega-3 fatty acids and soluble fiber")
                tips.append("🚶‍♂️ Add 30 minutes of daily cardio exercise")
        
        # White blood cell recommendations
        if "white_blood_cells" in blood_data:
            wbc_value = blood_data["white_blood_cells"]
            if wbc_value < 4500:
                tips.append("🛡️ Low WBC: Boost immunity with adequate sleep and stress management")
                tips.append("🧄 Include immune-supporting foods like garlic, ginger, and turmeric")
            elif wbc_value > 11000:
                tips.append("🔥 High WBC: May indicate infection - ensure adequate rest and hydration")
        
        # Platelet recommendations
        if "platelets" in blood_data:
            plt_value = blood_data["platelets"]
            if plt_value < 150000:
                tips.append("🩸 Low Platelets: Avoid excessive alcohol and get adequate sleep")
                tips.append("🥬 Include leafy greens and foods rich in folate and B12")
            elif plt_value == 150000:  # Borderline as in your PDF
                tips.append("⚠️ Borderline Platelets: Monitor closely and maintain healthy lifestyle")
        
        # Hematocrit/PCV recommendations
        if "hematocrit" in blood_data:
            hct_value = blood_data["hematocrit"]
            if hct_value > 50:  # High as in your PDF (57.5%)
                tips.append("🚰 High Hematocrit: Increase fluid intake and avoid dehydration")
                tips.append("🚭 Avoid smoking and reduce factors that thicken blood")
        
        return tips
    
    def check_emergency_indicators(self, blood_data: Dict) -> Dict:
        """Check for values that might need immediate medical attention"""
        emergency_indicators = {
            "critical_values": [],
            "urgent_consultation": [],
            "emergency_level": "normal"
        }
        
        # Use settings to check critical values
        for parameter, value in blood_data.items():
            critical_check = self.settings.is_critical_value(parameter, value)
            if critical_check['is_critical']:
                emergency_indicators["critical_values"].append({
                    "parameter": parameter,
                    "value": value,
                    "status": critical_check['level'],
                    "message": critical_check['message']
                })
                emergency_indicators["emergency_level"] = "critical"
        
        # Check for concerning patterns from your PDF
        if "hemoglobin" in blood_data and blood_data["hemoglobin"] < 13.0:
            emergency_indicators["urgent_consultation"].append({
                "parameter": "hemoglobin",
                "value": blood_data["hemoglobin"],
                "message": "Low hemoglobin suggests anemia - requires medical evaluation"
            })
            if emergency_indicators["emergency_level"] == "normal":
                emergency_indicators["emergency_level"] = "urgent"
        
        if "platelets" in blood_data and blood_data["platelets"] <= 150000:
            emergency_indicators["urgent_consultation"].append({
                "parameter": "platelets",
                "value": blood_data["platelets"],
                "message": "Low/borderline platelets - requires monitoring and medical consultation"
            })
            if emergency_indicators["emergency_level"] == "normal":
                emergency_indicators["emergency_level"] = "urgent"
        
        if "hematocrit" in blood_data and blood_data["hematocrit"] > 55:
            emergency_indicators["urgent_consultation"].append({
                "parameter": "hematocrit",
                "value": blood_data["hematocrit"],
                "message": "High hematocrit - may indicate polycythemia or dehydration"
            })
            if emergency_indicators["emergency_level"] == "normal":
                emergency_indicators["emergency_level"] = "urgent"
        
        return emergency_indicators
    
    def get_medicine_recommendations(self, parameter: str, value: float, patient_info: Dict) -> Dict:
        """Get specific medicine recommendations for abnormal values"""
        
        medicine_database = {
            "hemoglobin": {
                "low": {
                    "medications": ["Iron sulfate", "Iron fumarate", "Vitamin B12 (if deficient)"],
                    "natural_alternatives": ["Iron-rich foods", "Vitamin C supplements", "Folate"],
                    "lifestyle": ["Iron-rich diet", "Avoid tea with meals", "Cook in iron pans"]
                }
            },
            "glucose": {
                "high": {
                    "medications": ["Metformin", "Glipizide", "Insulin (if severe)"],
                    "natural_alternatives": ["Cinnamon supplement", "Chromium", "Alpha-lipoic acid"],
                    "lifestyle": ["Low-carb diet", "Regular exercise", "Weight management"]
                }
            },
            "cholesterol": {
                "high": {
                    "medications": ["Atorvastatin", "Simvastatin", "Rosuvastatin"],
                    "natural_alternatives": ["Red yeast rice", "Plant sterols", "Omega-3 fish oil"],
                    "lifestyle": ["Mediterranean diet", "Increase fiber", "Regular cardio"]
                }
            }
        }
        
        # Determine if value is high or low based on normal ranges
        try:
            normal_range = self.settings.get_normal_range(parameter, patient_info.get('gender'))
            if isinstance(normal_range, tuple):
                if value < normal_range[0]:
                    status = "low"
                elif value > normal_range[1]:
                    status = "high"
                else:
                    status = "normal"
            else:
                status = "normal"
            
            if parameter in medicine_database and status in medicine_database[parameter]:
                return medicine_database[parameter][status]
            else:
                return {"message": "No specific recommendations available for this parameter"}
                
        except Exception as e:
            return {"error": f"Could not generate recommendations: {str(e)}"}

# Create a global instance
recommendation_agent = RecommendationAgent()