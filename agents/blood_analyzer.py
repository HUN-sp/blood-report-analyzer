from crewai import Agent, Task, Crew
from utils.llm_clients import LLMClients
from config.settings import Settings

class BloodAnalyzerCrew:
    def __init__(self):
        self.llm_clients = LLMClients()
        self.settings = Settings()
        self.setup_agents()
    
    def setup_agents(self):
        """Setup enhanced CrewAI agents with better prompts"""
        
        self.blood_analyst = Agent(
            role='Senior Blood Test Analyst',
            goal='Provide comprehensive analysis of blood test results with clinical insights',
            backstory='''You are a senior medical laboratory scientist with 15+ years of experience 
            in blood test interpretation. You specialize in identifying patterns, correlations 
            between parameters, and providing detailed clinical insights. You understand the 
            clinical significance of each parameter and how they relate to overall health.''',
            verbose=True,
            allow_delegation=False
        )
        
        self.health_advisor = Agent(
            role='Clinical Health Advisor',
            goal='Provide evidence-based health recommendations and lifestyle modifications',
            backstory='''You are a clinical health advisor with expertise in preventive medicine 
            and lifestyle interventions. You excel at translating complex medical findings into 
            actionable health recommendations while considering patient safety and evidence-based practices.''',
            verbose=True,
            allow_delegation=False
        )
        
        self.report_generator = Agent(
            role='Medical Report Specialist',
            goal='Generate clear, comprehensive, and patient-friendly medical reports',
            backstory='''You are a medical communication specialist who creates detailed yet 
            understandable health reports. You excel at organizing complex medical information 
            into structured, actionable formats that both patients and healthcare providers can easily understand.''',
            verbose=True,
            allow_delegation=False
        )
    
    def analyze_blood_report(self, blood_data: dict, patient_info: dict) -> str:
        """Run comprehensive blood analysis workflow with enhanced structure"""
        
        # Enhanced Task 1: Detailed Blood Analysis
        analysis_task = Task(
            description=f"""
            Perform a comprehensive analysis of the following blood test results:
            
            **PATIENT PROFILE:**
            {self._format_patient_info(patient_info)}
            
            **BLOOD TEST RESULTS:**
            {self._format_blood_data_with_ranges(blood_data)}
            
            **ANALYSIS REQUIREMENTS:**
            
            1. **PARAMETER-BY-PARAMETER ANALYSIS:**
               - Analyze each blood parameter individually
               - Compare values against normal ranges (consider age/gender if provided)
               - Identify parameters that are outside normal ranges
               - Explain the clinical significance of abnormal values
            
            2. **PATTERN RECOGNITION:**
               - Look for patterns and correlations between different parameters
               - Identify any parameter combinations that suggest specific conditions
               - Note any conflicting or unusual parameter relationships
            
            3. **CLINICAL SIGNIFICANCE:**
               - Assess overall health status based on the complete panel
               - Identify potential health risks or conditions suggested by the results
               - Determine urgency level (normal, monitor, concerning, urgent)
            
            4. **POSITIVE FINDINGS:**
               - Highlight parameters within healthy ranges
               - Identify markers of good health
            
            **OUTPUT FORMAT:**
            Provide a structured analysis with clear sections and actionable insights.
            """,
            agent=self.blood_analyst,
            expected_output="A comprehensive blood test analysis report with clinical insights, risk assessment, and detailed interpretation of each parameter including normal and abnormal values."
        )
        
        # Enhanced Task 2: Health Recommendations
        recommendation_task = Task(
            description=f"""
            Based on the blood analysis results, provide comprehensive health recommendations:
            
            **BLOOD DATA FOR REFERENCE:**
            {self._format_blood_data_with_ranges(blood_data)}
            
            **PATIENT CONTEXT:**
            {self._format_patient_info(patient_info)}
            
            **RECOMMENDATION CATEGORIES:**
            
            1. **IMMEDIATE ACTIONS:**
               - Any urgent medical consultations needed
               - Parameters requiring immediate attention
               - Warning signs to watch for
            
            2. **DIETARY RECOMMENDATIONS:**
               - Specific foods to include for each abnormal parameter
               - Foods to avoid or limit
               - Nutritional supplements to consider
               - Meal timing and preparation tips
            
            3. **LIFESTYLE MODIFICATIONS:**
               - Exercise recommendations tailored to findings
               - Sleep optimization strategies
               - Stress management techniques
               - Hydration and general wellness tips
            
            4. **MONITORING PLAN:**
               - Recommended timeline for re-testing
               - Parameters to monitor closely
               - Frequency of follow-up tests
            
            5. **MEDICAL FOLLOW-UP:**
               - When to consult healthcare providers
               - Specialist referrals if needed
               - Questions to ask healthcare providers
            
            **SAFETY REQUIREMENTS:**
            - Include appropriate medical disclaimers
            - Emphasize the importance of professional medical consultation
            - Provide evidence-based recommendations only
            """,
            agent=self.health_advisor,
            expected_output="Comprehensive health recommendations including immediate actions, dietary modifications, lifestyle changes, monitoring plans, and medical follow-up requirements with appropriate safety disclaimers."
        )
        
        # Enhanced Task 3: Report Generation
        report_task = Task(
            description=f"""
            Create a comprehensive, well-structured health report that combines the analysis and recommendations:
            
            **INPUT DATA:**
            - Blood analysis findings
            - Health recommendations 
            - Patient information: {self._format_patient_info(patient_info)}
            - Blood results: {self._format_blood_data_with_ranges(blood_data)}
            
            **REPORT STRUCTURE:**
            
            # 🩸 COMPREHENSIVE BLOOD TEST ANALYSIS REPORT
            
            ## 📋 EXECUTIVE SUMMARY
            - Overall health assessment (2-3 sentences)
            - Key findings summary
            - Recommended actions priority list
            
            ## 📊 DETAILED FINDINGS
            ### Parameters Within Normal Range
            ### Parameters Requiring Attention
            ### Critical Values (if any)
            
            ## 🔍 CLINICAL ANALYSIS
            - Detailed interpretation of abnormal values
            - Potential health implications
            - Parameter correlations and patterns
            
            ## 💊 RECOMMENDATIONS
            ### Immediate Actions
            ### Lifestyle Modifications
            ### Monitoring Plan
            ### Medical Follow-up
            
            ## 📅 NEXT STEPS
            - Clear action items with timelines
            - Follow-up recommendations
            
            ## ⚠️ IMPORTANT DISCLAIMERS
            - Medical consultation requirements
            - Limitations of AI analysis
            
            **FORMATTING REQUIREMENTS:**
            - Use clear headings and subheadings
            - Include bullet points for easy reading
            - Highlight critical information
            - Maintain professional medical communication standards
            - Ensure patient-friendly language while being thorough
            """,
            agent=self.report_generator,
            expected_output="A professional, comprehensive medical report with clear structure, executive summary, detailed findings, clinical analysis, actionable recommendations, and appropriate medical disclaimers formatted for both patient and healthcare provider use."
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[self.blood_analyst, self.health_advisor, self.report_generator],
            tasks=[analysis_task, recommendation_task, report_task],
            verbose=True,
            process="sequential"  # Ensure tasks run in order
        )
        
        try:
            # Execute the crew workflow
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            # Fallback to basic analysis if CrewAI fails
            return self._fallback_analysis(blood_data, patient_info)
    
    def _format_patient_info(self, patient_info: dict) -> str:
        """Format patient information for agent prompts"""
        if not patient_info:
            return "Patient information not available"
        
        formatted = []
        for key, value in patient_info.items():
            if value:
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return '\n'.join(formatted) if formatted else "Patient information not available"
    
    def _format_blood_data_with_ranges(self, blood_data: dict) -> str:
        """Format blood data with normal ranges for context"""
        if not blood_data:
            return "No blood test data available"
        
        formatted = []
        
        for param, value in blood_data.items():
            param_name = param.replace('_', ' ').title()
            
            # Get normal range and status
            try:
                normal_range = self.settings.get_normal_range(param)
                if isinstance(normal_range, tuple):
                    range_str = f"{normal_range[0]}-{normal_range[1]}"
                    
                    # Determine status
                    if value < normal_range[0]:
                        status = "LOW ⬇️"
                    elif value > normal_range[1]:
                        status = "HIGH ⬆️"
                    else:
                        status = "NORMAL ✅"
                    
                    formatted.append(f"- {param_name}: {value} (Normal: {range_str}) [{status}]")
                else:
                    formatted.append(f"- {param_name}: {value}")
                    
                # Check for critical values
                critical_check = self.settings.is_critical_value(param, value)
                if critical_check['is_critical']:
                    formatted[-1] += f" ⚠️ CRITICAL: {critical_check['message']}"
                    
            except Exception as e:
                formatted.append(f"- {param_name}: {value}")
        
        return '\n'.join(formatted)
    
    def _fallback_analysis(self, blood_data: dict, patient_info: dict) -> str:
        """Provide fallback analysis when CrewAI fails"""
        try:
            # Use basic LLM analysis as fallback
            basic_analysis = self.llm_clients.analyze_with_openai(blood_data, patient_info)
            recommendations = self.llm_clients.get_comprehensive_recommendations(
                basic_analysis, blood_data, patient_info
            )
            
            fallback_report = f"""
# 🩸 BLOOD TEST ANALYSIS REPORT
*(Generated using fallback analysis)*

## 📊 ANALYSIS RESULTS
{basic_analysis}

## 💊 RECOMMENDATIONS
{recommendations}

## ⚠️ SYSTEM NOTE
This analysis was generated using fallback methods due to multi-agent system unavailability. 
For the most comprehensive analysis, please ensure all system components are properly configured.

## 🚨 MEDICAL DISCLAIMER
This analysis is for informational purposes only. Always consult qualified healthcare 
professionals for medical advice and treatment decisions.
            """
            
            return fallback_report
            
        except Exception as e:
            return f"""
# ❌ ANALYSIS ERROR

Unable to complete blood test analysis due to system error: {str(e)}

## 📝 RECOMMENDATIONS
1. Check API configuration and connectivity
2. Verify blood data format and completeness
3. Consult healthcare professional for manual interpretation
4. Retry analysis after resolving technical issues

## 🚨 IMPORTANT
Please consult a qualified healthcare professional for proper blood test interpretation.
            """
    
    def quick_analysis(self, blood_data: dict, patient_info: dict) -> dict:
        """Provide quick analysis without full CrewAI workflow"""
        try:
            # Quick parameter assessment
            results = {
                "normal_count": 0,
                "abnormal_count": 0,
                "critical_count": 0,
                "parameter_status": {},
                "quick_insights": [],
                "emergency_level": "normal"
            }
            
            for param, value in blood_data.items():
                try:
                    # Check normal range
                    normal_range = self.settings.get_normal_range(param)
                    if isinstance(normal_range, tuple):
                        if normal_range[0] <= value <= normal_range[1]:
                            status = "normal"
                            results["normal_count"] += 1
                        else:
                            status = "abnormal"
                            results["abnormal_count"] += 1
                    else:
                        status = "unknown"
                    
                    # Check critical values
                    critical_check = self.settings.is_critical_value(param, value)
                    if critical_check['is_critical']:
                        status = "critical"
                        results["critical_count"] += 1
                        results["emergency_level"] = "critical"
                        results["quick_insights"].append(f"🚨 {critical_check['message']}")
                    
                    results["parameter_status"][param] = status
                    
                except Exception:
                    results["parameter_status"][param] = "unknown"
            
            # Generate quick insights
            if results["critical_count"] > 0:
                results["quick_insights"].insert(0, f"⚠️ {results['critical_count']} critical values detected - seek immediate medical attention")
            elif results["abnormal_count"] > results["normal_count"]:
                results["quick_insights"].append(f"📊 {results['abnormal_count']} parameters outside normal range - consider medical consultation")
            else:
                results["quick_insights"].append("✅ Most parameters within normal ranges")
            
            return results
            
        except Exception as e:
            return {
                "error": str(e),
                "quick_insights": ["❌ Quick analysis failed - please try full analysis"],
                "emergency_level": "unknown"
            }