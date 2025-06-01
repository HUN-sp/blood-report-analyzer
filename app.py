import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.pdf_processor import BloodReportProcessor
from utils.llm_clients import LLMClients
from agents.blood_analyzer import BloodAnalyzerCrew
from agents.recommendation_agent import recommendation_agent
from config.settings import Settings
import json

def main():
    st.set_page_config(
        page_title="AI Blood Report Analyzer",
        page_icon="🩸",
        layout="wide"
    )
    
    st.title("🩸 AI Blood Report Analyzer")
    st.markdown("Upload your blood test report and get comprehensive AI-powered analysis with personalized recommendations")
    
    # Initialize settings
    settings = Settings()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        analysis_model = st.selectbox(
            "Choose Analysis Model",
            ["OpenAI GPT-3.5", "Google Gemini"],
            help="Select the AI model for blood report analysis"
        )
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        
        use_crewai = st.checkbox(
            "Use CrewAI Analysis", 
            value=True,
            help="Enable multi-agent analysis using CrewAI"
        )
        
        comprehensive_recommendations = st.checkbox(
            "Comprehensive Recommendations",
            value=True,
            help="Get detailed recommendations using specialized agents"
        )
        
        # API Status Check
        st.subheader("🔌 API Status")
        if st.button("Check API Connections"):
            check_api_status()
        
        # About section
        st.header("📋 About")
        st.info("""
        **Enhanced Features:**
        - 🤖 Direct LLM-powered PDF analysis
        - 📊 Structured data extraction
        - 🔍 Advanced pattern recognition
        - 💊 Comprehensive recommendations
        - 🚨 Critical value detection
        - 📈 Interactive visualizations
        - 📄 Downloadable reports
        
        ⚠️ **Medical Disclaimer**: 
        This tool is for informational purposes only. 
        Always consult qualified healthcare professionals.
        """)
        
        # Emergency indicators
        display_emergency_sidebar()
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Blood Report")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your blood test report in PDF format (max 10MB)"
        )
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.json(file_details)
            
            # Process PDF with enhanced extraction
            process_pdf_with_llm(uploaded_file, analysis_model)
    
    with col2:
        if 'extraction_results' in st.session_state and st.session_state.extraction_results.get('success'):
            display_extraction_results()
            
            # Analysis section
            st.header("🔬 AI Analysis")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
                    run_comprehensive_analysis(analysis_model, use_crewai, comprehensive_recommendations)
            
            with col_btn2:
                if st.button("🔄 Clear All", use_container_width=True):
                    clear_all_data()
            
            # Display analysis results
            if 'analysis_results' in st.session_state:
                display_comprehensive_results()
        else:
            st.info("👆 Please upload a blood report PDF to start")
            
            # Show sample format
            with st.expander("📋 Supported Blood Test Formats"):
                st.markdown("""
                **The AI can extract data from various blood test formats including:**
                - Complete Blood Count (CBC)
                - Basic Metabolic Panel (BMP)
                - Comprehensive Metabolic Panel (CMP)
                - Lipid Panel
                - Liver Function Tests
                - Thyroid Function Tests
                - Diabetes Markers (HbA1c, Glucose)
                - Vitamin and Mineral Tests
                
                **Supported Parameters:**
                - Hemoglobin, WBC, RBC, Platelets
                - Glucose, HbA1c
                - Cholesterol, HDL, LDL, Triglycerides
                - Creatinine, Urea, Bilirubin
                - ALT, AST, Alkaline Phosphatase
                - TSH, T3, T4
                - Vitamin D, B12, Folate
                - Iron, Ferritin
                - And many more...
                """)

def process_pdf_with_llm(uploaded_file, model_choice):
    """Process PDF using enhanced LLM extraction"""
    with st.spinner("🤖 Processing PDF with AI..."):
        try:
            processor = BloodReportProcessor()
            llm_clients = LLMClients()
            
            # Show debug info
            st.info("🔍 Debug Mode: Showing extraction process...")
            
            # Get comprehensive extraction
            extraction_results = processor.get_comprehensive_extraction(
                uploaded_file, llm_clients, model_choice
            )
            
            st.session_state.extraction_results = extraction_results
            
            # Debug output
            with st.expander("🐛 Debug Information"):
                st.write("**Extraction Results:**")
                st.json(extraction_results)
            
            if extraction_results['success']:
                st.success("✅ PDF processed successfully!")
                
                # Show extraction summary
                blood_values = extraction_results['blood_values']
                st.info(f"🔍 Extracted {len(blood_values)} blood parameters")
                
                # Show each extracted parameter
                if blood_values:
                    st.write("**Found Parameters:**")
                    for param, value in blood_values.items():
                        st.write(f"• {param.replace('_', ' ').title()}: {value}")
                
                # Show warnings if any
                if extraction_results.get('warnings'):
                    for warning in extraction_results['warnings']:
                        st.warning(f"⚠️ {warning}")
                        
            else:
                st.error("❌ Failed to extract blood values from PDF")
                if extraction_results.get('error'):
                    st.error(f"Error: {extraction_results['error']}")
                
                # Show raw text preview for debugging
                if extraction_results.get('raw_text_preview'):
                    with st.expander("🔍 Raw Text Preview (for debugging)"):
                        st.text(extraction_results['raw_text_preview'])
                
                # Show what we tried to extract
                st.info("💡 Trying to extract again with enhanced patterns...")
                
                # Try manual extraction as last resort
                try:
                    text = processor.extract_text_from_pdf(uploaded_file)
                    manual_extraction = processor._direct_extraction_fallback(text)
                    
                    if manual_extraction['blood_values']:
                        st.success("✅ Manual extraction succeeded!")
                        st.session_state.extraction_results = {
                            'success': True,
                            'blood_values': manual_extraction['blood_values'],
                            'patient_info': manual_extraction['patient_info'],
                            'warnings': ['Used manual extraction as fallback'],
                            'extraction_method': 'Manual fallback extraction'
                        }
                    else:
                        st.error("❌ Manual extraction also failed")
                        
                except Exception as manual_error:
                    st.error(f"Manual extraction error: {str(manual_error)}")
                
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            
            # Show full traceback for debugging
            import traceback
            with st.expander("🔧 Full Error Details"):
                st.code(traceback.format_exc())

def display_extraction_results():
    """Display extracted blood values and patient info"""
    results = st.session_state.extraction_results
    
    if results['blood_values']:
        st.subheader("📊 Extracted Blood Values")
        
        # Create DataFrame with status
        blood_data = results['blood_values']
        patient_info = results['patient_info']
        settings = Settings()
        
        df_data = []
        for param, value in blood_data.items():
            status = get_value_status(param, value, patient_info.get('gender'), settings)
            df_data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Value': value,
                'Status': status,
                'Unit': results.get('units', {}).get(param, ''),
                'Reference Range': results.get('reference_ranges', {}).get(param, '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with color coding
        st.dataframe(
            df.style.apply(color_code_status, axis=1),
            use_container_width=True
        )
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", len(blood_data))
        with col2:
            normal_count = sum(1 for row in df_data if row['Status'] == 'Normal')
            st.metric("Normal Values", normal_count)
        with col3:
            abnormal_count = len(blood_data) - normal_count
            st.metric("Abnormal Values", abnormal_count)
    
    # Patient information
    if results['patient_info']:
        st.subheader("👤 Patient Information")
        patient_df = pd.DataFrame(
            list(results['patient_info'].items()), 
            columns=['Field', 'Value']
        )
        st.dataframe(patient_df, use_container_width=True)

def run_comprehensive_analysis(model_choice, use_crewai, comprehensive_recommendations):
    """Run comprehensive analysis pipeline"""
    extraction_results = st.session_state.extraction_results
    blood_data = extraction_results['blood_values']
    patient_info = extraction_results['patient_info']
    
    if not blood_data:
        st.error("❌ No blood values available for analysis")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        llm_clients = LLMClients()
        
        # Step 1: Basic Analysis
        status_text.text("🔍 Performing basic analysis...")
        progress_bar.progress(25)
        
        if model_choice == "OpenAI GPT-3.5":
            basic_analysis = llm_clients.analyze_with_openai(blood_data, patient_info)
        else:
            basic_analysis = llm_clients.analyze_with_gemini(blood_data, patient_info)
        
        # Step 2: CrewAI Analysis (if enabled)
        crewai_analysis = None
        if use_crewai:
            status_text.text("🤖 Running multi-agent CrewAI analysis...")
            progress_bar.progress(50)
            try:
                crew = BloodAnalyzerCrew()
                crewai_analysis = crew.analyze_blood_report(blood_data, patient_info)
            except Exception as e:
                st.warning(f"CrewAI analysis failed: {str(e)}")
        
        # Step 3: Comprehensive Recommendations
        status_text.text("💊 Generating recommendations...")
        progress_bar.progress(75)
        
        recommendations = None
        emergency_indicators = None
        personalized_tips = None
        
        if comprehensive_recommendations:
            try:
                recommendation_results = recommendation_agent.generate_comprehensive_recommendations(
                    blood_data, basic_analysis, patient_info
                )
                recommendations = recommendation_results.get('structured_recommendations')
                emergency_indicators = recommendation_results.get('emergency_indicators')
                personalized_tips = recommendation_results.get('personalized_tips')
            except Exception as e:
                st.warning(f"Comprehensive recommendations failed: {str(e)}")
                recommendations = llm_clients.get_comprehensive_recommendations(
                    basic_analysis, blood_data, patient_info
                )
        else:
            recommendations = llm_clients.get_comprehensive_recommendations(
                basic_analysis, blood_data, patient_info
            )
        
        # Generate emergency indicators if not available
        if not emergency_indicators:
            emergency_indicators = check_emergency_indicators(blood_data)
        
        # Generate personalized tips if not available
        if not personalized_tips:
            personalized_tips = generate_personalized_tips(blood_data)
        
        # Step 4: Finalize
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.analysis_results = {
            'basic_analysis': basic_analysis,
            'crewai_analysis': crewai_analysis,
            'recommendations': recommendations,
            'emergency_indicators': emergency_indicators,
            'personalized_tips': personalized_tips,
            'model_used': model_choice,
            'crewai_enabled': use_crewai,
            'comprehensive_enabled': comprehensive_recommendations,
            'blood_data': blood_data,
            'patient_info': patient_info
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Comprehensive analysis completed!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_comprehensive_results():
    """Display comprehensive analysis results in tabs"""
    results = st.session_state.analysis_results
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analysis", 
        "💊 Recommendations", 
        "🎯 Tips",
        "📈 Charts", 
        "🚨 Alerts",
        "📄 Report"
    ])
    
    with tab1:
        st.subheader("🔬 Health Analysis")
        
        # Basic Analysis
        st.markdown("### 🤖 AI Analysis")
        st.write(results['basic_analysis'])
        st.caption(f"Analysis provided by: {results['model_used']}")
        
        # CrewAI Analysis
        if results.get('crewai_analysis'):
            st.markdown("### 👥 Multi-Agent Analysis")
            st.write(results['crewai_analysis'])
        
        # Quick insights
        display_quick_insights(results['blood_data'])
    
    with tab2:
        st.subheader("💊 Comprehensive Recommendations")
        st.write(results['recommendations'])
        
        # Medical disclaimer
        st.error("""
        🚨 **IMPORTANT MEDICAL DISCLAIMER**: 
        These recommendations are AI-generated for informational purposes only. 
        Always consult qualified healthcare professionals before making medical decisions.
        """)
    
    with tab3:
        st.subheader("🎯 Personalized Health Tips")
        if results.get('personalized_tips'):
            for i, tip in enumerate(results['personalized_tips'], 1):
                st.info(f"💡 **Tip {i}**: {tip}")
        else:
            st.info("No personalized tips available")
    
    with tab4:
        st.subheader("📈 Blood Values Visualization")
        create_enhanced_visualizations(results['blood_data'])
    
    with tab5:
        st.subheader("🚨 Risk Assessment & Alerts")
        display_risk_assessment(results['emergency_indicators'])
    
    with tab6:
        st.subheader("📄 Download Complete Report")
        generate_downloadable_report(results)

def display_quick_insights(blood_data):
    """Display quick insights about blood values"""
    st.markdown("### ⚡ Quick Insights")
    
    settings = Settings()
    insights = []
    
    for param, value in blood_data.items():
        try:
            param_info = settings.get_parameter_info(param)
            critical_check = settings.is_critical_value(param, value)
            
            if critical_check['is_critical']:
                insights.append(f"🚨 **{param_info['full_name']}**: {critical_check['message']}")
            else:
                normal_range = settings.get_normal_range(param)
                if isinstance(normal_range, tuple):
                    if value < normal_range[0]:
                        insights.append(f"⬇️ **{param_info['full_name']}**: Below normal range")
                    elif value > normal_range[1]:
                        insights.append(f"⬆️ **{param_info['full_name']}**: Above normal range")
                    else:
                        insights.append(f"✅ **{param_info['full_name']}**: Within normal range")
        except:
            continue
    
    if insights:
        for insight in insights[:5]:  # Show top 5 insights
            st.write(insight)
    else:
        st.info("All values appear to be within acceptable ranges")

def create_enhanced_visualizations(blood_data):
    """Create enhanced visualizations"""
    if not blood_data:
        st.info("No data available for visualization")
        return
    
    viz_type = st.selectbox(
        "Choose Visualization",
        ["Overview Dashboard", "Parameter Comparison", "Risk Assessment Chart"]
    )
    
    if viz_type == "Overview Dashboard":
        create_overview_dashboard(blood_data)
    elif viz_type == "Parameter Comparison":
        create_parameter_comparison(blood_data)
    elif viz_type == "Risk Assessment Chart":
        create_risk_chart(blood_data)

def create_overview_dashboard(blood_data):
    """Create an overview dashboard"""
    settings = Settings()
    
    # Create metrics grid
    cols = st.columns(3)
    params = list(blood_data.items())
    
    for i, (param, value) in enumerate(params[:6]):  # Show first 6 parameters
        with cols[i % 3]:
            try:
                normal_range = settings.get_normal_range(param)
                if isinstance(normal_range, tuple):
                    mid_point = (normal_range[0] + normal_range[1]) / 2
                    delta = value - mid_point
                    delta_str = f"{delta:+.1f} from midpoint"
                else:
                    delta_str = ""
                
                st.metric(
                    label=param.replace('_', ' ').title(),
                    value=f"{value}",
                    delta=delta_str
                )
            except:
                st.metric(
                    label=param.replace('_', ' ').title(),
                    value=f"{value}"
                )

def check_api_status():
    """Check API connection status"""
    llm_clients = LLMClients()
    status = llm_clients.test_api_connection()
    
    if status['openai']:
        st.success("✅ OpenAI API: Connected")
    else:
        st.error("❌ OpenAI API: Not connected")
    
    if status['gemini']:
        st.success("✅ Gemini API: Connected")
    else:
        st.error("❌ Gemini API: Not connected")
    
    if status['errors']:
        st.error("Errors:")
        for error in status['errors']:
            st.error(f"• {error}")

def display_emergency_sidebar():
    """Display emergency indicators in sidebar"""
    if 'analysis_results' in st.session_state:
        emergency_info = st.session_state.analysis_results.get('emergency_indicators', {})
        
        st.header("🚨 Emergency Status")
        
        emergency_level = emergency_info.get('emergency_level', 'normal')
        
        if emergency_level == 'critical':
            st.error("🚨 CRITICAL VALUES DETECTED")
            st.error("Seek immediate medical attention!")
        elif emergency_level == 'urgent':
            st.warning("⚠️ Urgent consultation needed")
        else:
            st.success("✅ No critical values")

# Helper Functions
def get_value_status(parameter, value, gender, settings):
    """Get status of blood value"""
    try:
        normal_range = settings.get_normal_range(parameter, gender)
        if isinstance(normal_range, tuple):
            if value < normal_range[0]:
                return "Low"
            elif value > normal_range[1]:
                return "High"
            else:
                return "Normal"
        else:
            return "Unknown"
    except:
        return "Unknown"

def color_code_status(row):
    """Color code DataFrame rows based on status"""
    colors = {
        'Normal': 'background-color: #d4edda',
        'High': 'background-color: #f8d7da', 
        'Low': 'background-color: #fff3cd',
        'Unknown': 'background-color: #e2e3e5'
    }
    
    color = colors.get(row['Status'], 'background-color: #e2e3e5')
    return [color] * len(row)

def check_emergency_indicators(blood_data):
    """Check for emergency indicators"""
    settings = Settings()
    emergency_indicators = {
        "critical_values": [],
        "urgent_consultation": [],
        "emergency_level": "normal"
    }
    
    for param, value in blood_data.items():
        critical_check = settings.is_critical_value(param, value)
        if critical_check['is_critical']:
            emergency_indicators["critical_values"].append({
                "parameter": param,
                "value": value,
                "level": critical_check['level'],
                "message": critical_check['message']
            })
            emergency_indicators["emergency_level"] = "critical"
    
    return emergency_indicators

def generate_personalized_tips(blood_data):
    """Generate personalized tips based on blood values"""
    tips = []
    
    for param, value in blood_data.items():
        if param == "glucose" and value > 100:
            tips.append("🍎 High glucose detected: Focus on low-glycemic foods and regular exercise")
        elif param == "cholesterol" and value > 200:
            tips.append("🥑 High cholesterol: Increase omega-3 intake and reduce saturated fats")
        elif param == "hemoglobin" and value < 12:
            tips.append("🥬 Low hemoglobin: Include iron-rich foods like spinach and lean meat")
    
    return tips

def display_risk_assessment(emergency_indicators):
    """Display comprehensive risk assessment"""
    if not emergency_indicators:
        st.info("No risk assessment data available")
        return
    
    emergency_level = emergency_indicators.get('emergency_level', 'normal')
    
    # Overall risk level
    if emergency_level == 'critical':
        st.error("🚨 **CRITICAL RISK LEVEL**")
        st.error("Immediate medical attention required!")
    elif emergency_level == 'urgent':
        st.warning("⚠️ **MODERATE RISK LEVEL**")
        st.warning("Consult healthcare provider within 24-48 hours")
    else:
        st.success("✅ **LOW RISK LEVEL**")
        st.success("No immediate concerns detected")
    
    # Critical values
    critical_values = emergency_indicators.get('critical_values', [])
    if critical_values:
        st.subheader("🚨 Critical Values")
        for value in critical_values:
            st.error(f"**{value['parameter'].title()}**: {value['value']} - {value['message']}")

def generate_downloadable_report(results):
    """Generate comprehensive downloadable report"""
    report_content = f"""
# 🩸 COMPREHENSIVE BLOOD TEST ANALYSIS REPORT

## 📋 PATIENT INFORMATION
{format_dict_for_report(results['patient_info'])}

## 📊 BLOOD TEST RESULTS
{format_dict_for_report(results['blood_data'])}

## 🔬 AI ANALYSIS
### Primary Analysis ({results['model_used']})
{results['basic_analysis']}

### Multi-Agent CrewAI Analysis
{results.get('crewai_analysis', 'Not performed')}

## 💊 COMPREHENSIVE RECOMMENDATIONS
{results['recommendations']}

## 🎯 PERSONALIZED HEALTH TIPS
{format_list_for_report(results.get('personalized_tips', []))}

## 🚨 EMERGENCY INDICATORS
Emergency Level: {results.get('emergency_indicators', {}).get('emergency_level', 'Normal').upper()}

Critical Values: {len(results.get('emergency_indicators', {}).get('critical_values', []))} detected

## ⚠️ MEDICAL DISCLAIMER
This report is generated by AI for informational purposes only and should NOT replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals before making any medical decisions.

## 📋 REPORT METADATA
- Generated by: AI Blood Report Analyzer v2.0
- Analysis Model: {results['model_used']}
- CrewAI Enabled: {results.get('crewai_enabled', False)}
- Comprehensive Mode: {results.get('comprehensive_enabled', False)}
- Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---
© 2025 AI Blood Report Analyzer - Enhanced LLM Integration
    """
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download Full Report",
            data=report_content,
            file_name=f"blood_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Create CSV data for blood values
        csv_data = pd.DataFrame(list(results['blood_data'].items()), columns=['Parameter', 'Value'])
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data.to_csv(index=False),
            file_name=f"blood_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Create JSON export
        json_data = {
            "patient_info": results['patient_info'],
            "blood_data": results['blood_data'],
            "analysis_summary": {
                "model_used": results['model_used'],
                "emergency_level": results.get('emergency_indicators', {}).get('emergency_level', 'normal'),
                "total_parameters": len(results['blood_data']),
                "generated_timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        st.download_button(
            label="📋 Download JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"blood_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Preview report
    with st.expander("👀 Preview Report"):
        st.text(report_content[:1500] + "..." if len(report_content) > 1500 else report_content)

def create_parameter_comparison(blood_data):
    """Create parameter comparison chart"""
    settings = Settings()
    
    # Prepare data for comparison
    comparison_data = []
    for param, value in list(blood_data.items())[:8]:  # Limit to 8 for readability
        try:
            normal_range = settings.get_normal_range(param)
            if isinstance(normal_range, tuple):
                comparison_data.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Current Value': value,
                    'Normal Min': normal_range[0],
                    'Normal Max': normal_range[1],
                    'Status': get_value_status(param, value, None, settings)
                })
        except:
            continue
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        # Add current values
        fig.add_trace(go.Bar(
            name='Current Value',
            x=df['Parameter'],
            y=df['Current Value'],
            marker_color=['red' if status != 'Normal' else 'green' for status in df['Status']]
        ))
        
        # Add normal range indicators
        fig.add_trace(go.Scatter(
            name='Normal Range',
            x=df['Parameter'],
            y=df['Normal Max'],
            mode='markers',
            marker=dict(symbol='line-ns', size=20, color='blue'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Blood Parameters vs Normal Ranges",
            xaxis_title="Parameters",
            yaxis_title="Values",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_risk_chart(blood_data):
    """Create risk assessment chart"""
    settings = Settings()
    
    risk_levels = {'Low': 0, 'Normal': 0, 'High': 0, 'Critical': 0}
    
    for param, value in blood_data.items():
        try:
            critical_check = settings.is_critical_value(param, value)
            if critical_check['is_critical']:
                risk_levels['Critical'] += 1
            else:
                status = get_value_status(param, value, None, settings)
                if status == 'Normal':
                    risk_levels['Normal'] += 1
                elif status in ['High', 'Low']:
                    risk_levels['High'] += 1
                else:
                    risk_levels['Low'] += 1
        except:
            risk_levels['Low'] += 1
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_levels.keys()),
        values=list(risk_levels.values()),
        hole=0.3,
        marker_colors=['green', 'blue', 'orange', 'red']
    )])
    
    fig.update_layout(
        title="Risk Distribution of Blood Parameters",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show risk summary
    total_params = sum(risk_levels.values())
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Normal", risk_levels['Normal'], f"{risk_levels['Normal']/total_params*100:.1f}%")
    with col2:
        st.metric("Low Risk", risk_levels['Low'], f"{risk_levels['Low']/total_params*100:.1f}%")
    with col3:
        st.metric("High Risk", risk_levels['High'], f"{risk_levels['High']/total_params*100:.1f}%")
    with col4:
        st.metric("Critical", risk_levels['Critical'], f"{risk_levels['Critical']/total_params*100:.1f}%")

def clear_all_data():
    """Clear all session data"""
    keys_to_clear = ['extraction_results', 'analysis_results']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ All data cleared successfully!")
    st.rerun()

def format_dict_for_report(data_dict):
    """Format dictionary for report"""
    if not data_dict:
        return "No data available"
    
    formatted = []
    for key, value in data_dict.items():
        formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
    return '\n'.join(formatted)

def format_list_for_report(data_list):
    """Format list for report"""
    if not data_list:
        return "No tips available"
    
    formatted = []
    for i, item in enumerate(data_list, 1):
        formatted.append(f"{i}. {item}")
    return '\n'.join(formatted)

if __name__ == "__main__":
    main()