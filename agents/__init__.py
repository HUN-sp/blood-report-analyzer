"""
Blood Report Analyzer Agents Module

This module contains all the AI agents used for blood report analysis,
including the main analyzer crew and recommendation agents.
"""

from .blood_analyzer import BloodAnalyzerCrew
from .recommendation_agent import RecommendationAgent, recommendation_agent

# Make agents available at package level
__all__ = [
    'BloodAnalyzerCrew',
    'RecommendationAgent', 
    'recommendation_agent'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Blood Report Analyzer Team'

# Package-level convenience functions
def get_analyzer_crew():
    """Get a new instance of BloodAnalyzerCrew"""
    return BloodAnalyzerCrew()

def get_recommendation_agent():
    """Get the global recommendation agent instance"""
    return recommendation_agent

def get_new_recommendation_agent():
    """Get a new instance of RecommendationAgent"""
    return RecommendationAgent()
