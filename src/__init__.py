"""
GT3/GTE Race Strategy Optimizer
==============================
A comprehensive race strategy optimization system for GT3/GTE endurance racing.

Modules:
- data_collection: Web scrapers and data generators
- preprocessing: Data cleaning and feature engineering
- analysis: Strategy optimization tools
- models: ML prediction models
- visualization: Dashboard and charts
"""

__version__ = '1.0.0'
__author__ = 'Race Strategy Team'

from .analysis import (
    PitWindowOptimizer,
    FuelStrategyCalculator,
    TireStrategyOptimizer,
    RaceSimulator
)
from .models import SafetyCarPredictor
from .data_collection import RaceDataGenerator

__all__ = [
    'PitWindowOptimizer',
    'FuelStrategyCalculator',
    'TireStrategyOptimizer',
    'RaceSimulator',
    'SafetyCarPredictor',
    'RaceDataGenerator'
]
