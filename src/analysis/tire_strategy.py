"""
Tire Strategy Optimizer
======================
Models tire degradation and optimizes tire change windows for GT3/GTE racing.

Features:
- Compound-specific degradation curves
- Temperature and track condition modeling
- Tire cliff prediction
- Multi-stint optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TireCompound(Enum):
    """Available tire compounds."""
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    WET = "wet"
    INTERMEDIATE = "intermediate"


@dataclass
class TireSet:
    """Represents a set of tires."""
    compound: TireCompound
    life_remaining: float  # 0-1
    laps_completed: int
    heat_cycles: int
    grip_level: float  # relative to new tire
    degradation_rate: float  # per lap
    cliff_threshold: float  # life at which cliff occurs
    
    def is_past_cliff(self) -> bool:
        return self.life_remaining < self.cliff_threshold


@dataclass
class TirePrediction:
    """Tire performance prediction."""
    lap: int
    life_remaining: float
    grip_level: float
    lap_time_delta: float  # vs new tire
    is_cliff: bool
    
    
@dataclass
class TireStrategy:
    """Complete tire strategy."""
    compound_sequence: List[TireCompound]
    change_laps: List[int]
    total_tire_sets: int
    expected_time_loss: float  # from degradation
    optimal_change_windows: List[Tuple[int, int]]
    risk_assessment: str


class TireStrategyOptimizer:
    """
    Optimizes tire strategy for endurance racing.
    
    Models:
    - Non-linear degradation curves
    - Compound characteristics
    - Temperature sensitivity
    - Track surface evolution
    """
    
    # Compound characteristics (baseline values)
    COMPOUND_PROFILES = {
        TireCompound.SOFT: {
            'grip_initial': 1.05,
            'degradation_base': 0.020,
            'cliff_threshold': 0.35,
            'optimal_temp_min': 85,
            'optimal_temp_max': 100,
            'warmup_laps': 1
        },
        TireCompound.MEDIUM: {
            'grip_initial': 1.00,
            'degradation_base': 0.015,
            'cliff_threshold': 0.30,
            'optimal_temp_min': 80,
            'optimal_temp_max': 95,
            'warmup_laps': 1.5
        },
        TireCompound.HARD: {
            'grip_initial': 0.97,
            'degradation_base': 0.010,
            'cliff_threshold': 0.25,
            'optimal_temp_min': 75,
            'optimal_temp_max': 90,
            'warmup_laps': 2
        },
        TireCompound.WET: {
            'grip_initial': 0.85,  # On wet surface
            'degradation_base': 0.008,
            'cliff_threshold': 0.40,
            'optimal_temp_min': 60,
            'optimal_temp_max': 80,
            'warmup_laps': 2
        },
        TireCompound.INTERMEDIATE: {
            'grip_initial': 0.92,
            'degradation_base': 0.012,
            'cliff_threshold': 0.35,
            'optimal_temp_min': 65,
            'optimal_temp_max': 85,
            'warmup_laps': 1.5
        }
    }
    
    # Track wear profiles
    TRACK_WEAR_MULTIPLIERS = {
        'low': 0.8,      # Paul Ricard, Le Mans
        'medium': 1.0,   # Spa, Nurburgring
        'high': 1.3,     # Sebring, Barcelona
        'very_high': 1.5  # Street circuits
    }
    
    def __init__(
        self,
        track_wear: str = 'medium',
        track_temp: float = 25.0,
        base_lap_time: float = 100.0,
        tire_change_time: float = 12.0,
        max_tire_sets: int = 10
    ):
        """
        Initialize tire optimizer.
        
        Args:
            track_wear: Track abrasiveness ('low', 'medium', 'high', 'very_high')
            track_temp: Track temperature in Celsius
            base_lap_time: Reference lap time on new tires
            tire_change_time: Time to change 4 tires (not including pit lane)
            max_tire_sets: Maximum tire sets available
        """
        self.track_wear = track_wear
        self.track_temp = track_temp
        self.base_lap_time = base_lap_time
        self.tire_change_time = tire_change_time
        self.max_tire_sets = max_tire_sets
        
        self.wear_multiplier = self.TRACK_WEAR_MULTIPLIERS.get(track_wear, 1.0)
        
        logger.info(f"Tire Optimizer initialized:")
        logger.info(f"  - Track wear: {track_wear} (x{self.wear_multiplier})")
        logger.info(f"  - Track temp: {track_temp}°C")
    
    def create_tire_set(
        self,
        compound: TireCompound,
        used: bool = False,
        previous_laps: int = 0
    ) -> TireSet:
        """Create a new tire set."""
        profile = self.COMPOUND_PROFILES[compound]
        
        # Adjust degradation for track
        deg_rate = profile['degradation_base'] * self.wear_multiplier
        
        # Temperature effect
        temp_factor = self._temperature_factor(compound)
        deg_rate *= temp_factor
        
        # Initial state
        if used:
            life = max(0, 1.0 - (previous_laps * deg_rate))
            grip = self._calculate_grip(compound, life)
        else:
            life = 1.0
            grip = profile['grip_initial']
        
        return TireSet(
            compound=compound,
            life_remaining=life,
            laps_completed=previous_laps,
            heat_cycles=1 if used else 0,
            grip_level=grip,
            degradation_rate=deg_rate,
            cliff_threshold=profile['cliff_threshold']
        )
    
    def _temperature_factor(self, compound: TireCompound) -> float:
        """Calculate temperature impact on degradation."""
        profile = self.COMPOUND_PROFILES[compound]
        opt_min = profile['optimal_temp_min']
        opt_max = profile['optimal_temp_max']
        
        if opt_min <= self.track_temp <= opt_max:
            return 1.0  # Optimal range
        elif self.track_temp < opt_min:
            # Too cold - slight increase in wear
            return 1.0 + (opt_min - self.track_temp) * 0.01
        else:
            # Too hot - increased wear
            return 1.0 + (self.track_temp - opt_max) * 0.02
    
    def _calculate_grip(
        self, 
        compound: TireCompound, 
        life: float
    ) -> float:
        """Calculate grip level based on tire life."""
        profile = self.COMPOUND_PROFILES[compound]
        initial_grip = profile['grip_initial']
        cliff = profile['cliff_threshold']
        
        if life > cliff:
            # Linear degradation above cliff
            degradation = (1 - life) * 0.1
            return initial_grip - degradation
        else:
            # Exponential drop below cliff
            cliff_drop = (cliff - life) * 0.5
            return initial_grip - 0.1 - cliff_drop
    
    def predict_stint(
        self,
        tire_set: TireSet,
        num_laps: int
    ) -> List[TirePrediction]:
        """
        Predict tire performance over a stint.
        
        Args:
            tire_set: Starting tire set
            num_laps: Number of laps to predict
            
        Returns:
            List of TirePrediction for each lap
        """
        predictions = []
        current_life = tire_set.life_remaining
        
        for lap in range(1, num_laps + 1):
            # Calculate grip
            grip = self._calculate_grip(tire_set.compound, current_life)
            
            # Lap time delta (vs new tire)
            grip_loss = tire_set.grip_level - grip
            lap_delta = grip_loss * 2.0  # ~2s per 1% grip loss
            
            # Warmup effect on first laps
            warmup_laps = self.COMPOUND_PROFILES[tire_set.compound]['warmup_laps']
            if lap <= warmup_laps:
                warmup_penalty = (warmup_laps - lap + 1) * 0.5
                lap_delta += warmup_penalty
            
            is_cliff = current_life < tire_set.cliff_threshold
            
            predictions.append(TirePrediction(
                lap=tire_set.laps_completed + lap,
                life_remaining=current_life,
                grip_level=grip,
                lap_time_delta=lap_delta,
                is_cliff=is_cliff
            ))
            
            # Degrade for next lap
            current_life -= tire_set.degradation_rate
            current_life = max(0, current_life)
        
        return predictions
    
    def find_optimal_change_lap(
        self,
        compound: TireCompound,
        max_stint: int = 50
    ) -> Dict:
        """
        Find optimal lap to change tires for a compound.
        
        Balances:
        - Grip loss over stint
        - Time lost to pit stop
        - Risk of hitting cliff
        
        Args:
            compound: Tire compound to analyze
            max_stint: Maximum possible stint length
            
        Returns:
            Analysis dictionary
        """
        tire = self.create_tire_set(compound)
        predictions = self.predict_stint(tire, max_stint)
        
        # Find cliff lap
        cliff_lap = next(
            (p.lap for p in predictions if p.is_cliff),
            max_stint
        )
        
        # Optimal is usually a few laps before cliff
        optimal_lap = max(1, cliff_lap - 3)
        
        # Calculate time loss at different change points
        time_losses = []
        for change_lap in range(10, min(cliff_lap + 5, max_stint)):
            total_delta = sum(p.lap_time_delta for p in predictions[:change_lap])
            time_losses.append({
                'change_lap': change_lap,
                'cumulative_time_loss': total_delta,
                'avg_time_loss_per_lap': total_delta / change_lap
            })
        
        # Find minimum average time loss
        if time_losses:
            best = min(time_losses, key=lambda x: x['avg_time_loss_per_lap'])
            optimal_lap = best['change_lap']
        
        return {
            'compound': compound.value,
            'cliff_lap': cliff_lap,
            'optimal_change_lap': optimal_lap,
            'safe_window': (optimal_lap - 3, optimal_lap + 2),
            'max_stint_before_cliff': cliff_lap - 1,
            'degradation_rate': tire.degradation_rate,
            'time_losses': time_losses[-10:] if time_losses else []
        }
    
    def optimize_strategy(
        self,
        race_laps: int,
        num_stops: int,
        available_compounds: List[TireCompound] = None,
        weather: str = 'dry'
    ) -> TireStrategy:
        """
        Optimize tire strategy for the race.
        
        Args:
            race_laps: Total race laps
            num_stops: Planned number of stops
            available_compounds: Compounds to consider
            weather: Current/expected weather
            
        Returns:
            TireStrategy with recommendations
        """
        if available_compounds is None:
            if weather == 'dry':
                available_compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
            elif weather == 'wet':
                available_compounds = [TireCompound.WET]
            else:
                available_compounds = [TireCompound.INTERMEDIATE]
        
        num_stints = num_stops + 1
        avg_stint = race_laps / num_stints
        
        # Analyze each compound
        compound_analysis = {}
        for compound in available_compounds:
            compound_analysis[compound] = self.find_optimal_change_lap(compound)
        
        # Build strategy
        # Strategy: Use harder compounds for longer stints, softer for shorter
        stints_with_compounds = []
        
        for i in range(num_stints):
            stint_length = int(avg_stint)
            if i == num_stints - 1:
                # Final stint - use remaining laps
                stint_length = race_laps - sum(s[0] for s in stints_with_compounds)
            
            # Pick best compound for this stint length
            best_compound = self._select_compound_for_stint(
                stint_length, 
                compound_analysis
            )
            stints_with_compounds.append((stint_length, best_compound))
        
        # Calculate change laps
        change_laps = []
        cumulative = 0
        for stint_length, _ in stints_with_compounds[:-1]:
            cumulative += stint_length
            change_laps.append(cumulative)
        
        # Calculate windows
        windows = []
        for change_lap in change_laps:
            windows.append((change_lap - 3, change_lap + 2))
        
        # Calculate expected time loss
        total_time_loss = 0
        for stint_length, compound in stints_with_compounds:
            tire = self.create_tire_set(compound)
            predictions = self.predict_stint(tire, stint_length)
            total_time_loss += sum(p.lap_time_delta for p in predictions)
        
        # Risk assessment
        risk = self._assess_strategy_risk(stints_with_compounds, compound_analysis)
        
        return TireStrategy(
            compound_sequence=[c for _, c in stints_with_compounds],
            change_laps=change_laps,
            total_tire_sets=num_stints,
            expected_time_loss=total_time_loss,
            optimal_change_windows=windows,
            risk_assessment=risk
        )
    
    def _select_compound_for_stint(
        self,
        stint_length: int,
        compound_analysis: Dict
    ) -> TireCompound:
        """Select best compound for a stint length."""
        best_compound = None
        best_score = float('inf')
        
        for compound, analysis in compound_analysis.items():
            # Penalty if stint exceeds optimal
            if stint_length > analysis['max_stint_before_cliff']:
                continue  # Can't do this stint on this compound
            
            # Score = how close to optimal usage
            optimal = analysis['optimal_change_lap']
            score = abs(stint_length - optimal)
            
            # Prefer harder compounds for longer stints
            if compound == TireCompound.HARD:
                score *= 0.9
            elif compound == TireCompound.SOFT:
                score *= 1.1
            
            if score < best_score:
                best_score = score
                best_compound = compound
        
        return best_compound or TireCompound.MEDIUM
    
    def _assess_strategy_risk(
        self,
        stints: List[Tuple[int, TireCompound]],
        analysis: Dict
    ) -> str:
        """Assess overall strategy risk."""
        risks = []
        
        for stint_length, compound in stints:
            compound_data = analysis.get(compound, {})
            cliff = compound_data.get('cliff_lap', 40)
            
            if stint_length >= cliff:
                risks.append(f"High: {compound.value} will hit cliff")
            elif stint_length >= cliff - 3:
                risks.append(f"Medium: {compound.value} close to cliff")
        
        if not risks:
            return "Low: All stints within safe windows"
        
        return "; ".join(risks)
    
    def degradation_chart(
        self,
        compound: TireCompound,
        num_laps: int = 40
    ) -> pd.DataFrame:
        """
        Generate degradation data for visualization.
        
        Args:
            compound: Tire compound
            num_laps: Number of laps to simulate
            
        Returns:
            DataFrame with lap-by-lap degradation
        """
        tire = self.create_tire_set(compound)
        predictions = self.predict_stint(tire, num_laps)
        
        data = []
        for p in predictions:
            data.append({
                'lap': p.lap,
                'compound': compound.value,
                'life_pct': p.life_remaining * 100,
                'grip_level': p.grip_level,
                'lap_time_delta': p.lap_time_delta,
                'is_cliff': p.is_cliff
            })
        
        return pd.DataFrame(data)


def main():
    """Demo tire strategy optimizer."""
    print("Tire Strategy Optimizer Demo")
    print("=" * 50)
    
    optimizer = TireStrategyOptimizer(
        track_wear='high',  # Spa-like
        track_temp=28,
        base_lap_time=138
    )
    
    # Analyze each compound
    print("\nCompound Analysis:")
    for compound in [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]:
        analysis = optimizer.find_optimal_change_lap(compound)
        print(f"\n  {compound.value.upper()}:")
        print(f"    - Cliff at lap: {analysis['cliff_lap']}")
        print(f"    - Optimal change: lap {analysis['optimal_change_lap']}")
        print(f"    - Safe window: laps {analysis['safe_window']}")
        print(f"    - Degradation rate: {analysis['degradation_rate']:.3f}/lap")
    
    # Optimize strategy for 6h race at Spa (~156 laps, 4 stops)
    print("\n" + "=" * 50)
    print("Race Strategy (156 laps, 4 stops):")
    
    strategy = optimizer.optimize_strategy(
        race_laps=156,
        num_stops=4
    )
    
    print(f"\nCompound sequence: {[c.value for c in strategy.compound_sequence]}")
    print(f"Change at laps: {strategy.change_laps}")
    print(f"Tire sets used: {strategy.total_tire_sets}")
    print(f"Expected time loss from degradation: {strategy.expected_time_loss:.1f}s")
    print(f"Risk assessment: {strategy.risk_assessment}")
    
    print("\nOptimal change windows:")
    for i, window in enumerate(strategy.optimal_change_windows, 1):
        print(f"  Stop {i}: Laps {window[0]}-{window[1]}")
    
    # Generate degradation chart data
    print("\n" + "=" * 50)
    print("Generating degradation chart data...")
    
    chart_data = optimizer.degradation_chart(TireCompound.MEDIUM, 50)
    print(f"\nMedium compound degradation (first 10 laps):")
    print(chart_data.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
