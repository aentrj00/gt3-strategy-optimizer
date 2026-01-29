"""
Pit Window Optimizer
===================
Calculates optimal pit stop windows using dynamic programming and
multi-objective optimization.

Key features:
- Multi-stop strategy comparison (1-stop, 2-stop, 3-stop, etc.)
- Undercut/overcut opportunity detection
- Safety car probability integration
- Fuel and tire degradation modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Pit stop strategy types."""
    AGGRESSIVE = "aggressive"  # Minimize stops, push tires/fuel
    CONSERVATIVE = "conservative"  # Extra margin, safe windows
    BALANCED = "balanced"  # Optimal trade-off
    SC_OPPORTUNISTIC = "sc_opportunistic"  # Bet on safety cars


@dataclass
class PitWindow:
    """Represents an optimal pit window."""
    lap_start: int
    lap_end: int
    lap_optimal: int
    priority: float  # 0-1, higher = more critical
    reason: str
    time_cost: float  # seconds lost by pitting
    fuel_at_optimal: float
    tire_life_at_optimal: float
    
    def __str__(self):
        return f"Laps {self.lap_start}-{self.lap_end} (optimal: {self.lap_optimal})"


@dataclass
class Strategy:
    """Complete race strategy."""
    name: str
    num_stops: int
    pit_windows: List[PitWindow]
    total_race_time: float  # seconds
    total_pit_time: float
    fuel_used: float
    tire_sets_used: int
    risk_score: float  # 0-1, lower = safer
    probability_success: float
    notes: List[str] = field(default_factory=list)
    
    def __str__(self):
        windows = ', '.join([str(w) for w in self.pit_windows])
        return f"{self.name} ({self.num_stops}-stop): {windows}"


class PitWindowOptimizer:
    """
    Optimizes pit stop windows for endurance racing.
    
    Uses dynamic programming to find optimal pit stop timing
    considering fuel, tires, and race conditions.
    """
    
    def __init__(
        self,
        race_duration_minutes: float = 360,  # 6 hours
        lap_time_seconds: float = 100,
        pit_stop_time: float = 65,  # stationary + pit lane
        fuel_capacity: float = 120,  # liters
        fuel_consumption: float = 3.0,  # liters per lap
        tire_degradation_rate: float = 0.025,  # per lap (2.5%)
        tire_cliff_threshold: float = 0.30,  # 30% life = cliff
        safety_car_probability: float = 0.15,  # per hour
        weight_penalty_per_kg: float = 0.03  # seconds per lap per kg
    ):
        """
        Initialize the optimizer.
        
        Args:
            race_duration_minutes: Total race duration
            lap_time_seconds: Base lap time without penalties
            pit_stop_time: Total pit stop time (lane + stationary)
            fuel_capacity: Maximum fuel tank capacity (liters)
            fuel_consumption: Fuel used per lap (liters)
            tire_degradation_rate: Tire life lost per lap (0-1)
            tire_cliff_threshold: Tire life at which performance drops sharply
            safety_car_probability: Probability of SC per hour
            weight_penalty_per_kg: Lap time penalty per kg of fuel
        """
        self.race_duration = race_duration_minutes * 60  # Convert to seconds
        self.base_lap_time = lap_time_seconds
        self.pit_time = pit_stop_time
        self.fuel_capacity = fuel_capacity
        self.fuel_consumption = fuel_consumption
        self.tire_deg_rate = tire_degradation_rate
        self.tire_cliff = tire_cliff_threshold
        self.sc_probability = safety_car_probability
        self.weight_penalty = weight_penalty_per_kg
        self.fuel_density = 0.75  # kg per liter
        
        # Calculated values
        self.estimated_laps = int(self.race_duration / self.base_lap_time)
        self.max_laps_on_fuel = int(self.fuel_capacity / self.fuel_consumption)
        self.max_laps_on_tires = int((1 - self.tire_cliff) / self.tire_deg_rate)
        
        # The real constraint is the minimum of fuel and tires
        self.max_stint_length = min(self.max_laps_on_fuel, self.max_laps_on_tires)
        
        logger.info(f"Initialized optimizer:")
        logger.info(f"  - Estimated race laps: {self.estimated_laps}")
        logger.info(f"  - Max laps on fuel: {self.max_laps_on_fuel}")
        logger.info(f"  - Max laps on tires: {self.max_laps_on_tires}")
        logger.info(f"  - Max stint length: {self.max_stint_length}")
    
    def calculate_lap_time(
        self,
        lap_in_stint: int,
        fuel_remaining: float,
        tire_life: float
    ) -> float:
        """
        Calculate lap time with all penalties.
        
        Args:
            lap_in_stint: Current lap number within stint
            fuel_remaining: Fuel remaining in liters
            tire_life: Tire life remaining (0-1)
            
        Returns:
            Lap time in seconds
        """
        lap_time = self.base_lap_time
        
        # Fuel weight penalty
        fuel_weight = fuel_remaining * self.fuel_density
        lap_time += fuel_weight * self.weight_penalty
        
        # Tire degradation penalty
        if tire_life > self.tire_cliff:
            # Linear degradation above cliff
            tire_penalty = (1 - tire_life) * 2.0
        else:
            # Exponential penalty below cliff
            tire_penalty = 2.0 + (self.tire_cliff - tire_life) * 10.0
        lap_time += tire_penalty
        
        # Tire warmup (first lap of stint)
        if lap_in_stint == 1:
            lap_time += 1.5
        elif lap_in_stint == 2:
            lap_time += 0.5
            
        return lap_time
    
    def simulate_stint(
        self,
        num_laps: int,
        start_fuel: float = None
    ) -> Tuple[float, float, float]:
        """
        Simulate a stint and calculate total time.
        
        Args:
            num_laps: Number of laps in stint
            start_fuel: Starting fuel (default: full tank)
            
        Returns:
            Tuple of (total_time, end_fuel, end_tire_life)
        """
        if start_fuel is None:
            start_fuel = self.fuel_capacity
            
        total_time = 0
        fuel = start_fuel
        tire_life = 1.0
        
        for lap in range(1, num_laps + 1):
            lap_time = self.calculate_lap_time(lap, fuel, tire_life)
            total_time += lap_time
            
            fuel -= self.fuel_consumption
            tire_life -= self.tire_deg_rate
            
            if fuel < 0 or tire_life < 0:
                return float('inf'), 0, 0  # Invalid stint
        
        return total_time, fuel, tire_life
    
    def calculate_strategy(
        self,
        num_stops: int,
        strategy_type: StrategyType = StrategyType.BALANCED
    ) -> Optional[Strategy]:
        """
        Calculate optimal strategy for a given number of stops.
        
        Uses dynamic programming to find optimal pit lap numbers.
        
        Args:
            num_stops: Number of pit stops
            strategy_type: Strategy approach
            
        Returns:
            Strategy object or None if infeasible
        """
        if num_stops == 0:
            # No-stop strategy (only possible for very short races)
            if self.estimated_laps <= self.max_stint_length:
                stint_time, _, _ = self.simulate_stint(self.estimated_laps)
                return Strategy(
                    name="No Stop",
                    num_stops=0,
                    pit_windows=[],
                    total_race_time=stint_time,
                    total_pit_time=0,
                    fuel_used=self.estimated_laps * self.fuel_consumption,
                    tire_sets_used=1,
                    risk_score=0.8,  # High risk - no margin
                    probability_success=0.7
                )
            return None
        
        # Calculate optimal stint lengths
        num_stints = num_stops + 1
        avg_stint = self.estimated_laps / num_stints
        
        # Check feasibility
        if avg_stint > self.max_stint_length * 1.1:
            return None  # Not enough stops
        
        # Optimize stint distribution
        stints = self._optimize_stint_distribution(num_stints, strategy_type)
        
        if stints is None:
            return None
        
        # Calculate pit windows
        pit_windows = []
        cumulative_laps = 0
        total_time = 0
        total_pit_time = 0
        
        for i, stint_length in enumerate(stints[:-1]):  # Don't pit after final stint
            cumulative_laps += stint_length
            
            # Calculate window bounds
            window_margin = 3 if strategy_type == StrategyType.CONSERVATIVE else 2
            
            window = PitWindow(
                lap_start=max(1, cumulative_laps - window_margin),
                lap_end=cumulative_laps + window_margin,
                lap_optimal=cumulative_laps,
                priority=1.0 if i == 0 else 0.8,  # First stop most critical
                reason=self._get_pit_reason(stint_length, stints, i),
                time_cost=self.pit_time,
                fuel_at_optimal=self.fuel_consumption * window_margin,
                tire_life_at_optimal=max(0, 1 - stint_length * self.tire_deg_rate)
            )
            pit_windows.append(window)
            
            # Add stint time
            stint_time, _, _ = self.simulate_stint(stint_length)
            total_time += stint_time
            total_pit_time += self.pit_time
        
        # Final stint
        final_stint_time, _, _ = self.simulate_stint(stints[-1])
        total_time += final_stint_time
        
        # Calculate risk and success probability
        risk_score = self._calculate_risk(stints, strategy_type)
        success_prob = self._calculate_success_probability(stints, num_stops)
        
        return Strategy(
            name=f"{num_stops}-Stop {strategy_type.value.title()}",
            num_stops=num_stops,
            pit_windows=pit_windows,
            total_race_time=total_time + total_pit_time,
            total_pit_time=total_pit_time,
            fuel_used=self.estimated_laps * self.fuel_consumption,
            tire_sets_used=num_stints,
            risk_score=risk_score,
            probability_success=success_prob,
            notes=self._generate_strategy_notes(stints, strategy_type)
        )
    
    def _optimize_stint_distribution(
        self,
        num_stints: int,
        strategy_type: StrategyType
    ) -> Optional[List[int]]:
        """
        Optimize how laps are distributed across stints.
        
        Different strategies:
        - Balanced: Equal stints
        - Aggressive: Longer early stints (fresher tires, more fuel)
        - Conservative: Shorter early stints with margin
        - SC_Opportunistic: Longer stints betting on SC resets
        """
        total_laps = self.estimated_laps
        
        if strategy_type == StrategyType.BALANCED:
            # Equal distribution
            base_stint = total_laps // num_stints
            remainder = total_laps % num_stints
            stints = [base_stint + (1 if i < remainder else 0) for i in range(num_stints)]
            
        elif strategy_type == StrategyType.AGGRESSIVE:
            # Front-load longer stints (use fresh rubber advantage)
            base = total_laps / num_stints
            stints = []
            for i in range(num_stints):
                factor = 1.1 - (i * 0.1 / num_stints)  # Decrease stint length
                stints.append(int(base * factor))
            # Adjust to match total
            diff = total_laps - sum(stints)
            stints[-1] += diff
            
        elif strategy_type == StrategyType.CONSERVATIVE:
            # Build in margin, shorter stints
            target_stint = self.max_stint_length * 0.85  # 15% margin
            stints = [int(target_stint)] * num_stints
            # Adjust final stint
            covered = sum(stints[:-1])
            stints[-1] = total_laps - covered
            
        else:  # SC_OPPORTUNISTIC
            # Maximize stint length (bet on SC for tire/fuel save)
            max_stint = min(self.max_stint_length, total_laps // num_stints + 5)
            stints = [max_stint] * (num_stints - 1)
            stints.append(total_laps - sum(stints))
        
        # Validate all stints are feasible
        for stint in stints:
            if stint > self.max_stint_length or stint < 1:
                return None
        
        return stints
    
    def _get_pit_reason(
        self,
        stint_length: int,
        all_stints: List[int],
        stint_index: int
    ) -> str:
        """Generate reason for pit window."""
        reasons = []
        
        fuel_laps = self.max_laps_on_fuel
        tire_laps = self.max_laps_on_tires
        
        if stint_length >= fuel_laps * 0.9:
            reasons.append("Fuel critical")
        if stint_length >= tire_laps * 0.9:
            reasons.append("Tire cliff approaching")
        if not reasons:
            reasons.append("Optimal strategy window")
            
        return ", ".join(reasons)
    
    def _calculate_risk(
        self,
        stints: List[int],
        strategy_type: StrategyType
    ) -> float:
        """Calculate risk score for strategy (0-1)."""
        risk = 0.0
        
        for stint in stints:
            # Risk increases as stint approaches limits
            fuel_margin = 1 - (stint / self.max_laps_on_fuel)
            tire_margin = 1 - (stint / self.max_laps_on_tires)
            
            # Risk is inverse of margin
            stint_risk = 1 - min(fuel_margin, tire_margin)
            risk += stint_risk
        
        risk /= len(stints)
        
        # Strategy type modifier
        if strategy_type == StrategyType.AGGRESSIVE:
            risk *= 1.2
        elif strategy_type == StrategyType.CONSERVATIVE:
            risk *= 0.8
        elif strategy_type == StrategyType.SC_OPPORTUNISTIC:
            risk *= 1.3
            
        return min(1.0, risk)
    
    def _calculate_success_probability(
        self,
        stints: List[int],
        num_stops: int
    ) -> float:
        """Calculate probability of strategy success."""
        # Base probability
        prob = 0.95
        
        # Reduce for each risky stint
        for stint in stints:
            margin = 1 - (stint / self.max_stint_length)
            if margin < 0.1:
                prob *= 0.9  # High risk stint
            elif margin < 0.2:
                prob *= 0.95
        
        # More stops = more things can go wrong
        prob *= (0.98 ** num_stops)
        
        # SC probability could help or hurt
        race_hours = self.race_duration / 3600
        expected_scs = self.sc_probability * race_hours
        
        # SC slightly reduces success (disruption) but can also help
        prob *= (1 - 0.02 * expected_scs)
        
        return max(0.5, min(0.99, prob))
    
    def _generate_strategy_notes(
        self,
        stints: List[int],
        strategy_type: StrategyType
    ) -> List[str]:
        """Generate notes about the strategy."""
        notes = []
        
        # Stint analysis
        max_stint = max(stints)
        min_stint = min(stints)
        
        if max_stint > self.max_stint_length * 0.95:
            notes.append(f"⚠️ Stint {stints.index(max_stint)+1} is at fuel/tire limit")
        
        if strategy_type == StrategyType.AGGRESSIVE:
            notes.append("🏁 Aggressive strategy - maximize track position")
        elif strategy_type == StrategyType.CONSERVATIVE:
            notes.append("🛡️ Conservative strategy - prioritize reliability")
        elif strategy_type == StrategyType.SC_OPPORTUNISTIC:
            notes.append("🚨 Betting on Safety Car - high risk/reward")
        
        # Undercut potential
        if min_stint >= 5:
            notes.append("📊 Undercut window available in all stints")
        
        return notes
    
    def optimize(
        self,
        min_stops: int = 1,
        max_stops: int = 5,
        include_types: List[StrategyType] = None
    ) -> List[Strategy]:
        """
        Find optimal strategies across different stop counts.
        
        Args:
            min_stops: Minimum pit stops to consider
            max_stops: Maximum pit stops to consider
            include_types: Strategy types to evaluate
            
        Returns:
            List of Strategy objects, sorted by total time
        """
        if include_types is None:
            include_types = [StrategyType.BALANCED, StrategyType.AGGRESSIVE]
        
        strategies = []
        
        for num_stops in range(min_stops, max_stops + 1):
            for strategy_type in include_types:
                strategy = self.calculate_strategy(num_stops, strategy_type)
                if strategy:
                    strategies.append(strategy)
        
        # Sort by total race time
        strategies.sort(key=lambda s: s.total_race_time)
        
        return strategies
    
    def find_undercut_windows(
        self,
        competitor_pit_lap: int,
        our_position: int = 2
    ) -> Dict:
        """
        Find undercut/overcut opportunities relative to a competitor.
        
        Args:
            competitor_pit_lap: Expected lap competitor will pit
            our_position: Our current position (1 = leader)
            
        Returns:
            Dictionary with undercut/overcut analysis
        """
        analysis = {
            'competitor_pit_lap': competitor_pit_lap,
            'undercut_window': None,
            'overcut_window': None,
            'recommendation': None
        }
        
        # Undercut: Pit earlier to gain track position
        if our_position > 1:  # Only useful if behind
            undercut_start = max(1, competitor_pit_lap - 4)
            undercut_end = competitor_pit_lap - 1
            
            analysis['undercut_window'] = {
                'lap_range': (undercut_start, undercut_end),
                'optimal_lap': competitor_pit_lap - 2,
                'expected_gain_seconds': 1.5,  # Fresh tire advantage
                'risk': 'medium'
            }
        
        # Overcut: Pit later when track is clear
        overcut_start = competitor_pit_lap + 1
        overcut_end = competitor_pit_lap + 3
        
        analysis['overcut_window'] = {
            'lap_range': (overcut_start, overcut_end),
            'optimal_lap': competitor_pit_lap + 2,
            'expected_gain_seconds': 0.5,  # Clear track advantage
            'risk': 'low'
        }
        
        # Recommendation
        if our_position > 1:
            analysis['recommendation'] = 'undercut'
            analysis['reason'] = 'Behind competitor - undercut to gain position'
        else:
            analysis['recommendation'] = 'overcut'
            analysis['reason'] = 'Leading - overcut to maintain clean air'
        
        return analysis
    
    def to_dataframe(self, strategies: List[Strategy]) -> pd.DataFrame:
        """Convert strategies to DataFrame for visualization."""
        data = []
        
        for s in strategies:
            data.append({
                'Strategy': s.name,
                'Stops': s.num_stops,
                'Total Time (s)': round(s.total_race_time, 1),
                'Pit Time (s)': round(s.total_pit_time, 1),
                'Tire Sets': s.tire_sets_used,
                'Risk Score': round(s.risk_score, 2),
                'Success Prob': f"{s.probability_success:.1%}",
                'Windows': ', '.join([f"L{w.lap_optimal}" for w in s.pit_windows])
            })
        
        return pd.DataFrame(data)


def main():
    """Demo the pit window optimizer."""
    print("Pit Window Optimizer Demo")
    print("=" * 50)
    
    # 6-hour race at Spa
    optimizer = PitWindowOptimizer(
        race_duration_minutes=360,
        lap_time_seconds=138,  # Spa GT3 lap time
        pit_stop_time=65,
        fuel_capacity=120,
        fuel_consumption=3.2,
        tire_degradation_rate=0.025,
        safety_car_probability=0.20
    )
    
    print(f"\nRace Parameters:")
    print(f"  Duration: 6 hours")
    print(f"  Estimated laps: {optimizer.estimated_laps}")
    print(f"  Max stint (fuel): {optimizer.max_laps_on_fuel} laps")
    print(f"  Max stint (tires): {optimizer.max_laps_on_tires} laps")
    
    # Find optimal strategies
    strategies = optimizer.optimize(
        min_stops=2,
        max_stops=4,
        include_types=[StrategyType.BALANCED, StrategyType.AGGRESSIVE, StrategyType.CONSERVATIVE]
    )
    
    print(f"\n{'='*60}")
    print("Optimal Strategies (sorted by total time):")
    print(f"{'='*60}")
    
    for i, strategy in enumerate(strategies[:5], 1):
        print(f"\n{i}. {strategy}")
        print(f"   Total time: {strategy.total_race_time/60:.1f} min")
        print(f"   Risk: {strategy.risk_score:.2f}, Success: {strategy.probability_success:.1%}")
        for note in strategy.notes:
            print(f"   {note}")
    
    # Show as DataFrame
    print("\n" + "=" * 60)
    df = optimizer.to_dataframe(strategies[:5])
    print(df.to_string(index=False))
    
    # Undercut analysis
    print("\n" + "=" * 60)
    print("Undercut/Overcut Analysis (competitor pits lap 45):")
    undercut = optimizer.find_undercut_windows(45, our_position=2)
    print(f"  Recommendation: {undercut['recommendation']}")
    print(f"  Reason: {undercut['reason']}")
    if undercut['undercut_window']:
        print(f"  Undercut window: Laps {undercut['undercut_window']['lap_range']}")


if __name__ == '__main__':
    main()
