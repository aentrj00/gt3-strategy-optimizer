"""
Fuel Strategy Calculator
========================
Optimizes fuel strategy considering weight penalties, splash & dash options,
and variable fuel loads for different race conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FuelScenario:
    """Represents a fuel strategy scenario."""
    name: str
    fuel_load: float  # liters at start of stint
    stint_laps: int
    weight_penalty_total: float  # seconds lost to weight over stint
    time_in_pit: float  # refueling time
    end_fuel: float  # fuel remaining at pit
    total_time_cost: float  # weight penalty + pit time
    is_splash_dash: bool
    margin_laps: float  # extra laps possible with remaining fuel


@dataclass
class FuelPlan:
    """Complete fuel plan for the race."""
    total_fuel_used: float
    num_stops: int
    scenarios: List[FuelScenario]
    total_time: float
    total_weight_penalty: float
    total_pit_time: float
    efficiency_score: float  # 0-1, higher = better


class FuelStrategyCalculator:
    """
    Calculates optimal fuel strategies for endurance racing.
    
    Considers:
    - Fuel weight impact on lap times
    - Full tank vs partial fill trade-offs
    - Splash & dash strategies
    - Safety car fuel saving
    """
    
    def __init__(
        self,
        fuel_capacity: float = 120.0,  # liters
        fuel_consumption: float = 3.0,  # liters per lap
        fuel_flow_rate: float = 2.0,  # liters per second (refueling)
        fuel_density: float = 0.75,  # kg per liter
        weight_penalty: float = 0.035,  # seconds per lap per kg
        base_lap_time: float = 100.0,  # seconds
        pit_lane_time: float = 30.0,  # pit lane transit (not stationary)
        min_fuel_buffer: float = 2.0  # liters safety margin
    ):
        """
        Initialize the fuel calculator.
        
        Args:
            fuel_capacity: Maximum fuel tank capacity
            fuel_consumption: Fuel used per lap (average)
            fuel_flow_rate: Refueling rate in liters/second
            fuel_density: Fuel density for weight calculations
            weight_penalty: Lap time penalty per kg of fuel
            base_lap_time: Base lap time without penalties
            pit_lane_time: Time to traverse pit lane
            min_fuel_buffer: Minimum fuel to keep as safety margin
        """
        self.capacity = fuel_capacity
        self.consumption = fuel_consumption
        self.flow_rate = fuel_flow_rate
        self.density = fuel_density
        self.weight_penalty = weight_penalty
        self.base_lap_time = base_lap_time
        self.pit_lane_time = pit_lane_time
        self.min_buffer = min_fuel_buffer
        
        # Derived values
        self.max_stint_laps = int((fuel_capacity - min_fuel_buffer) / fuel_consumption)
        self.full_tank_weight = fuel_capacity * fuel_density
        
        logger.info(f"Fuel Calculator initialized:")
        logger.info(f"  - Max stint: {self.max_stint_laps} laps")
        logger.info(f"  - Full tank weight: {self.full_tank_weight:.1f} kg")
    
    def calculate_weight_penalty(
        self,
        start_fuel: float,
        num_laps: int
    ) -> Tuple[float, List[float]]:
        """
        Calculate cumulative weight penalty over a stint.
        
        Args:
            start_fuel: Starting fuel in liters
            num_laps: Number of laps in stint
            
        Returns:
            Tuple of (total_penalty, per_lap_penalties)
        """
        penalties = []
        total = 0
        fuel = start_fuel
        
        for lap in range(num_laps):
            # Fuel weight at start of lap
            fuel_weight = fuel * self.density
            
            # Penalty for this lap
            lap_penalty = fuel_weight * self.weight_penalty
            penalties.append(lap_penalty)
            total += lap_penalty
            
            # Consume fuel
            fuel -= self.consumption
        
        return total, penalties
    
    def calculate_refuel_time(self, fuel_to_add: float) -> float:
        """Calculate time to add fuel."""
        return fuel_to_add / self.flow_rate
    
    def optimize_fuel_load(
        self,
        required_laps: int,
        allow_splash_dash: bool = True
    ) -> List[FuelScenario]:
        """
        Find optimal fuel load for a given stint length.
        
        Compares:
        1. Full tank (more weight but fewer stops)
        2. Minimum fuel (less weight but tight margins)
        3. Optimal balance (minimize total time cost)
        
        Args:
            required_laps: Laps to complete in this stint
            allow_splash_dash: Whether to consider minimal fuel stops
            
        Returns:
            List of FuelScenario options, sorted by time cost
        """
        scenarios = []
        
        # Minimum fuel needed
        min_fuel = required_laps * self.consumption + self.min_buffer
        
        if min_fuel > self.capacity:
            logger.warning(f"Stint of {required_laps} laps exceeds fuel capacity!")
            return []
        
        # Scenario 1: Exact fuel (minimum weight)
        if allow_splash_dash or required_laps < self.max_stint_laps * 0.8:
            penalty, _ = self.calculate_weight_penalty(min_fuel, required_laps)
            refuel_time = self.calculate_refuel_time(min_fuel)
            
            scenarios.append(FuelScenario(
                name="Minimum Fuel",
                fuel_load=min_fuel,
                stint_laps=required_laps,
                weight_penalty_total=penalty,
                time_in_pit=refuel_time + self.pit_lane_time,
                end_fuel=self.min_buffer,
                total_time_cost=penalty + refuel_time,
                is_splash_dash=True,
                margin_laps=0
            ))
        
        # Scenario 2: Full tank
        penalty, _ = self.calculate_weight_penalty(self.capacity, required_laps)
        refuel_time = self.calculate_refuel_time(self.capacity)
        end_fuel = self.capacity - (required_laps * self.consumption)
        
        scenarios.append(FuelScenario(
            name="Full Tank",
            fuel_load=self.capacity,
            stint_laps=required_laps,
            weight_penalty_total=penalty,
            time_in_pit=refuel_time + self.pit_lane_time,
            end_fuel=end_fuel,
            total_time_cost=penalty + refuel_time,
            is_splash_dash=False,
            margin_laps=(end_fuel - self.min_buffer) / self.consumption
        ))
        
        # Scenario 3: Optimal (search for best balance)
        best_time = float('inf')
        best_fuel = min_fuel
        
        for fuel in np.arange(min_fuel, self.capacity + 1, 5):  # 5L increments
            penalty, _ = self.calculate_weight_penalty(fuel, required_laps)
            refuel_time = self.calculate_refuel_time(fuel)
            total_time = penalty + refuel_time
            
            if total_time < best_time:
                best_time = total_time
                best_fuel = fuel
        
        if best_fuel != min_fuel and best_fuel != self.capacity:
            penalty, _ = self.calculate_weight_penalty(best_fuel, required_laps)
            refuel_time = self.calculate_refuel_time(best_fuel)
            end_fuel = best_fuel - (required_laps * self.consumption)
            
            scenarios.append(FuelScenario(
                name="Optimal Balance",
                fuel_load=best_fuel,
                stint_laps=required_laps,
                weight_penalty_total=penalty,
                time_in_pit=refuel_time + self.pit_lane_time,
                end_fuel=end_fuel,
                total_time_cost=penalty + refuel_time,
                is_splash_dash=False,
                margin_laps=(end_fuel - self.min_buffer) / self.consumption
            ))
        
        # Sort by total time cost
        scenarios.sort(key=lambda s: s.total_time_cost)
        
        return scenarios
    
    def plan_race_fuel(
        self,
        race_laps: int,
        num_stops: int,
        optimize_distribution: bool = True
    ) -> FuelPlan:
        """
        Create a complete fuel plan for the race.
        
        Args:
            race_laps: Total race laps
            num_stops: Number of planned pit stops
            optimize_distribution: Whether to optimize stint lengths
            
        Returns:
            FuelPlan with all scenarios
        """
        num_stints = num_stops + 1
        
        if optimize_distribution:
            # Optimize stint lengths (longer early stints = more fuel weight saving)
            stints = self._optimize_stint_distribution(race_laps, num_stints)
        else:
            # Equal distribution
            base = race_laps // num_stints
            remainder = race_laps % num_stints
            stints = [base + (1 if i < remainder else 0) for i in range(num_stints)]
        
        # Generate fuel scenarios for each stint
        scenarios = []
        total_weight_penalty = 0
        total_pit_time = 0
        
        for i, stint_laps in enumerate(stints):
            stint_scenarios = self.optimize_fuel_load(stint_laps)
            
            if stint_scenarios:
                # Pick optimal scenario for this stint
                best = stint_scenarios[0]
                best.name = f"Stint {i+1}: {best.name}"
                scenarios.append(best)
                
                total_weight_penalty += best.weight_penalty_total
                if i < len(stints) - 1:  # No pit after final stint
                    total_pit_time += best.time_in_pit
        
        # Calculate efficiency
        baseline_penalty, _ = self.calculate_weight_penalty(
            self.capacity, 
            sum(s.stint_laps for s in scenarios)
        )
        efficiency = 1 - (total_weight_penalty / baseline_penalty) if baseline_penalty > 0 else 1.0
        
        return FuelPlan(
            total_fuel_used=sum(s.fuel_load for s in scenarios),
            num_stops=num_stops,
            scenarios=scenarios,
            total_time=sum(s.total_time_cost for s in scenarios),
            total_weight_penalty=total_weight_penalty,
            total_pit_time=total_pit_time,
            efficiency_score=efficiency
        )
    
    def _optimize_stint_distribution(
        self,
        total_laps: int,
        num_stints: int
    ) -> List[int]:
        """
        Optimize stint length distribution for fuel efficiency.
        
        Strategy: Slightly longer early stints (burn more fuel early)
        to reduce weight penalty in later stints.
        """
        # Start with longer stints early
        base = total_laps / num_stints
        stints = []
        
        for i in range(num_stints):
            # Taper from 105% to 95% of average
            factor = 1.05 - (0.10 * i / max(1, num_stints - 1))
            stint = int(base * factor)
            stints.append(min(stint, self.max_stint_laps))
        
        # Adjust to match total
        diff = total_laps - sum(stints)
        stints[-1] += diff
        
        # Ensure no stint exceeds max
        for i, stint in enumerate(stints):
            if stint > self.max_stint_laps:
                overflow = stint - self.max_stint_laps
                stints[i] = self.max_stint_laps
                # Redistribute overflow
                for j in range(len(stints)):
                    if j != i and stints[j] < self.max_stint_laps:
                        add = min(overflow, self.max_stint_laps - stints[j])
                        stints[j] += add
                        overflow -= add
                        if overflow <= 0:
                            break
        
        return stints
    
    def calculate_safety_car_savings(
        self,
        sc_laps: int,
        reduced_consumption_factor: float = 0.6
    ) -> Dict:
        """
        Calculate fuel saved during safety car periods.
        
        Args:
            sc_laps: Number of laps under SC/FCY
            reduced_consumption_factor: Fuel use ratio during SC (0.6 = 40% less)
            
        Returns:
            Dictionary with savings analysis
        """
        normal_consumption = sc_laps * self.consumption
        sc_consumption = sc_laps * self.consumption * reduced_consumption_factor
        fuel_saved = normal_consumption - sc_consumption
        
        # This fuel can extend the stint
        extra_laps = fuel_saved / self.consumption
        
        return {
            'sc_laps': sc_laps,
            'fuel_saved_liters': round(fuel_saved, 2),
            'extra_laps_possible': round(extra_laps, 1),
            'strategy_impact': 'May allow extending stint or reducing fuel load',
            'recommendation': f"If SC occurs, can extend by ~{int(extra_laps)} laps"
        }
    
    def splash_and_dash_analysis(
        self,
        current_fuel: float,
        laps_to_finish: int,
        pit_window_available: bool = True
    ) -> Dict:
        """
        Analyze whether splash & dash is beneficial.
        
        Args:
            current_fuel: Current fuel level
            laps_to_finish: Laps remaining in race
            pit_window_available: Whether there's a safe pit window
            
        Returns:
            Analysis dictionary
        """
        fuel_needed = laps_to_finish * self.consumption + self.min_buffer
        fuel_deficit = max(0, fuel_needed - current_fuel)
        
        if fuel_deficit <= 0:
            return {
                'splash_needed': False,
                'can_finish': True,
                'fuel_margin': current_fuel - fuel_needed,
                'recommendation': 'Sufficient fuel to finish'
            }
        
        # Splash & dash calculation
        splash_amount = fuel_deficit + 5  # Add small buffer
        splash_time = self.calculate_refuel_time(splash_amount)
        total_pit_time = splash_time + self.pit_lane_time
        
        # Compare with continuing (if somehow possible)
        analysis = {
            'splash_needed': True,
            'fuel_deficit': round(fuel_deficit, 1),
            'splash_amount': round(splash_amount, 1),
            'splash_time_seconds': round(splash_time, 1),
            'total_stop_time': round(total_pit_time, 1),
            'pit_window_safe': pit_window_available
        }
        
        if pit_window_available:
            analysis['recommendation'] = f"Splash {splash_amount:.0f}L - adds {total_pit_time:.0f}s to race"
        else:
            analysis['recommendation'] = "WARNING: No safe pit window - fuel save required!"
            analysis['fuel_save_needed_per_lap'] = round(fuel_deficit / laps_to_finish, 3)
        
        return analysis
    
    def to_dataframe(self, scenarios: List[FuelScenario]) -> pd.DataFrame:
        """Convert fuel scenarios to DataFrame."""
        data = []
        for s in scenarios:
            data.append({
                'Scenario': s.name,
                'Fuel Load (L)': round(s.fuel_load, 1),
                'Stint Laps': s.stint_laps,
                'Weight Penalty (s)': round(s.weight_penalty_total, 2),
                'Pit Time (s)': round(s.time_in_pit, 1),
                'Total Cost (s)': round(s.total_time_cost, 2),
                'End Fuel (L)': round(s.end_fuel, 1),
                'Margin Laps': round(s.margin_laps, 1)
            })
        return pd.DataFrame(data)


def main():
    """Demo fuel strategy calculator."""
    print("Fuel Strategy Calculator Demo")
    print("=" * 50)
    
    calc = FuelStrategyCalculator(
        fuel_capacity=120,
        fuel_consumption=3.2,
        fuel_flow_rate=2.0,
        base_lap_time=138,  # Spa
        weight_penalty=0.035
    )
    
    print(f"\nVehicle Parameters:")
    print(f"  - Fuel capacity: {calc.capacity}L")
    print(f"  - Consumption: {calc.consumption}L/lap")
    print(f"  - Max stint: {calc.max_stint_laps} laps")
    print(f"  - Full tank weight: {calc.full_tank_weight:.1f}kg")
    
    # Optimize a 35-lap stint
    print("\n" + "=" * 50)
    print("Fuel Options for 35-lap stint:")
    scenarios = calc.optimize_fuel_load(35)
    df = calc.to_dataframe(scenarios)
    print(df.to_string(index=False))
    
    # Complete race plan
    print("\n" + "=" * 50)
    print("Complete Race Fuel Plan (156 laps, 4 stops):")
    plan = calc.plan_race_fuel(156, num_stops=4)
    
    print(f"\nTotal fuel used: {plan.total_fuel_used:.1f}L")
    print(f"Total weight penalty: {plan.total_weight_penalty:.2f}s")
    print(f"Total pit time: {plan.total_pit_time:.1f}s")
    print(f"Efficiency score: {plan.efficiency_score:.1%}")
    
    print("\nStint breakdown:")
    for s in plan.scenarios:
        print(f"  - {s.name}: {s.fuel_load:.0f}L for {s.stint_laps} laps")
    
    # Safety car analysis
    print("\n" + "=" * 50)
    print("Safety Car Fuel Savings (5 laps under SC):")
    sc_savings = calc.calculate_safety_car_savings(5)
    print(f"  - Fuel saved: {sc_savings['fuel_saved_liters']}L")
    print(f"  - Extra laps possible: {sc_savings['extra_laps_possible']}")
    print(f"  - {sc_savings['recommendation']}")


if __name__ == '__main__':
    main()
