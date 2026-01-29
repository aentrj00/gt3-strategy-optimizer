"""
Race Simulator
=============
Monte Carlo simulation for race strategy comparison.

Simulates complete races with:
- Lap-by-lap progression
- Fuel and tire degradation
- Safety car events
- Position changes
- Multiple strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackStatus(Enum):
    """Current track status."""
    GREEN = "green"
    YELLOW = "yellow"  # Local yellow
    SAFETY_CAR = "safety_car"
    FCY = "full_course_yellow"
    RED = "red"


@dataclass
class CarState:
    """Current state of a car during simulation."""
    car_id: int
    position: int
    lap: int
    total_time: float
    fuel: float
    tire_life: float
    stint: int
    last_lap_time: float
    pits_completed: int
    status: str = "running"  # running, dnf, finished
    gap_to_leader: float = 0.0
    
    
@dataclass
class RaceState:
    """Current state of the entire race."""
    current_lap: int
    track_status: TrackStatus
    safety_car_laps: int
    total_safety_cars: int
    cars: List[CarState]
    weather: str
    
    
@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    final_position: int
    total_time: float
    laps_completed: int
    pits_made: int
    safety_cars_encountered: int
    dnf: bool
    gap_to_winner: float
    position_history: List[int]
    lap_times: List[float]


@dataclass  
class StrategyResult:
    """Aggregated results for a strategy across all simulations."""
    strategy_name: str
    simulations: int
    avg_position: float
    position_std: float
    win_probability: float
    podium_probability: float
    top5_probability: float
    top10_probability: float
    dnf_rate: float
    avg_time: float
    position_distribution: Dict[int, float]


class RaceSimulator:
    """
    Monte Carlo race simulator for strategy evaluation.
    
    Runs thousands of race simulations to evaluate strategy
    performance under various conditions.
    """
    
    def __init__(
        self,
        num_cars: int = 30,
        race_laps: int = 156,
        base_lap_time: float = 138.0,
        fuel_capacity: float = 120.0,
        fuel_consumption: float = 3.2,
        tire_deg_rate: float = 0.025,
        pit_stop_time: float = 65.0,
        safety_car_probability: float = 0.15,  # per hour
        dnf_probability: float = 0.05,  # per car per race
        random_seed: int = None
    ):
        """
        Initialize the race simulator.
        
        Args:
            num_cars: Number of cars in race
            race_laps: Total race laps
            base_lap_time: Base lap time (seconds)
            fuel_capacity: Fuel tank size (liters)
            fuel_consumption: Fuel per lap (liters)
            tire_deg_rate: Tire degradation per lap
            pit_stop_time: Total pit stop time (seconds)
            safety_car_probability: SC probability per hour
            dnf_probability: DNF chance per car
            random_seed: Random seed for reproducibility
        """
        self.num_cars = num_cars
        self.race_laps = race_laps
        self.base_lap_time = base_lap_time
        self.fuel_capacity = fuel_capacity
        self.fuel_consumption = fuel_consumption
        self.tire_deg_rate = tire_deg_rate
        self.pit_stop_time = pit_stop_time
        self.sc_probability = safety_car_probability
        self.dnf_probability = dnf_probability
        
        if random_seed:
            np.random.seed(random_seed)
        
        # Calculate race duration for SC probability
        self.race_hours = (race_laps * base_lap_time) / 3600
        
        logger.info(f"Race Simulator initialized:")
        logger.info(f"  - {num_cars} cars, {race_laps} laps")
        logger.info(f"  - Expected duration: {self.race_hours:.1f} hours")
    
    def simulate_race(
        self,
        our_strategy: Dict,
        grid_position: int = 15,
        driver_skill: float = 1.0
    ) -> SimulationResult:
        """
        Simulate a single race.
        
        Args:
            our_strategy: Dictionary with pit_laps, fuel_loads, tire_compounds
            grid_position: Starting grid position
            driver_skill: Skill factor (1.0 = average, <1.0 = faster)
            
        Returns:
            SimulationResult with race outcome
        """
        # Initialize cars
        cars = self._initialize_cars(grid_position, driver_skill, our_strategy)
        
        # Track state
        track_status = TrackStatus.GREEN
        sc_laps_remaining = 0
        total_scs = 0
        
        position_history = []
        lap_times = []
        
        # Simulate lap by lap
        for lap in range(1, self.race_laps + 1):
            # Check for safety car
            if track_status == TrackStatus.GREEN:
                if self._should_safety_car_occur(lap):
                    track_status = TrackStatus.SAFETY_CAR
                    sc_laps_remaining = np.random.randint(4, 10)
                    total_scs += 1
            elif sc_laps_remaining > 0:
                sc_laps_remaining -= 1
                if sc_laps_remaining == 0:
                    track_status = TrackStatus.GREEN
            
            # Simulate each car's lap
            for car in cars:
                if car.status != "running":
                    continue
                
                # Check for DNF
                if np.random.random() < self.dnf_probability / self.race_laps:
                    car.status = "dnf"
                    continue
                
                # Check for pit stop
                is_pit_lap = self._should_pit(car, lap, our_strategy if car.car_id == 0 else None)
                
                if is_pit_lap:
                    pit_time = self._calculate_pit_time(car, track_status)
                    car.total_time += pit_time
                    car.pits_completed += 1
                    car.fuel = self.fuel_capacity
                    car.tire_life = 1.0
                    car.stint += 1
                
                # Calculate lap time
                lap_time = self._calculate_lap_time(
                    car, 
                    track_status,
                    is_pit_lap
                )
                
                car.last_lap_time = lap_time
                car.total_time += lap_time
                car.lap = lap
                
                # Consume resources
                if track_status == TrackStatus.SAFETY_CAR:
                    car.fuel -= self.fuel_consumption * 0.6
                    car.tire_life -= self.tire_deg_rate * 0.3
                else:
                    car.fuel -= self.fuel_consumption
                    car.tire_life -= self.tire_deg_rate
                
                # Check for running out
                if car.fuel <= 0:
                    car.status = "dnf"
                
                if car.car_id == 0:
                    lap_times.append(lap_time)
            
            # Update positions
            self._update_positions(cars)
            
            # Record our position
            our_car = next(c for c in cars if c.car_id == 0)
            position_history.append(our_car.position)
        
        # Get final result for our car
        our_car = next(c for c in cars if c.car_id == 0)
        
        if our_car.status == "dnf":
            return SimulationResult(
                final_position=self.num_cars,
                total_time=float('inf'),
                laps_completed=our_car.lap,
                pits_made=our_car.pits_completed,
                safety_cars_encountered=total_scs,
                dnf=True,
                gap_to_winner=float('inf'),
                position_history=position_history,
                lap_times=lap_times
            )
        
        # Find winner's time
        winner = min([c for c in cars if c.status == "running"], 
                     key=lambda x: x.total_time, default=our_car)
        
        return SimulationResult(
            final_position=our_car.position,
            total_time=our_car.total_time,
            laps_completed=our_car.lap,
            pits_made=our_car.pits_completed,
            safety_cars_encountered=total_scs,
            dnf=False,
            gap_to_winner=our_car.total_time - winner.total_time,
            position_history=position_history,
            lap_times=lap_times
        )
    
    def _initialize_cars(
        self, 
        our_grid_position: int,
        our_skill: float,
        our_strategy: Dict
    ) -> List[CarState]:
        """Initialize all cars for race start."""
        cars = []
        
        for i in range(self.num_cars):
            if i == 0:  # Our car
                position = our_grid_position
                skill = our_skill
            else:
                position = i + 1 if i < our_grid_position else i
                skill = np.random.uniform(0.98, 1.02)  # Random skill
            
            cars.append(CarState(
                car_id=i,
                position=position,
                lap=0,
                total_time=position * 0.5,  # Stagger start
                fuel=self.fuel_capacity,
                tire_life=1.0,
                stint=1,
                last_lap_time=self.base_lap_time,
                pits_completed=0
            ))
        
        return cars
    
    def _should_safety_car_occur(self, lap: int) -> bool:
        """Determine if a safety car should occur this lap."""
        # SC probability per lap based on hourly rate
        laps_per_hour = 3600 / self.base_lap_time
        prob_per_lap = self.sc_probability / laps_per_hour
        
        # Slightly higher probability in first few laps
        if lap <= 3:
            prob_per_lap *= 1.5
        
        return np.random.random() < prob_per_lap
    
    def _should_pit(
        self, 
        car: CarState, 
        current_lap: int,
        strategy: Dict = None
    ) -> bool:
        """Determine if car should pit this lap."""
        if strategy and car.car_id == 0:
            # Our car follows strategy
            pit_laps = strategy.get('pit_laps', [])
            return current_lap in pit_laps
        else:
            # AI cars pit based on fuel/tire
            fuel_critical = car.fuel < self.fuel_consumption * 2
            tire_critical = car.tire_life < 0.25
            return fuel_critical or tire_critical
    
    def _calculate_pit_time(
        self,
        car: CarState,
        track_status: TrackStatus
    ) -> float:
        """Calculate total pit stop time."""
        base_time = self.pit_stop_time
        
        # Random variation (±3 seconds)
        variation = np.random.uniform(-3, 3)
        
        # Faster pit lane under SC
        if track_status == TrackStatus.SAFETY_CAR:
            base_time *= 0.9  # Pit lane not as relatively slow
        
        return base_time + variation
    
    def _calculate_lap_time(
        self,
        car: CarState,
        track_status: TrackStatus,
        is_pit_lap: bool
    ) -> float:
        """Calculate lap time with all factors."""
        lap_time = self.base_lap_time
        
        # Skill factor
        skill_factor = 1.0 + (car.car_id % 5) * 0.005  # Spread out skill
        if car.car_id == 0:
            skill_factor = 1.0  # Our car is average
        lap_time *= skill_factor
        
        # Fuel weight penalty
        fuel_weight = car.fuel * 0.75  # kg
        lap_time += fuel_weight * 0.0003  # 0.3ms per kg
        
        # Tire degradation penalty
        if car.tire_life > 0.3:
            tire_penalty = (1 - car.tire_life) * 2
        else:
            tire_penalty = 2 + (0.3 - car.tire_life) * 10
        lap_time += tire_penalty
        
        # Track status
        if track_status == TrackStatus.SAFETY_CAR:
            lap_time *= 1.4
        elif track_status == TrackStatus.FCY:
            lap_time *= 1.3
        
        # Pit in/out laps
        if is_pit_lap:
            lap_time += 5  # Extra time for pit entry
        
        # Random variation (±1%)
        lap_time *= np.random.uniform(0.99, 1.01)
        
        return lap_time
    
    def _update_positions(self, cars: List[CarState]) -> None:
        """Update race positions based on total time."""
        # Sort by total time (DNF cars last)
        running_cars = [(c, c.total_time) for c in cars if c.status == "running"]
        dnf_cars = [c for c in cars if c.status != "running"]
        
        running_cars.sort(key=lambda x: x[1])
        
        for pos, (car, _) in enumerate(running_cars, 1):
            car.position = pos
            car.gap_to_leader = car.total_time - running_cars[0][1] if running_cars else 0
        
        for i, car in enumerate(dnf_cars):
            car.position = len(running_cars) + i + 1
    
    def monte_carlo(
        self,
        strategy: Dict,
        grid_position: int = 15,
        driver_skill: float = 1.0,
        num_simulations: int = 1000,
        parallel: bool = True
    ) -> StrategyResult:
        """
        Run Monte Carlo simulation for a strategy.
        
        Args:
            strategy: Strategy dictionary
            grid_position: Starting position
            driver_skill: Driver skill factor
            num_simulations: Number of simulations to run
            parallel: Use parallel processing
            
        Returns:
            StrategyResult with aggregated statistics
        """
        results = []
        
        logger.info(f"Running {num_simulations} simulations for strategy '{strategy.get('name', 'Unknown')}'...")
        
        for i in range(num_simulations):
            if i % 100 == 0:
                logger.debug(f"Simulation {i}/{num_simulations}")
            
            result = self.simulate_race(
                strategy,
                grid_position,
                driver_skill
            )
            results.append(result)
        
        return self._aggregate_results(results, strategy.get('name', 'Unknown'))
    
    def _aggregate_results(
        self,
        results: List[SimulationResult],
        strategy_name: str
    ) -> StrategyResult:
        """Aggregate simulation results into statistics."""
        positions = [r.final_position for r in results if not r.dnf]
        all_positions = [r.final_position for r in results]
        times = [r.total_time for r in results if not r.dnf]
        dnfs = sum(1 for r in results if r.dnf)
        
        # Position distribution
        position_counts = {}
        for pos in all_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        position_dist = {k: v / len(results) for k, v in position_counts.items()}
        
        return StrategyResult(
            strategy_name=strategy_name,
            simulations=len(results),
            avg_position=np.mean(all_positions),
            position_std=np.std(all_positions),
            win_probability=sum(1 for p in positions if p == 1) / len(results),
            podium_probability=sum(1 for p in positions if p <= 3) / len(results),
            top5_probability=sum(1 for p in positions if p <= 5) / len(results),
            top10_probability=sum(1 for p in positions if p <= 10) / len(results),
            dnf_rate=dnfs / len(results),
            avg_time=np.mean(times) if times else float('inf'),
            position_distribution=position_dist
        )
    
    def compare_strategies(
        self,
        strategies: List[Dict],
        grid_position: int = 15,
        num_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: List of strategy dictionaries
            grid_position: Starting position
            num_simulations: Simulations per strategy
            
        Returns:
            DataFrame comparing strategy performance
        """
        results = []
        
        for strategy in strategies:
            result = self.monte_carlo(
                strategy,
                grid_position,
                num_simulations=num_simulations
            )
            results.append(result)
        
        # Create comparison DataFrame
        data = []
        for r in results:
            data.append({
                'Strategy': r.strategy_name,
                'Avg Position': round(r.avg_position, 2),
                'Std Dev': round(r.position_std, 2),
                'Win %': f"{r.win_probability:.1%}",
                'Podium %': f"{r.podium_probability:.1%}",
                'Top 5 %': f"{r.top5_probability:.1%}",
                'Top 10 %': f"{r.top10_probability:.1%}",
                'DNF %': f"{r.dnf_rate:.1%}"
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Avg Position')
    
    def position_probability_chart(
        self,
        result: StrategyResult,
        max_position: int = 20
    ) -> pd.DataFrame:
        """Generate position probability data for charting."""
        data = []
        cumulative = 0
        
        for pos in range(1, max_position + 1):
            prob = result.position_distribution.get(pos, 0)
            cumulative += prob
            data.append({
                'Position': pos,
                'Probability': prob,
                'Cumulative': cumulative
            })
        
        return pd.DataFrame(data)


def main():
    """Demo race simulator."""
    print("Race Simulator Demo")
    print("=" * 50)
    
    simulator = RaceSimulator(
        num_cars=30,
        race_laps=156,  # ~6h at Spa
        base_lap_time=138,
        fuel_capacity=120,
        fuel_consumption=3.2,
        pit_stop_time=65,
        safety_car_probability=0.20,
        random_seed=42
    )
    
    # Define strategies to compare
    strategies = [
        {
            'name': '4-Stop Balanced',
            'pit_laps': [31, 62, 93, 124],
            'fuel_loads': [120, 120, 120, 120, 120]
        },
        {
            'name': '3-Stop Aggressive',
            'pit_laps': [39, 78, 117],
            'fuel_loads': [120, 120, 120, 120]
        },
        {
            'name': '5-Stop Conservative',
            'pit_laps': [26, 52, 78, 104, 130],
            'fuel_loads': [100, 100, 100, 100, 100, 100]
        }
    ]
    
    # Run single simulation
    print("\nSingle Race Simulation (4-Stop Balanced):")
    result = simulator.simulate_race(
        strategies[0],
        grid_position=12,
        driver_skill=0.99
    )
    
    print(f"  Final Position: P{result.final_position}")
    print(f"  Laps Completed: {result.laps_completed}")
    print(f"  Pit Stops: {result.pits_made}")
    print(f"  Safety Cars: {result.safety_cars_encountered}")
    print(f"  Gap to Winner: {result.gap_to_winner:.1f}s")
    
    # Monte Carlo comparison
    print("\n" + "=" * 50)
    print("Monte Carlo Simulation (500 runs per strategy):")
    
    comparison = simulator.compare_strategies(
        strategies,
        grid_position=12,
        num_simulations=500
    )
    
    print(comparison.to_string(index=False))
    
    # Detailed results for best strategy
    print("\n" + "=" * 50)
    print("Detailed Analysis - 4-Stop Balanced:")
    
    detailed = simulator.monte_carlo(
        strategies[0],
        grid_position=12,
        num_simulations=500
    )
    
    prob_chart = simulator.position_probability_chart(detailed)
    print("\nPosition Probability Distribution:")
    print(prob_chart.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
