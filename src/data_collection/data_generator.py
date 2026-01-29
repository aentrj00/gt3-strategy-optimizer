"""
Race Data Generator
==================
Generates realistic sample race data for testing and demonstration.
Based on real-world GT3/GTE racing patterns from WEC and IMSA.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import random


class RaceDataGenerator:
    """
    Generates realistic race data including:
    - Lap times with degradation
    - Pit stops
    - Safety car periods
    - Weather changes
    - Multi-class racing scenarios
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data generator.
        
        Args:
            config_path: Path to configuration files directory
        """
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent.parent / 'config'
        self.circuits = self._load_config('circuits.yaml')
        self.cars = self._load_config('cars.yaml')
        
        # Random seed for reproducibility
        np.random.seed(42)
        
    def _load_config(self, filename: str) -> Dict:
        """Load YAML configuration file."""
        config_file = self.config_path / filename
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def generate_race(
        self,
        circuit: str = 'spa',
        duration_hours: float = 6.0,
        num_cars: int = 30,
        classes: List[str] = None,
        weather_changes: bool = True,
        include_safety_cars: bool = True,
        seed: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete race dataset.
        
        Args:
            circuit: Circuit identifier from config
            duration_hours: Race duration in hours
            num_cars: Number of cars in the race
            classes: List of classes (default: ['LMGT3', 'LMP2', 'Hypercar'])
            weather_changes: Whether to include weather changes
            include_safety_cars: Whether to include safety car periods
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing DataFrames:
            - lap_times: All lap times for all cars
            - pit_stops: All pit stop events
            - safety_cars: Safety car periods
            - weather: Weather conditions throughout race
            - race_info: General race information
        """
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            
        if classes is None:
            classes = ['LMGT3', 'LMP2', 'Hypercar']
            
        circuit_config = self.circuits.get('circuits', {}).get(circuit, self._default_circuit())
        
        # Calculate race parameters
        base_lap_time = circuit_config['lap_time_gt3_avg']
        total_laps_estimate = int((duration_hours * 3600) / base_lap_time)
        
        # Generate entries
        entries = self._generate_entries(num_cars, classes, circuit_config)
        
        # Generate weather
        weather_df = self._generate_weather(duration_hours, weather_changes)
        
        # Generate safety car periods
        safety_cars_df = self._generate_safety_cars(
            duration_hours, 
            circuit_config['safety_car_probability'],
            include_safety_cars
        )
        
        # Generate lap times for each car
        all_lap_times = []
        all_pit_stops = []
        
        for entry in entries:
            car_laps, car_pits = self._generate_car_race(
                entry,
                circuit_config,
                duration_hours,
                weather_df,
                safety_cars_df
            )
            all_lap_times.append(car_laps)
            all_pit_stops.append(car_pits)
            
        lap_times_df = pd.concat(all_lap_times, ignore_index=True)
        pit_stops_df = pd.concat(all_pit_stops, ignore_index=True)
        
        # Create race info
        race_info = pd.DataFrame([{
            'circuit': circuit,
            'circuit_name': circuit_config['name'],
            'country': circuit_config['country'],
            'duration_hours': duration_hours,
            'track_length_km': circuit_config['length_km'],
            'num_entries': num_cars,
            'classes': ','.join(classes),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_laps_leader': lap_times_df.groupby('car_number')['lap_number'].max().max()
        }])
        
        return {
            'lap_times': lap_times_df,
            'pit_stops': pit_stops_df,
            'safety_cars': safety_cars_df,
            'weather': weather_df,
            'race_info': race_info,
            'entries': pd.DataFrame(entries)
        }
    
    def _default_circuit(self) -> Dict:
        """Return default circuit configuration."""
        return {
            'name': 'Generic Circuit',
            'country': 'Unknown',
            'length_km': 5.0,
            'lap_time_gt3_avg': 100,
            'fuel_consumption_avg': 2.5,
            'tire_wear': 'medium',
            'safety_car_probability': 0.15,
            'pit_lane_time': 30
        }
    
    def _generate_entries(
        self, 
        num_cars: int, 
        classes: List[str],
        circuit_config: Dict
    ) -> List[Dict]:
        """Generate car entries for the race."""
        entries = []
        
        # Get available cars from config
        available_cars = list(self.cars.get('cars', {}).keys())
        if not available_cars:
            available_cars = ['generic_gt3']
            
        # Distribute cars across classes
        class_distribution = {
            'Hypercar': 0.20,
            'LMP2': 0.25,
            'LMGT3': 0.55
        }
        
        # Team names for variety
        team_prefixes = ['Racing', 'Motorsport', 'Competition', 'Performance', 'Endurance']
        team_suffixes = ['Team', 'Racing', 'Sport', 'Works', 'Factory']
        
        car_number = 1
        for class_name in classes:
            ratio = class_distribution.get(class_name, 0.33)
            num_class_cars = max(1, int(num_cars * ratio))
            
            for i in range(num_class_cars):
                if car_number > num_cars:
                    break
                    
                car_model = random.choice(available_cars)
                car_config = self.cars.get('cars', {}).get(car_model, {})
                
                # Generate realistic car number
                if class_name == 'Hypercar':
                    car_num = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 83, 93, 94])
                elif class_name == 'LMP2':
                    car_num = random.randint(20, 49)
                else:  # GT3
                    car_num = random.randint(50, 99)
                
                # Ensure unique car numbers
                while any(e['car_number'] == car_num for e in entries):
                    car_num = random.randint(1, 199)
                
                entry = {
                    'car_number': car_num,
                    'class': class_name,
                    'car_model': car_config.get('model', 'Generic GT3'),
                    'manufacturer': car_config.get('manufacturer', 'Unknown'),
                    'team': f"{random.choice(team_prefixes)} {random.choice(team_suffixes)}",
                    'fuel_capacity': car_config.get('fuel', {}).get('capacity_liters', 120),
                    'fuel_consumption': circuit_config['fuel_consumption_avg'] * (
                        0.85 if class_name == 'Hypercar' else
                        0.95 if class_name == 'LMP2' else 1.0
                    ),
                    'base_lap_time': circuit_config['lap_time_gt3_avg'] * (
                        0.88 if class_name == 'Hypercar' else
                        0.94 if class_name == 'LMP2' else 1.0
                    ),
                    'driver_skill': np.random.uniform(0.98, 1.02),  # ±2% skill variance
                    'tire_degradation_factor': np.random.uniform(0.95, 1.05)
                }
                entries.append(entry)
                car_number += 1
                
        return entries
    
    def _generate_weather(
        self, 
        duration_hours: float,
        include_changes: bool
    ) -> pd.DataFrame:
        """Generate weather conditions throughout the race."""
        # Generate weather every 5 minutes
        intervals = int(duration_hours * 12)
        
        weather_data = []
        current_temp = np.random.uniform(18, 32)  # Track temp
        current_condition = 'dry'
        rain_probability = 0.15 if include_changes else 0
        
        for i in range(intervals):
            time_minutes = i * 5
            
            # Temperature variation
            current_temp += np.random.uniform(-0.5, 0.5)
            current_temp = np.clip(current_temp, 15, 45)
            
            # Weather change
            if include_changes and np.random.random() < rain_probability / 12:
                conditions = ['dry', 'damp', 'wet', 'heavy_rain']
                current_idx = conditions.index(current_condition)
                # Weather tends to change gradually
                new_idx = np.clip(current_idx + np.random.choice([-1, 1]), 0, 3)
                current_condition = conditions[new_idx]
            
            weather_data.append({
                'time_minutes': time_minutes,
                'track_temp_c': round(current_temp, 1),
                'air_temp_c': round(current_temp - 8, 1),
                'condition': current_condition,
                'humidity_pct': np.random.uniform(40, 80),
                'wind_speed_kph': np.random.uniform(5, 25)
            })
            
        return pd.DataFrame(weather_data)
    
    def _generate_safety_cars(
        self,
        duration_hours: float,
        base_probability: float,
        include: bool
    ) -> pd.DataFrame:
        """Generate safety car periods."""
        if not include:
            return pd.DataFrame(columns=['start_minute', 'end_minute', 'type', 'cause'])
            
        safety_cars = []
        race_minutes = int(duration_hours * 60)
        
        # Probability per hour
        expected_scs = base_probability * duration_hours
        num_scs = np.random.poisson(expected_scs)
        
        # Generate SC periods
        sc_starts = sorted(np.random.uniform(10, race_minutes - 20, num_scs))
        
        causes = [
            'Crash', 'Spin', 'Mechanical failure', 'Debris on track',
            'Weather', 'Oil on track', 'Barrier repair', 'Medical'
        ]
        
        for start in sc_starts:
            # SC duration: 5-15 minutes typically
            duration = np.random.uniform(5, 15)
            sc_type = np.random.choice(['Full Course Yellow', 'Safety Car'], p=[0.4, 0.6])
            
            safety_cars.append({
                'start_minute': round(start, 1),
                'end_minute': round(start + duration, 1),
                'duration_minutes': round(duration, 1),
                'type': sc_type,
                'cause': np.random.choice(causes)
            })
            
        return pd.DataFrame(safety_cars)
    
    def _generate_car_race(
        self,
        entry: Dict,
        circuit_config: Dict,
        duration_hours: float,
        weather_df: pd.DataFrame,
        safety_cars_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete race data for a single car."""
        
        base_lap_time = entry['base_lap_time']
        fuel_consumption = entry['fuel_consumption']
        fuel_capacity = entry['fuel_capacity']
        driver_skill = entry['driver_skill']
        
        laps = []
        pit_stops = []
        
        current_fuel = fuel_capacity
        current_tire_life = 1.0  # 100%
        stint_lap = 0
        current_stint = 1
        current_driver = 1
        total_time_elapsed = 0
        lap_number = 0
        race_duration_seconds = duration_hours * 3600
        
        while total_time_elapsed < race_duration_seconds:
            lap_number += 1
            stint_lap += 1
            
            # Get current weather
            current_minute = total_time_elapsed / 60
            weather_idx = min(int(current_minute / 5), len(weather_df) - 1)
            current_weather = weather_df.iloc[weather_idx]
            
            # Check if under safety car
            under_sc = False
            sc_type = None
            for _, sc in safety_cars_df.iterrows():
                if sc['start_minute'] <= current_minute <= sc['end_minute']:
                    under_sc = True
                    sc_type = sc['type']
                    break
            
            # Calculate lap time
            lap_time = self._calculate_lap_time(
                base_lap_time=base_lap_time,
                driver_skill=driver_skill,
                fuel_load=current_fuel,
                tire_life=current_tire_life,
                weather=current_weather,
                under_sc=under_sc,
                sc_type=sc_type,
                stint_lap=stint_lap
            )
            
            # Fuel consumption
            fuel_used = fuel_consumption * (1 + np.random.uniform(-0.05, 0.05))
            current_fuel -= fuel_used
            
            # Tire degradation
            tire_wear_rate = self._get_tire_wear_rate(
                circuit_config['tire_wear'],
                current_weather['condition'],
                entry['tire_degradation_factor']
            )
            current_tire_life -= tire_wear_rate
            
            # Record lap
            laps.append({
                'car_number': entry['car_number'],
                'lap_number': lap_number,
                'lap_time_seconds': round(lap_time, 3),
                'sector_1': round(lap_time * 0.33 + np.random.uniform(-0.5, 0.5), 3),
                'sector_2': round(lap_time * 0.35 + np.random.uniform(-0.5, 0.5), 3),
                'sector_3': round(lap_time * 0.32 + np.random.uniform(-0.5, 0.5), 3),
                'stint': current_stint,
                'stint_lap': stint_lap,
                'driver_number': current_driver,
                'fuel_remaining': round(current_fuel, 1),
                'tire_life_pct': round(current_tire_life * 100, 1),
                'track_status': 'SC' if under_sc else 'Green',
                'weather': current_weather['condition'],
                'track_temp_c': current_weather['track_temp_c'],
                'elapsed_time_seconds': round(total_time_elapsed + lap_time, 3),
                'class': entry['class']
            })
            
            total_time_elapsed += lap_time
            
            # Check for pit stop need
            need_fuel = current_fuel < fuel_consumption * 2
            need_tires = current_tire_life < 0.25
            
            # Optimal pit window (for endurance)
            optimal_stint_length = fuel_capacity / fuel_consumption * 0.95
            approaching_window = stint_lap > optimal_stint_length * 0.85
            
            # Pit stop decision
            if need_fuel or need_tires or (approaching_window and not under_sc):
                if total_time_elapsed + 60 < race_duration_seconds:  # Not at end of race
                    pit_time = self._generate_pit_stop(
                        entry, 
                        circuit_config,
                        current_fuel,
                        current_tire_life
                    )
                    
                    pit_stops.append({
                        'car_number': entry['car_number'],
                        'lap_number': lap_number,
                        'pit_time_seconds': round(pit_time, 3),
                        'fuel_added': round(fuel_capacity - current_fuel, 1),
                        'tire_change': current_tire_life < 0.5,
                        'driver_change': np.random.random() < 0.3,  # 30% chance
                        'stint_completed': current_stint,
                        'elapsed_time_seconds': round(total_time_elapsed, 3)
                    })
                    
                    total_time_elapsed += pit_time
                    current_fuel = fuel_capacity
                    current_tire_life = 1.0
                    stint_lap = 0
                    current_stint += 1
                    if pit_stops[-1]['driver_change']:
                        current_driver = (current_driver % 3) + 1  # Rotate drivers
        
        return pd.DataFrame(laps), pd.DataFrame(pit_stops)
    
    def _calculate_lap_time(
        self,
        base_lap_time: float,
        driver_skill: float,
        fuel_load: float,
        tire_life: float,
        weather: pd.Series,
        under_sc: bool,
        sc_type: str,
        stint_lap: int
    ) -> float:
        """Calculate realistic lap time with all factors."""
        
        lap_time = base_lap_time
        
        # Driver skill
        lap_time *= driver_skill
        
        # Fuel effect (heavier = slower, ~0.03s per kg)
        fuel_weight_effect = fuel_load * 0.75 * 0.00025  # Convert to time penalty
        lap_time += fuel_weight_effect
        
        # Tire degradation (exponential near cliff)
        if tire_life > 0.5:
            tire_penalty = (1 - tire_life) * 2  # Linear degradation
        else:
            tire_penalty = 2 + (0.5 - tire_life) * 8  # Exponential cliff
        lap_time += tire_penalty
        
        # Weather effect
        weather_conditions = self.cars.get('weather_conditions', {})
        condition = weather.get('condition', 'dry')
        weather_config = weather_conditions.get(condition, {'grip_modifier': 1.0})
        lap_time *= (2 - weather_config.get('grip_modifier', 1.0))
        
        # Temperature effect (optimal around 25-30°C track temp)
        temp = weather.get('track_temp_c', 25)
        temp_penalty = abs(temp - 28) * 0.02
        lap_time += temp_penalty
        
        # Safety car
        if under_sc:
            if sc_type == 'Full Course Yellow':
                lap_time *= 1.4  # 40% slower
            else:
                lap_time *= 1.5  # 50% slower
        
        # Random variation (±1%)
        lap_time *= np.random.uniform(0.99, 1.01)
        
        # Stint warmup (first 2 laps slightly slower)
        if stint_lap <= 2:
            lap_time *= 1.01
            
        return lap_time
    
    def _get_tire_wear_rate(
        self, 
        track_wear: str, 
        weather: str,
        car_factor: float
    ) -> float:
        """Calculate tire wear rate per lap."""
        base_rates = {
            'low': 0.015,
            'medium': 0.025,
            'high': 0.035
        }
        
        weather_modifiers = {
            'dry': 1.0,
            'damp': 0.9,
            'wet': 0.7,
            'heavy_rain': 0.6
        }
        
        base = base_rates.get(track_wear, 0.025)
        weather_mod = weather_modifiers.get(weather, 1.0)
        
        return base * weather_mod * car_factor
    
    def _generate_pit_stop(
        self,
        entry: Dict,
        circuit_config: Dict,
        current_fuel: float,
        current_tire_life: float
    ) -> float:
        """Generate realistic pit stop time."""
        
        pit_lane_time = circuit_config['pit_lane_time']
        
        # Base stationary time
        stationary_time = 0
        
        # Fuel time
        fuel_needed = entry['fuel_capacity'] - current_fuel
        fuel_time = fuel_needed / 2.0  # 2 L/s flow rate
        
        # Tire change time (if needed)
        tire_time = 12 if current_tire_life < 0.5 else 0
        
        # Driver change time (sometimes)
        driver_time = np.random.choice([0, 15], p=[0.7, 0.3])
        
        # Stationary = max of fuel and tire (done in parallel)
        stationary_time = max(fuel_time, tire_time) + driver_time
        
        # Add some variation (±2 seconds)
        stationary_time += np.random.uniform(-2, 2)
        
        # Add pit lane transit time
        total_pit_time = pit_lane_time + stationary_time
        
        return max(total_pit_time, pit_lane_time + 10)  # Minimum pit stop
    
    def save_race_data(
        self, 
        race_data: Dict[str, pd.DataFrame],
        output_dir: str = None,
        race_name: str = None
    ) -> None:
        """Save generated race data to CSV files."""
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'sample'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if race_name is None:
            race_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        for name, df in race_data.items():
            filename = f"{race_name}_{name}.csv"
            df.to_csv(output_dir / filename, index=False)
            print(f"Saved: {output_dir / filename}")
    
    def generate_historical_dataset(
        self,
        num_races: int = 20,
        circuits: List[str] = None,
        output_dir: str = None
    ) -> pd.DataFrame:
        """
        Generate a historical dataset of multiple races for ML training.
        
        Args:
            num_races: Number of races to generate
            circuits: List of circuits to use (random selection)
            output_dir: Directory to save files
            
        Returns:
            Combined DataFrame of all races
        """
        if circuits is None:
            circuits = list(self.circuits.get('circuits', {}).keys())
            if not circuits:
                circuits = ['spa', 'le_mans', 'daytona']
                
        all_races = []
        
        for i in range(num_races):
            circuit = random.choice(circuits)
            duration = random.choice([6.0, 8.0, 12.0, 24.0])
            
            print(f"Generating race {i+1}/{num_races}: {circuit} ({duration}h)")
            
            race_data = self.generate_race(
                circuit=circuit,
                duration_hours=duration,
                num_cars=random.randint(25, 40),
                seed=42 + i
            )
            
            # Add race identifier
            race_id = f"race_{i+1:03d}_{circuit}"
            for df_name, df in race_data.items():
                if not df.empty:
                    df['race_id'] = race_id
                    df['race_index'] = i + 1
            
            all_races.append(race_data)
            
            # Save individual race
            if output_dir:
                self.save_race_data(race_data, output_dir, race_id)
        
        # Combine all lap times for ML training
        combined_laps = pd.concat(
            [r['lap_times'] for r in all_races], 
            ignore_index=True
        )
        
        combined_sc = pd.concat(
            [r['safety_cars'] for r in all_races if not r['safety_cars'].empty],
            ignore_index=True
        )
        
        return {
            'lap_times': combined_laps,
            'safety_cars': combined_sc,
            'race_count': num_races
        }


def main():
    """Generate sample data for the project."""
    print("GT3/GTE Race Strategy Optimizer - Data Generator")
    print("=" * 50)
    
    generator = RaceDataGenerator()
    
    # Generate a single sample race
    print("\n1. Generating sample 6-hour race at Spa...")
    race_data = generator.generate_race(
        circuit='spa',
        duration_hours=6.0,
        num_cars=30,
        seed=42
    )
    
    generator.save_race_data(race_data, race_name='spa_6h_sample')
    
    # Print summary
    print(f"\nRace Summary:")
    print(f"  - Total laps recorded: {len(race_data['lap_times']):,}")
    print(f"  - Pit stops: {len(race_data['pit_stops']):,}")
    print(f"  - Safety car periods: {len(race_data['safety_cars'])}")
    print(f"  - Entries: {len(race_data['entries'])}")
    
    # Generate historical dataset for ML training
    print("\n2. Generating historical dataset (10 races) for ML training...")
    historical = generator.generate_historical_dataset(
        num_races=10,
        output_dir=Path(__file__).parent.parent.parent / 'data' / 'sample'
    )
    
    print(f"\nHistorical Dataset Summary:")
    print(f"  - Total laps: {len(historical['lap_times']):,}")
    print(f"  - Safety car events: {len(historical['safety_cars'])}")
    
    print("\n✅ Sample data generation complete!")
    print(f"   Files saved to: data/sample/")


if __name__ == '__main__':
    main()
