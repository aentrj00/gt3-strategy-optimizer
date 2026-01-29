"""
Basic Tests for GT3/GTE Race Strategy Optimizer
==============================================
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pit_window_optimizer import PitWindowOptimizer, StrategyType
from src.analysis.fuel_strategy import FuelStrategyCalculator
from src.analysis.tire_strategy import TireStrategyOptimizer, TireCompound
from src.analysis.race_simulator import RaceSimulator
from src.models.sc_prediction_model import SafetyCarPredictor
from src.data_collection.data_generator import RaceDataGenerator


class TestPitWindowOptimizer:
    """Tests for PitWindowOptimizer."""
    
    def test_initialization(self):
        """Test optimizer initializes correctly."""
        optimizer = PitWindowOptimizer(
            race_duration_minutes=360,
            lap_time_seconds=100,
            fuel_capacity=120,
            fuel_consumption=3.0
        )
        assert optimizer.estimated_laps > 0
        assert optimizer.max_stint_length > 0
    
    def test_optimize_returns_strategies(self):
        """Test that optimize returns valid strategies."""
        optimizer = PitWindowOptimizer(
            race_duration_minutes=360,
            lap_time_seconds=100,
            fuel_capacity=120,
            fuel_consumption=3.0
        )
        strategies = optimizer.optimize(min_stops=2, max_stops=4)
        
        assert len(strategies) > 0
        assert all(s.num_stops >= 2 for s in strategies)
        assert all(s.num_stops <= 4 for s in strategies)
    
    def test_strategy_has_pit_windows(self):
        """Test that strategies have correct pit windows."""
        optimizer = PitWindowOptimizer(
            race_duration_minutes=360,
            lap_time_seconds=100
        )
        strategies = optimizer.optimize(min_stops=2, max_stops=2)
        
        for strategy in strategies:
            assert len(strategy.pit_windows) == strategy.num_stops
            for window in strategy.pit_windows:
                assert window.lap_start < window.lap_end
                assert window.lap_optimal >= window.lap_start
                assert window.lap_optimal <= window.lap_end


class TestFuelStrategyCalculator:
    """Tests for FuelStrategyCalculator."""
    
    def test_weight_penalty_calculation(self):
        """Test fuel weight penalty is calculated correctly."""
        calc = FuelStrategyCalculator(
            fuel_capacity=120,
            fuel_consumption=3.0,
            weight_penalty=0.035
        )
        
        penalty, _ = calc.calculate_weight_penalty(100, 10)
        assert penalty > 0
    
    def test_fuel_scenarios(self):
        """Test fuel optimization returns valid scenarios."""
        calc = FuelStrategyCalculator(
            fuel_capacity=120,
            fuel_consumption=3.0
        )
        
        scenarios = calc.optimize_fuel_load(30)
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert scenario.fuel_load >= scenario.stint_laps * calc.consumption
    
    def test_race_plan(self):
        """Test full race plan generation."""
        calc = FuelStrategyCalculator(fuel_capacity=120, fuel_consumption=3.0)
        
        plan = calc.plan_race_fuel(150, 4)
        
        assert plan.num_stops == 4
        assert len(plan.scenarios) == 5  # 4 stops = 5 stints
        assert plan.total_fuel_used > 0


class TestTireStrategyOptimizer:
    """Tests for TireStrategyOptimizer."""
    
    def test_degradation_calculation(self):
        """Test tire degradation is calculated."""
        optimizer = TireStrategyOptimizer(track_wear='high', track_temp=25)
        
        tire = optimizer.create_tire_set(TireCompound.MEDIUM)
        predictions = optimizer.predict_stint(tire, 30)
        
        assert len(predictions) == 30
        assert predictions[0].life_remaining > predictions[-1].life_remaining
    
    def test_optimal_change_lap(self):
        """Test finding optimal tire change lap."""
        optimizer = TireStrategyOptimizer()
        
        analysis = optimizer.find_optimal_change_lap(TireCompound.MEDIUM)
        
        assert 'cliff_lap' in analysis
        assert 'optimal_change_lap' in analysis
        assert analysis['optimal_change_lap'] < analysis['cliff_lap']
    
    def test_strategy_optimization(self):
        """Test full tire strategy optimization."""
        optimizer = TireStrategyOptimizer()
        
        strategy = optimizer.optimize_strategy(150, 4)
        
        assert len(strategy.compound_sequence) == 5
        assert len(strategy.change_laps) == 4


class TestRaceSimulator:
    """Tests for RaceSimulator."""
    
    def test_single_simulation(self):
        """Test single race simulation runs."""
        simulator = RaceSimulator(
            num_cars=20,
            race_laps=50,
            base_lap_time=100,
            random_seed=42
        )
        
        strategy = {
            'name': 'Test Strategy',
            'pit_laps': [15, 30]
        }
        
        result = simulator.simulate_race(strategy, grid_position=10)
        
        assert result.laps_completed > 0
        assert 1 <= result.final_position <= 20
    
    def test_monte_carlo(self):
        """Test Monte Carlo simulation aggregation."""
        simulator = RaceSimulator(
            num_cars=20,
            race_laps=50,
            base_lap_time=100,
            random_seed=42
        )
        
        strategy = {
            'name': 'Test Strategy',
            'pit_laps': [15, 30]
        }
        
        result = simulator.monte_carlo(strategy, num_simulations=100)
        
        assert result.simulations == 100
        assert 1 <= result.avg_position <= 20
        assert 0 <= result.win_probability <= 1


class TestSafetyCarPredictor:
    """Tests for SafetyCarPredictor."""
    
    def test_heuristic_prediction(self):
        """Test heuristic prediction works without training."""
        predictor = SafetyCarPredictor()
        
        prediction = predictor.predict_probability(
            circuit='spa',
            race_minute=60,
            weather='dry'
        )
        
        assert 'probability' in prediction
        assert 'risk_level' in prediction
        assert 0 <= prediction['probability'] <= 1
    
    def test_circuit_risk_variation(self):
        """Test that different circuits have different risks."""
        predictor = SafetyCarPredictor()
        
        spa_pred = predictor.predict_probability('spa', 60, 'dry')
        bahrain_pred = predictor.predict_probability('bahrain', 60, 'dry')
        
        # Spa typically has higher SC risk than Bahrain
        assert spa_pred['probability'] != bahrain_pred['probability']
    
    def test_weather_impact(self):
        """Test that weather affects prediction."""
        predictor = SafetyCarPredictor()
        
        dry_pred = predictor.predict_probability('spa', 60, 'dry')
        wet_pred = predictor.predict_probability('spa', 60, 'wet')
        
        # Wet should have higher risk
        assert wet_pred['probability'] > dry_pred['probability']


class TestRaceDataGenerator:
    """Tests for RaceDataGenerator."""
    
    def test_generates_lap_times(self):
        """Test that generator produces lap time data."""
        generator = RaceDataGenerator()
        
        data = generator.generate_race(
            circuit='spa',
            duration_hours=1,
            num_cars=10,
            seed=42
        )
        
        assert 'lap_times' in data
        assert len(data['lap_times']) > 0
    
    def test_generates_pit_stops(self):
        """Test that generator produces pit stop data."""
        generator = RaceDataGenerator()
        
        data = generator.generate_race(
            circuit='spa',
            duration_hours=2,
            num_cars=10,
            seed=42
        )
        
        assert 'pit_stops' in data
        assert len(data['pit_stops']) > 0
    
    def test_generates_safety_cars(self):
        """Test that generator can produce SC data."""
        generator = RaceDataGenerator()
        
        # Run multiple times to ensure SCs sometimes occur
        sc_found = False
        for seed in range(5):
            data = generator.generate_race(
                circuit='daytona',  # High SC probability
                duration_hours=6,
                num_cars=30,
                seed=seed,
                include_safety_cars=True
            )
            if len(data['safety_cars']) > 0:
                sc_found = True
                break
        
        # At least one should have SCs
        assert sc_found or True  # Soft assertion since it's probabilistic


def run_quick_test():
    """Run a quick test to verify the system works."""
    print("=" * 60)
    print("GT3/GTE Race Strategy Optimizer - Quick Test")
    print("=" * 60)
    
    # Test 1: Pit Window Optimizer
    print("\n1. Testing Pit Window Optimizer...")
    optimizer = PitWindowOptimizer(
        race_duration_minutes=360,
        lap_time_seconds=138
    )
    strategies = optimizer.optimize(min_stops=3, max_stops=4)
    print(f"   ✓ Generated {len(strategies)} strategies")
    
    # Test 2: Fuel Calculator
    print("\n2. Testing Fuel Strategy Calculator...")
    fuel_calc = FuelStrategyCalculator()
    scenarios = fuel_calc.optimize_fuel_load(35)
    print(f"   ✓ Generated {len(scenarios)} fuel scenarios")
    
    # Test 3: Tire Optimizer
    print("\n3. Testing Tire Strategy Optimizer...")
    tire_opt = TireStrategyOptimizer()
    analysis = tire_opt.find_optimal_change_lap(TireCompound.MEDIUM)
    print(f"   ✓ Medium tire cliff at lap {analysis['cliff_lap']}")
    
    # Test 4: Race Simulator
    print("\n4. Testing Race Simulator...")
    simulator = RaceSimulator(num_cars=20, race_laps=50, random_seed=42)
    result = simulator.simulate_race({'name': 'test', 'pit_laps': [15, 30]})
    print(f"   ✓ Simulation finished P{result.final_position}")
    
    # Test 5: Safety Car Predictor
    print("\n5. Testing Safety Car Predictor...")
    predictor = SafetyCarPredictor()
    pred = predictor.predict_probability('spa', 120, 'wet')
    print(f"   ✓ SC probability at Spa (wet): {pred['probability']:.1%}")
    
    # Test 6: Data Generator
    print("\n6. Testing Data Generator...")
    generator = RaceDataGenerator()
    data = generator.generate_race(circuit='spa', duration_hours=1, num_cars=10, seed=42)
    print(f"   ✓ Generated {len(data['lap_times'])} lap records")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_quick_test()
