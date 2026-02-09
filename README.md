# GT3/GTE Race Strategy Optimizer 🏁

A comprehensive race strategy optimization system for GT3/GTE endurance racing with ML-based safety car prediction, pit window optimization, and Monte Carlo race simulation.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Pit Window Optimizer**: Calculate optimal pit stop windows using dynamic programming
- **Safety Car Predictor**: ML model predicting SC/FCY probability (calibrated with historical data)
- **Fuel Strategy Calculator**: Optimize fuel loads considering weight penalties
- **Tire Strategy Optimizer**: Model tire degradation and find optimal change windows
- **Race Simulator**: Monte Carlo simulation for strategy comparison
- **Driver Stint Planner**: FIA-compliant driver rotations for multi-driver endurance races
- **Multi-Class Traffic**: Realistic Hypercar/LMP2/GT3 traffic interaction modeling
- **Interactive Dashboard**: Streamlit-based visualization and analysis

## Project Structure

```
gt3-strategy-optimizer/
│
├── data/
│   ├── raw/                    # Raw scraped/downloaded data
│   ├── processed/              # Cleaned and processed data
│   ├── models/                 # Trained ML models
│   └── sample/                 # Sample data for testing
│
├── src/
│   ├── data_collection/
│   │   ├── scraper_wec.py      # WEC data scraper
│   │   ├── scraper_imsa.py     # IMSA data scraper
│   │   └── data_generator.py   # Sample data generator
│   │
│   ├── analysis/
│   │   ├── pit_window_optimizer.py
│   │   ├── fuel_strategy.py
│   │   ├── tire_strategy.py
│   │   ├── race_simulator.py
│   │   ├── driver_stint_planner.py 
│   │   └── multi_class_traffic.py   
│   │
│   ├── models/
│   │   └── sc_prediction_model.py   
│   │
│   └── visualization/
│       └── dashboard.py
│
├── config/
│   ├── circuits.yaml           # 20+ circuit configurations
│   └── cars.yaml               # GT3 car specifications + tire compounds
│
├── tests/
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gt3-strategy-optimizer.git
cd gt3-strategy-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data (for testing)

```bash
python -m src.data_collection.data_generator
```

### 2. Run the Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

### 3. Use Individual Modules

```python
from src.analysis.pit_window_optimizer import PitWindowOptimizer
from src.analysis.race_simulator import RaceSimulator

# Initialize optimizer
optimizer = PitWindowOptimizer(
    race_duration_minutes=360,  # 6 hours
    pit_stop_time=65,           # seconds
    fuel_capacity=120,          # liters
    fuel_consumption=3.2        # liters per lap
)

# Get optimal strategies
strategies = optimizer.optimize()
```

## Data Sources

The project supports data from:
- **FIA WEC** (World Endurance Championship)
- **IMSA** (WeatherTech SportsCar Championship)
- **GT World Challenge** (Europe/America)

### Using Real Data

Due to data licensing restrictions, this project includes a sample data generator for demonstration. To use real data:

1. Download timing data from official sources (CSV format)
2. Place files in `data/raw/`
3. Run the preprocessing pipeline:

```bash
python -m src.preprocessing.data_cleaner
```

## Modules Documentation

### Pit Window Optimizer

Calculates optimal pit stop windows considering:
- Fuel consumption and tank capacity
- Tire degradation curves
- Pit stop duration (including drive-through)
- Undercut/overcut opportunities

### Safety Car Predictor 

XGBoost model with calibrated heuristics:
- Features: circuit risk, race phase, weather, car count, night/day
- Race phase modeling: start (2.5x risk), mid-race (0.8x), final hour (1.4x)
- Weather impact: wet = 2.2x, heavy rain = 3.0x
- Returns probability for 15-minute window

```python
from src.models.sc_prediction_model import SafetyCarPredictor

predictor = SafetyCarPredictor()
pred = predictor.predict_probability('spa', race_minute=5, weather='dry')
# Returns: 53% Very High (race start)
pred = predictor.predict_probability('spa', race_minute=180, weather='wet')
# Returns: 29% Very High (mid-race + rain)
```

### Fuel Strategy Calculator

Optimizes fuel loads considering:
- Weight penalty (~0.03s per kg per lap)
- Minimum fuel requirements
- Splash & dash vs full tank strategies

### Tire Strategy Optimizer

Models tire degradation using:
- Track temperature
- Circuit aggressiveness
- Traffic conditions
- Stint length

### Race Simulator

Monte Carlo simulation supporting:
- Multiple strategy comparison
- Safety car probability integration
- Position tracking
- 1000+ iterations for statistical confidence

### Driver Stint Planner 

Plans driver rotations for endurance races:
- FIA regulation compliance (max 4h continuous, 14h total in 24h race)
- Fatigue modeling and recovery rates
- Skill-based assignment (wet specialist, night specialist)
- Condition-optimized planning (weather, day/night)

```python
from src.analysis.driver_stint_planner import DriverStintPlanner, Driver

drivers = [
    Driver(name="Pro", pace=0.95, wet_skill=0.85, night_skill=0.80),
    Driver(name="Silver", pace=0.82, night_skill=0.90),
    Driver(name="Bronze", pace=0.72, wet_skill=0.90)
]

planner = DriverStintPlanner(drivers, race_duration_hours=24, lap_time_seconds=234)
plan = planner.create_equal_time_plan(race_start_hour=16)
print(planner.format_plan_table(plan))
```

### Multi-Class Traffic 

Models realistic traffic in WEC/IMSA:
- Class speed differences (Hypercar ~15% faster than GT3)
- Blue flag compliance
- Time lost when lapping/being lapped
- Incident probability in traffic

```python
from src.analysis.multi_class_traffic import MultiClassSimulator, RaceClass

sim = MultiClassSimulator(series="WEC", base_lap_time=138.0)
traffic = sim.estimate_traffic_for_class(
    RaceClass.LMGT3,
    race_duration_hours=6,
    class_distribution={RaceClass.HYPERCAR: 10, RaceClass.LMP2: 8, RaceClass.LMGT3: 18}
)
# Shows: being lapped ~335 times, ~2.6 min lost to traffic
```

## Dashboard Features

1. **Race Setup**: Configure circuit, weather, car specifications
2. **Strategy Comparison**: Side-by-side strategy visualization
3. **Safety Car Analysis**: Probability heatmaps by lap
4. **Tire & Fuel Strategy**: Degradation curves and consumption
5. **Race Simulation**: Interactive Monte Carlo results

## Skills Demonstrated

-  **Data Engineering**: Web scraping, ETL pipelines, databases
-  **Machine Learning**: Classification, feature engineering, XGBoost
-  **Optimization**: Dynamic programming, genetic algorithms
-  **Domain Knowledge**: Motorsport strategy expertise
-  **Visualization**: Interactive Streamlit dashboards
-  **Software Engineering**: Modular code, testing, documentation



## License

MIT License - see [LICENSE](LICENSE) file for details.


