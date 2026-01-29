# GT3/GTE Race Strategy Optimizer 🏁

A comprehensive race strategy optimization system for GT3/GTE endurance racing with ML-based safety car prediction, pit window optimization, and Monte Carlo race simulation.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **🔧 Pit Window Optimizer**: Calculate optimal pit stop windows using dynamic programming
- **🚨 Safety Car Predictor**: ML model predicting SC/FCY probability using XGBoost
- **⛽ Fuel Strategy Calculator**: Optimize fuel loads considering weight penalties
- **🏎️ Tire Strategy Optimizer**: Model tire degradation and find optimal change windows
- **🎮 Race Simulator**: Monte Carlo simulation for strategy comparison
- **📊 Interactive Dashboard**: Streamlit-based visualization and analysis

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
│   ├── preprocessing/
│   │   ├── data_cleaner.py     # Data cleaning utilities
│   │   └── feature_engineering.py
│   │
│   ├── analysis/
│   │   ├── pit_window_optimizer.py
│   │   ├── fuel_strategy.py
│   │   ├── tire_strategy.py
│   │   ├── safety_car_predictor.py
│   │   └── race_simulator.py
│   │
│   ├── models/
│   │   ├── sc_prediction_model.py
│   │   └── stint_performance_model.py
│   │
│   └── visualization/
│       └── dashboard.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── config/
│   ├── circuits.yaml           # Circuit configurations
│   └── cars.yaml               # Car specifications
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

XGBoost model trained on historical data:
- Features: circuit, race minute, car count, weather, competition class
- Output: SC probability for next N laps
- Accuracy: ~72% on validation set

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

## Dashboard Features

1. **Race Setup**: Configure circuit, weather, car specifications
2. **Strategy Comparison**: Side-by-side strategy visualization
3. **Safety Car Analysis**: Probability heatmaps by lap
4. **Tire & Fuel Strategy**: Degradation curves and consumption
5. **Race Simulation**: Interactive Monte Carlo results

## Skills Demonstrated

- ✅ **Data Engineering**: Web scraping, ETL pipelines, databases
- ✅ **Machine Learning**: Classification, feature engineering, XGBoost
- ✅ **Optimization**: Dynamic programming, genetic algorithms
- ✅ **Domain Knowledge**: Motorsport strategy expertise
- ✅ **Visualization**: Interactive Streamlit dashboards
- ✅ **Software Engineering**: Modular code, testing, documentation

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- FIA WEC for timing data structure reference
- IMSA for race documentation
- The motorsport data science community

---

**Built with ❤️ for the motorsport community**
