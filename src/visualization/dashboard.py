"""
GT3/GTE Race Strategy Dashboard
==============================
Interactive Streamlit dashboard for race strategy analysis.

Run with: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.pit_window_optimizer import PitWindowOptimizer, StrategyType
from src.analysis.fuel_strategy import FuelStrategyCalculator
from src.analysis.tire_strategy import TireStrategyOptimizer, TireCompound
from src.analysis.race_simulator import RaceSimulator
from src.models.sc_prediction_model import SafetyCarPredictor

# Page config
st.set_page_config(
    page_title="GT3/GTE Race Strategy Optimizer",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<p class="main-header">GT3/GTE Race Strategy Optimizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional race strategy analysis with ML-powered predictions</p>', unsafe_allow_html=True)
    
    # Sidebar - Race Setup
    with st.sidebar:
        st.header("Race Setup")
        
        circuit = st.selectbox(
            "Circuit",
            options=['spa', 'le_mans', 'monza', 'daytona', 'sebring', 'nurburgring', 'fuji', 'bahrain'],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        race_duration = st.slider(
            "Race Duration (hours)",
            min_value=1,
            max_value=24,
            value=6,
            step=1
        )
        
        # Circuit-specific defaults with class multipliers
        # Base times are GT3 reference (2024 data)
        circuit_configs = {
            'spa': {'lap_time': 138, 'fuel_consumption': 3.2, 'tire_wear': 'high', 'track_length': 7.004},
            'le_mans': {'lap_time': 234, 'fuel_consumption': 4.8, 'tire_wear': 'low', 'track_length': 13.626},
            'monza': {'lap_time': 108, 'fuel_consumption': 2.6, 'tire_wear': 'low', 'track_length': 5.793},
            'daytona': {'lap_time': 102, 'fuel_consumption': 2.5, 'tire_wear': 'medium', 'track_length': 5.729},
            'sebring': {'lap_time': 115, 'fuel_consumption': 2.8, 'tire_wear': 'high', 'track_length': 6.019},
            'nurburgring': {'lap_time': 98, 'fuel_consumption': 2.4, 'tire_wear': 'medium', 'track_length': 5.148},
            'fuji': {'lap_time': 92, 'fuel_consumption': 2.2, 'tire_wear': 'medium', 'track_length': 4.563},
            'bahrain': {'lap_time': 105, 'fuel_consumption': 2.5, 'tire_wear': 'high', 'track_length': 5.412}
        }
        
        # Class speed multipliers (relative to GT3 baseline)
        class_multipliers = {
            'GT3/LMGT3': 1.00,
            'LMP2': 0.92,  # ~8% faster than GT3
            'Hypercar/GTP': 0.85  # ~15% faster than GT3
        }
        
        config = circuit_configs.get(circuit, circuit_configs['spa'])
        
        st.subheader("Car Setup")
        
        # Class selection
        car_class = st.selectbox(
            "Class",
            options=list(class_multipliers.keys()),
            index=0
        )
        
        # Calculate reference lap time for selected class
        gt3_reference = config['lap_time']
        class_multiplier = class_multipliers[car_class]
        calculated_lap_time = int(gt3_reference * class_multiplier)
        
        # Show reference info
        st.caption(f"**{circuit.replace('_', ' ').title()} Reference Times:**")
        col_ref1, col_ref2 = st.columns(2)
        with col_ref1:
            st.caption(f"GT3: {gt3_reference}s")
            st.caption(f"LMP2: {int(gt3_reference * 0.92)}s")
        with col_ref2:
            st.caption(f"Hyper: {int(gt3_reference * 0.85)}s")
            st.caption(f"Track: {config['track_length']}km")
        
        # Lap time input with calculated default
        use_auto_laptime = st.checkbox("Use reference lap time", value=True)
        
        if use_auto_laptime:
            lap_time = calculated_lap_time
            st.info(f"Auto: **{lap_time}s** ({car_class})")
        else:
            lap_time = st.number_input(
                "Custom Lap Time (s)",
                min_value=60,
                max_value=300,
                value=calculated_lap_time,
                help=f"Reference for {car_class}: {calculated_lap_time}s"
            )
        
        fuel_capacity = st.number_input(
            "Fuel Capacity (L)",
            min_value=50,
            max_value=200,
            value=120 if 'GT3' in car_class else 110 if 'LMP2' in car_class else 100
        )
        
        # Fuel consumption also varies by class
        base_fuel_consumption = config['fuel_consumption']
        class_fuel_multipliers = {
            'GT3/LMGT3': 1.00,
            'LMP2': 0.90,  # More efficient
            'Hypercar/GTP': 0.85  # Hybrid efficiency
        }
        calculated_fuel = round(base_fuel_consumption * class_fuel_multipliers[car_class], 1)
        
        use_auto_fuel = st.checkbox("Use reference fuel consumption", value=True)
        
        if use_auto_fuel:
            fuel_consumption = calculated_fuel
            st.caption(f"Auto: **{fuel_consumption} L/lap**")
        else:
            fuel_consumption = st.number_input(
                "Custom Fuel Consumption (L/lap)",
                min_value=1.0,
                max_value=6.0,
                value=calculated_fuel,
                step=0.1
            )
        
        st.subheader("Conditions")
        
        weather = st.selectbox(
            "Weather",
            options=['dry', 'damp', 'wet', 'heavy_rain'],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        track_temp = st.slider(
            "Track Temperature (°C)",
            min_value=10,
            max_value=50,
            value=25
        )
        
        grid_position = st.slider(
            "Starting Position",
            min_value=1,
            max_value=40,
            value=15
        )
    
    # Calculate race parameters
    race_minutes = race_duration * 60
    estimated_laps = int((race_minutes * 60) / lap_time)
    max_stint_fuel = int(fuel_capacity / fuel_consumption)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Strategy Comparison",
        "Safety Car Analysis",
        "Fuel Strategy",
        "Tire Strategy",
        "Race Simulation"
    ])
    
    # Tab 1: Strategy Comparison
    with tab1:
        st.header("Strategy Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Estimated Laps", estimated_laps)
        with col2:
            st.metric("Max Stint (Fuel)", f"{max_stint_fuel} laps")
        with col3:
            st.metric("Min Pit Stops", max(1, estimated_laps // max_stint_fuel))
        with col4:
            st.metric("Race Duration", f"{race_duration}h")
        
        st.divider()
        
        # Initialize optimizer
        optimizer = PitWindowOptimizer(
            race_duration_minutes=race_minutes,
            lap_time_seconds=lap_time,
            pit_stop_time=65,
            fuel_capacity=fuel_capacity,
            fuel_consumption=fuel_consumption
        )
        
        # Strategy options
        col1, col2 = st.columns([1, 2])
        
        with col1:
            min_stops = st.number_input("Min Stops", 1, 10, 2)
            max_stops = st.number_input("Max Stops", min_stops, 10, min(5, min_stops + 3))
            
            strategy_types = st.multiselect(
                "Strategy Types",
                options=['Balanced', 'Aggressive', 'Conservative'],
                default=['Balanced', 'Aggressive']
            )
            
            type_mapping = {
                'Balanced': StrategyType.BALANCED,
                'Aggressive': StrategyType.AGGRESSIVE,
                'Conservative': StrategyType.CONSERVATIVE
            }
            selected_types = [type_mapping[t] for t in strategy_types]
        
        # Calculate strategies
        strategies = optimizer.optimize(
            min_stops=int(min_stops),
            max_stops=int(max_stops),
            include_types=selected_types
        )
        
        with col2:
            if strategies:
                # Create comparison chart
                df = optimizer.to_dataframe(strategies[:8])
                
                fig = px.bar(
                    df,
                    x='Strategy',
                    y='Total Time (s)',
                    color='Stops',
                    title='Strategy Comparison - Total Race Time',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Strategy details
        if strategies:
            st.subheader("Strategy Details")
            st.dataframe(
                optimizer.to_dataframe(strategies[:8]),
                use_container_width=True,
                hide_index=True
            )
            
            # Pit window visualization
            st.subheader("Pit Stop Windows")
            
            fig = go.Figure()
            
            for i, strategy in enumerate(strategies[:5]):
                for j, window in enumerate(strategy.pit_windows):
                    fig.add_trace(go.Scatter(
                        x=[window.lap_start, window.lap_optimal, window.lap_end],
                        y=[i, i, i],
                        mode='lines+markers',
                        name=f"{strategy.name} - Stop {j+1}",
                        line=dict(width=8),
                        marker=dict(size=10)
                    ))
            
            fig.update_layout(
                title="Optimal Pit Windows by Strategy",
                xaxis_title="Lap Number",
                yaxis_title="Strategy",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Safety Car Analysis
    with tab2:
        st.header("Safety Car Probability Analysis")
        
        predictor = SafetyCarPredictor()
        
        # Current prediction
        col1, col2, col3 = st.columns(3)
        
        current_minute = st.slider(
            "Current Race Time (minutes)",
            0, race_minutes, race_minutes // 3
        )
        
        prediction = predictor.predict_probability(
            circuit=circuit,
            race_minute=current_minute,
            weather=weather,
            track_temp=track_temp
        )
        
        with col1:
            st.metric(
                "SC Probability (next 10 laps)",
                f"{prediction['probability']:.1%}"
            )
        with col2:
            risk_colors = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High', 'Very High': 'Very High'}
            st.metric(
                "Risk Level",
                f"{risk_colors.get(prediction['risk_level'], 'Unknown')} {prediction['risk_level']}"
            )
        with col3:
            st.info(prediction['recommendation'])
        
        st.divider()
        
        # Timeline prediction
        st.subheader("SC Probability Throughout Race")
        
        timeline = predictor.predict_race_windows(
            circuit=circuit,
            race_duration_minutes=race_minutes,
            weather=weather,
            interval_minutes=10
        )
        
        fig = px.area(
            timeline,
            x='minute',
            y='probability',
            title=f'Safety Car Probability - {circuit.title()} ({weather.title()})',
            color_discrete_sequence=['#EF4444']
        )
        
        fig.add_hline(y=0.25, line_dash="dash", line_color="orange",
                      annotation_text="High Risk Threshold")
        fig.add_vline(x=current_minute, line_dash="dash", line_color="blue",
                      annotation_text="Current Position")
        
        fig.update_layout(
            xaxis_title="Race Time (minutes)",
            yaxis_title="SC Probability",
            height=400,
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Circuit comparison
        st.subheader("Circuit Risk Comparison")
        
        circuits_comparison = []
        for c in ['spa', 'daytona', 'le_mans', 'sebring', 'monza', 'bahrain']:
            pred = predictor.predict_probability(c, 120, weather)
            circuits_comparison.append({
                'Circuit': c.replace('_', ' ').title(),
                'Risk': predictor.CIRCUIT_RISK.get(c, 0.15),
                'Current Probability': pred['probability']
            })
        
        fig = px.bar(
            pd.DataFrame(circuits_comparison),
            x='Circuit',
            y='Risk',
            title='Base Safety Car Risk by Circuit',
            color='Risk',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=350, yaxis=dict(tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Fuel Strategy
    with tab3:
        st.header("Fuel Strategy Calculator")
        
        fuel_calc = FuelStrategyCalculator(
            fuel_capacity=fuel_capacity,
            fuel_consumption=fuel_consumption,
            base_lap_time=lap_time
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            stint_length = st.slider(
                "Stint Length (laps)",
                min_value=10,
                max_value=max_stint_fuel,
                value=min(35, max_stint_fuel - 2)
            )
            
            num_stops = st.slider(
                "Planned Stops",
                min_value=1,
                max_value=8,
                value=4
            )
        
        # Calculate fuel options
        scenarios = fuel_calc.optimize_fuel_load(stint_length)
        
        with col2:
            if scenarios:
                df = fuel_calc.to_dataframe(scenarios)
                
                fig = px.bar(
                    df,
                    x='Scenario',
                    y=['Weight Penalty (s)', 'Pit Time (s)'],
                    title='Fuel Strategy Comparison',
                    barmode='group'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Full race plan
        st.subheader("Complete Race Fuel Plan")
        
        plan = fuel_calc.plan_race_fuel(estimated_laps, num_stops)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fuel Used", f"{plan.total_fuel_used:.0f} L")
        with col2:
            st.metric("Weight Penalty", f"{plan.total_weight_penalty:.1f} s")
        with col3:
            st.metric("Pit Time", f"{plan.total_pit_time:.0f} s")
        with col4:
            st.metric("Efficiency", f"{plan.efficiency_score:.1%}")
        
        # Stint breakdown
        stint_data = []
        cumulative_lap = 0
        for i, s in enumerate(plan.scenarios, 1):
            cumulative_lap += s.stint_laps
            stint_data.append({
                'Stint': i,
                'Laps': s.stint_laps,
                'Fuel Load': f"{s.fuel_load:.0f} L",
                'End Lap': cumulative_lap,
                'Time Cost': f"{s.total_time_cost:.1f} s"
            })
        
        st.dataframe(pd.DataFrame(stint_data), use_container_width=True, hide_index=True)
        
        # Weight chart
        st.subheader("Fuel Weight Over Race")
        
        weight_data = []
        for lap in range(1, estimated_laps + 1):
            stint = (lap - 1) // (estimated_laps // (num_stops + 1)) + 1
            stint_lap = (lap - 1) % (estimated_laps // (num_stops + 1)) + 1
            fuel_remaining = fuel_capacity - (stint_lap * fuel_consumption)
            weight_data.append({
                'Lap': lap,
                'Fuel (L)': max(0, fuel_remaining),
                'Weight (kg)': max(0, fuel_remaining * 0.75)
            })
        
        fig = px.line(
            pd.DataFrame(weight_data),
            x='Lap',
            y='Weight (kg)',
            title='Fuel Weight Throughout Race'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Tire Strategy
    with tab4:
        st.header("Tire Strategy Optimizer")
        
        tire_opt = TireStrategyOptimizer(
            track_wear=config['tire_wear'],
            track_temp=track_temp,
            base_lap_time=lap_time
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_compound = st.selectbox(
                "Analyze Compound",
                options=['soft', 'medium', 'hard'],
                index=1,
                format_func=str.title
            )
            
            compound_map = {
                'soft': TireCompound.SOFT,
                'medium': TireCompound.MEDIUM,
                'hard': TireCompound.HARD
            }
        
        # Degradation analysis
        analysis = tire_opt.find_optimal_change_lap(compound_map[selected_compound])
        
        with col2:
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Cliff Lap", analysis['cliff_lap'])
            with col2b:
                st.metric("Optimal Change", f"Lap {analysis['optimal_change_lap']}")
            with col2c:
                st.metric("Safe Window", f"Laps {analysis['safe_window'][0]}-{analysis['safe_window'][1]}")
        
        st.divider()
        
        # Degradation chart
        st.subheader("Tire Degradation Curves")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Tire Life', 'Lap Time Delta'])
        
        colors = {'soft': '#EF4444', 'medium': '#F59E0B', 'hard': '#3B82F6'}
        
        for compound_name, compound_enum in compound_map.items():
            chart_data = tire_opt.degradation_chart(compound_enum, 50)
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data['lap'],
                    y=chart_data['life_pct'],
                    mode='lines',
                    name=f'{compound_name.title()} - Life',
                    line=dict(color=colors[compound_name])
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data['lap'],
                    y=chart_data['lap_time_delta'],
                    mode='lines',
                    name=f'{compound_name.title()} - Delta',
                    line=dict(color=colors[compound_name], dash='dash')
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Lap", row=1, col=1)
        fig.update_xaxes(title_text="Lap", row=1, col=2)
        fig.update_yaxes(title_text="Life (%)", row=1, col=1)
        fig.update_yaxes(title_text="Time Loss (s)", row=1, col=2)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal strategy
        st.subheader("Recommended Tire Strategy")
        
        tire_strategy = tire_opt.optimize_strategy(
            race_laps=estimated_laps,
            num_stops=num_stops
        )
        
        st.write(f"**Compound Sequence:** {' → '.join([c.value.title() for c in tire_strategy.compound_sequence])}")
        st.write(f"**Change at Laps:** {tire_strategy.change_laps}")
        st.write(f"**Expected Time Loss:** {tire_strategy.expected_time_loss:.1f}s")
        st.write(f"**Risk Assessment:** {tire_strategy.risk_assessment}")
    
    # Tab 5: Race Simulation
    with tab5:
        st.header("Monte Carlo Race Simulation")
        
        st.info("Run simulations to compare strategy performance under various conditions (SC, weather, etc.)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=2000,
                value=500,
                step=100
            )
            
            st.subheader("Strategy 1")
            stops_1 = st.number_input("Stops (Strategy 1)", 1, 8, 4, key='stops1')
            name_1 = st.text_input("Name", f"{stops_1}-Stop Balanced", key='name1')
            
            st.subheader("Strategy 2")
            stops_2 = st.number_input("Stops (Strategy 2)", 1, 8, 3, key='stops2')
            name_2 = st.text_input("Name", f"{stops_2}-Stop Aggressive", key='name2')
        
        with col2:
            if st.button("Run Simulation", type="primary"):
                with st.spinner(f"Running {num_simulations} simulations..."):
                    # Initialize simulator
                    simulator = RaceSimulator(
                        num_cars=30,
                        race_laps=estimated_laps,
                        base_lap_time=lap_time,
                        fuel_capacity=fuel_capacity,
                        fuel_consumption=fuel_consumption,
                        safety_car_probability=predictor.CIRCUIT_RISK.get(circuit, 0.15)
                    )
                    
                    # Define strategies
                    stint_len_1 = estimated_laps // (stops_1 + 1)
                    stint_len_2 = estimated_laps // (stops_2 + 1)
                    
                    strategies = [
                        {
                            'name': name_1,
                            'pit_laps': [stint_len_1 * (i + 1) for i in range(stops_1)]
                        },
                        {
                            'name': name_2,
                            'pit_laps': [stint_len_2 * (i + 1) for i in range(stops_2)]
                        }
                    ]
                    
                    # Run simulations
                    comparison = simulator.compare_strategies(
                        strategies,
                        grid_position=grid_position,
                        num_simulations=num_simulations
                    )
                    
                    st.success("Simulation complete!")
                    
                    # Results
                    st.subheader("Simulation Results")
                    st.dataframe(comparison, use_container_width=True, hide_index=True)
                    
                    # Position distribution chart
                    fig = go.Figure()
                    
                    for strategy in strategies:
                        result = simulator.monte_carlo(
                            strategy,
                            grid_position=grid_position,
                            num_simulations=num_simulations // 2
                        )
                        
                        positions = list(result.position_distribution.keys())
                        probabilities = list(result.position_distribution.values())
                        
                        fig.add_trace(go.Bar(
                            x=positions,
                            y=probabilities,
                            name=strategy['name']
                        ))
                    
                    fig.update_layout(
                        title="Finishing Position Distribution",
                        xaxis_title="Position",
                        yaxis_title="Probability",
                        barmode='group',
                        height=400,
                        yaxis=dict(tickformat='.0%')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>GT3/GTE Race Strategy Optimizer | Built for the motorsport community</p>
        <p>Data sources: WEC, IMSA, GT World Challenge</p>
    </div>
    """, unsafe_allow_html=True)


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == '__main__':
    main()
