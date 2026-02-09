"""
Multi-Class Traffic Simulation
==============================

Models realistic traffic interactions in multi-class endurance racing:
- WEC: Hypercar, LMP2, LMGT3
- IMSA: GTP, LMP2, GTD Pro, GTD

Key factors:
- Time lost when being lapped by faster classes
- Time lost when lapping slower classes  
- Blue flag compliance
- Increased incident probability in traffic
- Class-specific pace differences
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceClass(Enum):
    """Racing classes in endurance racing."""
    # WEC Classes
    HYPERCAR = "hypercar"
    LMP2 = "lmp2"
    LMGT3 = "lmgt3"
    
    # IMSA Classes  
    GTP = "gtp"
    LMP2_IMSA = "lmp2_imsa"
    GTD_PRO = "gtd_pro"
    GTD = "gtd"


@dataclass
class ClassProfile:
    """Performance profile for a racing class."""
    name: str
    base_lap_time_factor: float  # 1.0 = baseline, lower = faster
    lap_time_variance: float     # Seconds of random variation
    fuel_consumption_factor: float  # 1.0 = baseline
    tire_deg_factor: float       # 1.0 = baseline
    reliability: float           # 0-1, higher = more reliable
    
    # Traffic behavior
    blue_flag_compliance: float  # 0-1, how quickly they let faster cars by
    aggression: float           # 0-1, affects incidents and overtaking
    
    
# WEC Class Profiles (2024 regulations)
WEC_CLASSES = {
    RaceClass.HYPERCAR: ClassProfile(
        name="Hypercar",
        base_lap_time_factor=0.85,    # ~15% faster than GT3
        lap_time_variance=0.5,
        fuel_consumption_factor=0.90,  # Hybrid efficiency
        tire_deg_factor=1.1,           # Harder on tires
        reliability=0.92,
        blue_flag_compliance=0.0,      # Don't need to move (fastest class)
        aggression=0.8
    ),
    RaceClass.LMP2: ClassProfile(
        name="LMP2",
        base_lap_time_factor=0.92,    # ~8% faster than GT3
        lap_time_variance=0.8,
        fuel_consumption_factor=0.95,
        tire_deg_factor=1.05,
        reliability=0.94,
        blue_flag_compliance=0.85,    # Must yield to Hypercar
        aggression=0.7
    ),
    RaceClass.LMGT3: ClassProfile(
        name="LMGT3",
        base_lap_time_factor=1.0,     # Baseline
        lap_time_variance=1.0,
        fuel_consumption_factor=1.0,
        tire_deg_factor=1.0,
        reliability=0.96,             # GT cars very reliable
        blue_flag_compliance=0.90,    # Must yield to prototypes
        aggression=0.6
    )
}

# IMSA Class Profiles
IMSA_CLASSES = {
    RaceClass.GTP: ClassProfile(
        name="GTP",
        base_lap_time_factor=0.85,
        lap_time_variance=0.5,
        fuel_consumption_factor=0.88,
        tire_deg_factor=1.1,
        reliability=0.90,             # New cars, some issues
        blue_flag_compliance=0.0,
        aggression=0.85
    ),
    RaceClass.LMP2_IMSA: ClassProfile(
        name="LMP2",
        base_lap_time_factor=0.92,
        lap_time_variance=0.8,
        fuel_consumption_factor=0.95,
        tire_deg_factor=1.05,
        reliability=0.93,
        blue_flag_compliance=0.85,
        aggression=0.7
    ),
    RaceClass.GTD_PRO: ClassProfile(
        name="GTD Pro",
        base_lap_time_factor=1.0,
        lap_time_variance=0.8,
        fuel_consumption_factor=1.0,
        tire_deg_factor=1.0,
        reliability=0.95,
        blue_flag_compliance=0.90,
        aggression=0.65
    ),
    RaceClass.GTD: ClassProfile(
        name="GTD",
        base_lap_time_factor=1.02,    # Slightly slower (Am drivers)
        lap_time_variance=1.5,        # More variation
        fuel_consumption_factor=1.0,
        tire_deg_factor=0.95,         # More conservative
        reliability=0.95,
        blue_flag_compliance=0.92,
        aggression=0.5                # More cautious
    )
}


@dataclass
class MultiClassCar:
    """A car in multi-class simulation."""
    car_id: int
    race_class: RaceClass
    class_profile: ClassProfile
    
    # State
    current_lap: int = 0
    total_time: float = 0.0
    position_overall: int = 0
    position_in_class: int = 0
    
    # Performance
    skill_factor: float = 1.0       # Driver/team skill (0.95-1.05)
    current_lap_time: float = 0.0
    
    # For traffic calculations
    track_position_pct: float = 0.0  # 0-100% around the lap
    
    def __post_init__(self):
        # Initialize with some random skill variation
        if self.skill_factor == 1.0:
            self.skill_factor = np.random.uniform(0.97, 1.03)


@dataclass
class TrafficEvent:
    """Record of a traffic interaction."""
    lap: int
    car_id: int
    car_class: str
    other_car_id: int
    other_class: str
    event_type: str  # "being_lapped", "lapping", "battle"
    time_lost: float
    incident: bool = False


class MultiClassSimulator:
    """
    Simulates multi-class endurance racing with realistic traffic.
    
    Models the complex interactions between different classes:
    - Hypercars lapping GTs multiple times per hour
    - Time lost for both faster and slower cars
    - Increased incident risk in traffic
    """
    
    # Time lost per interaction (seconds)
    TIME_LOSS_BEING_LAPPED = {
        # (faster_class, slower_class): (time_lost_by_slower, time_lost_by_faster)
        (RaceClass.HYPERCAR, RaceClass.LMGT3): (0.5, 1.5),  # GT loses 0.5s, Hypercar loses 1.5s
        (RaceClass.HYPERCAR, RaceClass.LMP2): (0.3, 1.0),
        (RaceClass.LMP2, RaceClass.LMGT3): (0.4, 1.2),
        (RaceClass.GTP, RaceClass.GTD): (0.5, 1.5),
        (RaceClass.GTP, RaceClass.GTD_PRO): (0.4, 1.2),
        (RaceClass.GTP, RaceClass.LMP2_IMSA): (0.3, 1.0),
        (RaceClass.LMP2_IMSA, RaceClass.GTD): (0.4, 1.2),
        (RaceClass.LMP2_IMSA, RaceClass.GTD_PRO): (0.3, 1.0),
        (RaceClass.GTD_PRO, RaceClass.GTD): (0.2, 0.5),
    }
    
    # Incident probability increase when in traffic
    TRAFFIC_INCIDENT_MULTIPLIER = 2.5
    
    def __init__(
        self,
        series: str = "WEC",  # "WEC" or "IMSA"
        base_lap_time: float = 138.0,  # GT3 baseline
        safety_car_prob_per_hour: float = 0.3
    ):
        """
        Initialize multi-class simulator.
        
        Args:
            series: "WEC" or "IMSA"
            base_lap_time: Base lap time for GT3/GTD class
            safety_car_prob_per_hour: Base SC probability
        """
        self.series = series
        self.base_lap_time = base_lap_time
        self.safety_car_prob = safety_car_prob_per_hour
        
        if series == "WEC":
            self.class_profiles = WEC_CLASSES
        else:
            self.class_profiles = IMSA_CLASSES
            
        self.cars: List[MultiClassCar] = []
        self.traffic_events: List[TrafficEvent] = []
        
        logger.info(f"Multi-class simulator initialized for {series}")
    
    def create_field(
        self,
        class_distribution: Dict[RaceClass, int]
    ) -> List[MultiClassCar]:
        """
        Create the race field with specified class distribution.
        
        Args:
            class_distribution: Dict mapping class to number of cars
            
        Returns:
            List of MultiClassCar objects
        """
        self.cars = []
        car_id = 0
        
        for race_class, count in class_distribution.items():
            profile = self.class_profiles.get(race_class)
            if profile is None:
                logger.warning(f"Unknown class {race_class}, skipping")
                continue
                
            for i in range(count):
                car = MultiClassCar(
                    car_id=car_id,
                    race_class=race_class,
                    class_profile=profile
                )
                self.cars.append(car)
                car_id += 1
        
        logger.info(f"Created field with {len(self.cars)} cars:")
        for rc, count in class_distribution.items():
            logger.info(f"  - {rc.value}: {count} cars")
        
        return self.cars
    
    def calculate_lap_time(
        self,
        car: MultiClassCar,
        traffic_density: float = 0.0,
        is_in_traffic: bool = False,
        fuel_load_pct: float = 1.0,
        tire_life_pct: float = 1.0,
        track_status: str = "green"
    ) -> Tuple[float, List[TrafficEvent]]:
        """
        Calculate lap time for a car considering all factors.
        
        Args:
            car: The car to calculate for
            traffic_density: 0-1, how much traffic on track
            is_in_traffic: Whether currently in traffic
            fuel_load_pct: Current fuel as percentage of full tank
            tire_life_pct: Current tire life percentage
            track_status: "green", "yellow", or "safety_car"
            
        Returns:
            Tuple of (lap_time, list of traffic events)
        """
        profile = car.class_profile
        events = []
        
        # Base lap time for this class
        class_lap_time = self.base_lap_time * profile.base_lap_time_factor
        
        # Apply skill factor
        lap_time = class_lap_time * car.skill_factor
        
        # Add random variance
        lap_time += np.random.normal(0, profile.lap_time_variance)
        
        # Fuel weight effect (heavier = slower)
        # Full tank adds ~0.1s per lap per 10kg
        fuel_penalty = fuel_load_pct * 3.0  # Up to 3s slower with full tank
        lap_time += fuel_penalty
        
        # Tire degradation effect
        if tire_life_pct < 0.3:  # Below cliff
            tire_penalty = (0.3 - tire_life_pct) * 15  # Up to 4.5s slower
        else:
            tire_penalty = (1 - tire_life_pct) * 1.5  # Gradual 0-1.5s
        lap_time += tire_penalty
        
        # Traffic effects
        if is_in_traffic:
            # Determine if we're catching slower cars or being caught
            traffic_time_loss = np.random.uniform(0.5, 2.5) * traffic_density
            lap_time += traffic_time_loss
            
            # Record event
            events.append(TrafficEvent(
                lap=car.current_lap,
                car_id=car.car_id,
                car_class=profile.name,
                other_car_id=-1,  # Generic traffic
                other_class="mixed",
                event_type="traffic",
                time_lost=traffic_time_loss
            ))
        
        # Track status effects
        if track_status == "safety_car":
            # SC lap time is ~40% slower
            lap_time = class_lap_time * 1.4
        elif track_status == "yellow":
            lap_time += 5.0  # Slowed in yellow zone
        
        return lap_time, events
    
    def simulate_lap_interactions(
        self,
        lap_number: int,
        race_time_seconds: float
    ) -> List[TrafficEvent]:
        """
        Simulate traffic interactions for one lap.
        
        Checks which cars are likely to interact based on
        relative pace and track position.
        """
        events = []
        
        # Sort cars by total time (race position)
        sorted_cars = sorted(self.cars, key=lambda c: c.total_time)
        
        # Check for lapping situations
        for i, faster_car in enumerate(sorted_cars[:-1]):
            for slower_car in sorted_cars[i+1:]:
                # Skip same class battles (handled differently)
                if faster_car.race_class == slower_car.race_class:
                    continue
                
                # Check if faster car is catching slower car
                pace_diff = (self.base_lap_time * slower_car.class_profile.base_lap_time_factor - 
                           self.base_lap_time * faster_car.class_profile.base_lap_time_factor)
                
                if pace_diff > 2.0:  # Significant pace difference
                    # Calculate probability of interaction this lap
                    # Based on how much faster and track position
                    interaction_prob = min(0.8, pace_diff / 20.0)
                    
                    if np.random.random() < interaction_prob:
                        # Interaction occurs
                        time_key = (faster_car.race_class, slower_car.race_class)
                        time_loss = self.TIME_LOSS_BEING_LAPPED.get(
                            time_key, (0.3, 1.0)
                        )
                        
                        # Blue flag compliance affects time loss
                        compliance = slower_car.class_profile.blue_flag_compliance
                        adjusted_faster_loss = time_loss[1] * (1.5 - compliance)
                        adjusted_slower_loss = time_loss[0] * compliance
                        
                        # Check for incident
                        incident = False
                        base_incident_prob = 0.002  # 0.2% base
                        traffic_incident_prob = base_incident_prob * self.TRAFFIC_INCIDENT_MULTIPLIER
                        
                        if np.random.random() < traffic_incident_prob:
                            incident = True
                            adjusted_faster_loss += 30  # Major time loss
                            adjusted_slower_loss += 30
                        
                        events.append(TrafficEvent(
                            lap=lap_number,
                            car_id=faster_car.car_id,
                            car_class=faster_car.class_profile.name,
                            other_car_id=slower_car.car_id,
                            other_class=slower_car.class_profile.name,
                            event_type="lapping",
                            time_lost=adjusted_faster_loss,
                            incident=incident
                        ))
                        
                        events.append(TrafficEvent(
                            lap=lap_number,
                            car_id=slower_car.car_id,
                            car_class=slower_car.class_profile.name,
                            other_car_id=faster_car.car_id,
                            other_class=faster_car.class_profile.name,
                            event_type="being_lapped",
                            time_lost=adjusted_slower_loss,
                            incident=incident
                        ))
        
        return events
    
    def estimate_traffic_for_class(
        self,
        target_class: RaceClass,
        race_duration_hours: float,
        class_distribution: Dict[RaceClass, int]
    ) -> Dict:
        """
        Estimate traffic impact for a specific class over race duration.
        
        Returns statistics about expected traffic interactions.
        """
        target_profile = self.class_profiles.get(target_class)
        if not target_profile:
            return {}
        
        results = {
            'class': target_class.value,
            'interactions_per_hour': 0,
            'time_lost_per_hour': 0,
            'lapping_events': 0,
            'being_lapped_events': 0,
            'incident_risk_multiplier': 1.0
        }
        
        # Calculate expected interactions with each other class
        for other_class, count in class_distribution.items():
            if other_class == target_class:
                continue
                
            other_profile = self.class_profiles.get(other_class)
            if not other_profile:
                continue
            
            # Pace difference per lap (seconds)
            pace_diff = self.base_lap_time * (
                other_profile.base_lap_time_factor - 
                target_profile.base_lap_time_factor
            )
            
            # Estimate laps per hour
            laps_per_hour = 3600 / (self.base_lap_time * target_profile.base_lap_time_factor)
            
            if pace_diff > 0:
                # Target class is faster - will lap others
                # How many times per hour do we catch each slower car?
                catches_per_hour = abs(pace_diff) / (self.base_lap_time * other_profile.base_lap_time_factor) * laps_per_hour
                catches_per_hour *= count  # Multiply by number of slower cars
                
                time_key = (target_class, other_class)
                time_loss = self.TIME_LOSS_BEING_LAPPED.get(time_key, (0.3, 1.0))[1]
                
                results['lapping_events'] += catches_per_hour * race_duration_hours
                results['time_lost_per_hour'] += catches_per_hour * time_loss
                results['interactions_per_hour'] += catches_per_hour
                
            elif pace_diff < 0:
                # Target class is slower - will be lapped
                catches_per_hour = abs(pace_diff) / (self.base_lap_time * target_profile.base_lap_time_factor) * laps_per_hour
                catches_per_hour *= count
                
                time_key = (other_class, target_class)
                time_loss = self.TIME_LOSS_BEING_LAPPED.get(time_key, (0.3, 1.0))[0]
                
                results['being_lapped_events'] += catches_per_hour * race_duration_hours
                results['time_lost_per_hour'] += catches_per_hour * time_loss
                results['interactions_per_hour'] += catches_per_hour
        
        # Calculate total time lost
        results['total_time_lost_seconds'] = results['time_lost_per_hour'] * race_duration_hours
        results['total_time_lost_minutes'] = results['total_time_lost_seconds'] / 60
        
        # Incident risk increases with more interactions
        results['incident_risk_multiplier'] = 1 + (results['interactions_per_hour'] * 0.05)
        
        return results
    
    def print_traffic_analysis(
        self,
        race_duration_hours: float,
        class_distribution: Dict[RaceClass, int]
    ):
        """Print detailed traffic analysis for all classes."""
        print("\n" + "=" * 70)
        print(f"MULTI-CLASS TRAFFIC ANALYSIS - {self.series}")
        print(f"Race Duration: {race_duration_hours}h")
        print("=" * 70)
        
        # Print class distribution
        print("\nField Composition:")
        for rc, count in class_distribution.items():
            profile = self.class_profiles.get(rc)
            if profile:
                lap_time = self.base_lap_time * profile.base_lap_time_factor
                print(f"  {profile.name:12} | {count:2} cars | ~{lap_time:.1f}s lap time")
        
        print("\n" + "-" * 70)
        print("Traffic Impact by Class:")
        print("-" * 70)
        
        for rc in class_distribution.keys():
            analysis = self.estimate_traffic_for_class(
                rc, race_duration_hours, class_distribution
            )
            
            if analysis:
                print(f"\n{analysis['class'].upper()}:")
                print(f"  Interactions/hour:    {analysis['interactions_per_hour']:.1f}")
                print(f"  Time lost/hour:       {analysis['time_lost_per_hour']:.1f}s")
                print(f"  Total time lost:      {analysis['total_time_lost_minutes']:.1f} min")
                print(f"  Lapping others:       {analysis['lapping_events']:.0f} times")
                print(f"  Being lapped:         {analysis['being_lapped_events']:.0f} times")
                print(f"  Incident risk:        {analysis['incident_risk_multiplier']:.2f}x")
        
        print("\n" + "=" * 70)


def main():
    """Demo multi-class traffic simulation."""
    
    # WEC Example: 6 Hours of Spa
    print("\n" + "=" * 70)
    print("WEC 6 Hours of Spa - Traffic Analysis")
    print("=" * 70)
    
    wec_sim = MultiClassSimulator(
        series="WEC",
        base_lap_time=138.0,  # GT3 baseline at Spa
        safety_car_prob_per_hour=0.35
    )
    
    wec_field = {
        RaceClass.HYPERCAR: 10,
        RaceClass.LMP2: 8,
        RaceClass.LMGT3: 18
    }
    
    wec_sim.print_traffic_analysis(
        race_duration_hours=6,
        class_distribution=wec_field
    )
    
    # IMSA Example: 24 Hours of Daytona
    print("\n" + "=" * 70)
    print("IMSA 24 Hours of Daytona - Traffic Analysis")
    print("=" * 70)
    
    imsa_sim = MultiClassSimulator(
        series="IMSA",
        base_lap_time=102.0,  # GTD baseline at Daytona
        safety_car_prob_per_hour=0.45
    )
    
    imsa_field = {
        RaceClass.GTP: 10,
        RaceClass.LMP2_IMSA: 5,
        RaceClass.GTD_PRO: 8,
        RaceClass.GTD: 12
    }
    
    imsa_sim.print_traffic_analysis(
        race_duration_hours=24,
        class_distribution=imsa_field
    )
    
    # Show specific impact on GT3 at Le Mans
    print("\n" + "=" * 70)
    print("LMGT3 Traffic Impact at Le Mans 24h")
    print("=" * 70)
    
    lemans_sim = MultiClassSimulator(
        series="WEC",
        base_lap_time=234.0,  # GT3 at Le Mans (~3:54)
        safety_car_prob_per_hour=0.25
    )
    
    lemans_field = {
        RaceClass.HYPERCAR: 16,
        RaceClass.LMP2: 12,
        RaceClass.LMGT3: 23
    }
    
    gt3_analysis = lemans_sim.estimate_traffic_for_class(
        RaceClass.LMGT3,
        race_duration_hours=24,
        class_distribution=lemans_field
    )
    
    print(f"\nAs a LMGT3 driver at Le Mans 24h, expect:")
    print(f"  - To be lapped ~{gt3_analysis['being_lapped_events']:.0f} times by prototypes")
    print(f"  - Lose ~{gt3_analysis['total_time_lost_minutes']:.1f} minutes to traffic")
    print(f"  - {gt3_analysis['interactions_per_hour']:.1f} close interactions per hour")
    print(f"  - {gt3_analysis['incident_risk_multiplier']:.1f}x higher incident risk than solo running")


if __name__ == "__main__":
    main()
