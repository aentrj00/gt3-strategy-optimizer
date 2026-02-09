"""
Driver Stint Planner for Endurance Racing

Plans optimal driver rotations considering:
- FIA driving time regulations (max stint, max total drive time)
- Driver skills (wet weather, night driving, consistency, pace)
- Fatigue modeling
- Strategic windows (SC probability, weather changes)

Regulations reference (FIA WEC):
- Maximum continuous driving: 4 hours
- Maximum total driving per driver: 14 hours (24h race), proportional for shorter
- Minimum 3 drivers for races > 6 hours
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrivingCondition(Enum):
    """Track conditions that affect driver selection."""
    DRY_DAY = "dry_day"
    DRY_NIGHT = "dry_night"
    WET_DAY = "wet_day"
    WET_NIGHT = "wet_night"
    MIXED = "mixed"


class StintType(Enum):
    """Type of stint based on race phase."""
    OPENING = "opening"      # First stint - aggressive start
    STANDARD = "standard"    # Normal racing
    NIGHT = "night"          # Night phase (if applicable)
    CLOSING = "closing"      # Final stint - bring it home
    SC_RESPONSE = "sc_response"  # Short stint due to Safety Car


@dataclass
class Driver:
    """Represents a driver with their characteristics."""
    name: str
    
    # Skill ratings (0.0 to 1.0, where 1.0 is best)
    pace: float = 0.8           # Raw speed
    consistency: float = 0.8    # Lap time variation
    wet_skill: float = 0.7      # Performance in rain
    night_skill: float = 0.7    # Performance at night
    tire_management: float = 0.8  # How well they preserve tires
    fuel_saving: float = 0.7    # Ability to save fuel when needed
    
    # Physical characteristics
    fatigue_resistance: float = 0.8  # How well they handle long stints
    recovery_rate: float = 0.8       # How fast they recover between stints
    
    # Current state (updated during race)
    total_drive_time_minutes: float = 0.0
    last_stint_end_minute: float = 0.0
    stints_completed: int = 0
    current_fatigue: float = 0.0  # 0.0 = fresh, 1.0 = exhausted
    
    def reset_for_new_race(self):
        """Reset driver state for a new race."""
        self.total_drive_time_minutes = 0.0
        self.last_stint_end_minute = 0.0
        self.stints_completed = 0
        self.current_fatigue = 0.0
    
    def get_effective_pace(self, condition: DrivingCondition) -> float:
        """Calculate effective pace based on conditions and fatigue."""
        base_pace = self.pace
        
        # Apply condition modifiers
        if condition in [DrivingCondition.WET_DAY, DrivingCondition.WET_NIGHT]:
            base_pace *= (0.7 + 0.3 * self.wet_skill)
        
        if condition in [DrivingCondition.DRY_NIGHT, DrivingCondition.WET_NIGHT]:
            base_pace *= (0.8 + 0.2 * self.night_skill)
        
        # Apply fatigue penalty (up to 5% slower when exhausted)
        fatigue_penalty = self.current_fatigue * 0.05
        base_pace *= (1 - fatigue_penalty)
        
        return base_pace
    
    def update_fatigue(self, stint_duration_minutes: float, rest_duration_minutes: float = 0):
        """Update fatigue based on stint and rest duration."""
        # Fatigue increases with stint length
        fatigue_increase = (stint_duration_minutes / 240) * (1.1 - self.fatigue_resistance)
        
        # Fatigue decreases with rest
        fatigue_decrease = (rest_duration_minutes / 120) * self.recovery_rate
        
        self.current_fatigue = max(0, min(1, self.current_fatigue + fatigue_increase - fatigue_decrease))


@dataclass
class PlannedStint:
    """A planned stint assignment."""
    stint_number: int
    driver: Driver
    start_minute: float
    end_minute: float
    duration_minutes: float
    condition: DrivingCondition
    stint_type: StintType
    expected_laps: int
    notes: str = ""
    
    @property
    def duration_hours(self) -> float:
        return self.duration_minutes / 60


@dataclass 
class StintPlan:
    """Complete stint plan for a race."""
    stints: List[PlannedStint] = field(default_factory=list)
    total_race_duration_minutes: float = 0
    warnings: List[str] = field(default_factory=list)
    
    def get_driver_summary(self) -> Dict[str, Dict]:
        """Get summary of driving time per driver."""
        summary = {}
        for stint in self.stints:
            name = stint.driver.name
            if name not in summary:
                summary[name] = {
                    'total_minutes': 0,
                    'total_hours': 0,
                    'stints': 0,
                    'conditions': []
                }
            summary[name]['total_minutes'] += stint.duration_minutes
            summary[name]['total_hours'] = summary[name]['total_minutes'] / 60
            summary[name]['stints'] += 1
            if stint.condition.value not in summary[name]['conditions']:
                summary[name]['conditions'].append(stint.condition.value)
        
        return summary


class DriverStintPlanner:
    """
    Plans optimal driver rotations for endurance races.
    
    Considers:
    - FIA regulations on driving time
    - Driver skills for different conditions
    - Fatigue management
    - Strategic considerations
    """
    
    # FIA WEC Regulations (2024)
    MAX_CONTINUOUS_DRIVE_MINUTES = 240  # 4 hours max per stint
    MAX_TOTAL_DRIVE_24H_MINUTES = 840   # 14 hours max in 24h race
    MIN_DRIVERS_OVER_6H = 3
    
    def __init__(
        self,
        drivers: List[Driver],
        race_duration_hours: float,
        lap_time_seconds: float,
        pit_stop_duration_seconds: float = 60,
        driver_change_extra_seconds: float = 15
    ):
        """
        Initialize the planner.
        
        Args:
            drivers: List of Driver objects
            race_duration_hours: Total race duration in hours
            lap_time_seconds: Average lap time in seconds
            pit_stop_duration_seconds: Base pit stop time
            driver_change_extra_seconds: Extra time for driver change vs fuel-only stop
        """
        self.drivers = drivers
        self.race_duration_hours = race_duration_hours
        self.race_duration_minutes = race_duration_hours * 60
        self.lap_time_seconds = lap_time_seconds
        self.pit_stop_duration = pit_stop_duration_seconds
        self.driver_change_extra = driver_change_extra_seconds
        
        # Calculate derived values
        self.total_laps = int((race_duration_hours * 3600) / lap_time_seconds)
        
        # Calculate max drive time per driver (proportional to race length)
        proportion = min(1.0, race_duration_hours / 24)
        self.max_driver_total_minutes = self.MAX_TOTAL_DRIVE_24H_MINUTES * proportion
        
        # Reset all drivers
        for driver in self.drivers:
            driver.reset_for_new_race()
        
        logger.info(f"Stint Planner initialized:")
        logger.info(f"  - Race duration: {race_duration_hours}h ({self.total_laps} laps)")
        logger.info(f"  - {len(drivers)} drivers available")
        logger.info(f"  - Max drive time per driver: {self.max_driver_total_minutes/60:.1f}h")
    
    def _get_condition_at_time(
        self,
        race_minute: float,
        race_start_hour: int = 14,  # 2 PM default start
        weather_changes: Optional[List[Tuple[float, str]]] = None
    ) -> DrivingCondition:
        """
        Determine driving condition at a given race time.
        
        Args:
            race_minute: Minutes into the race
            race_start_hour: Hour of day when race starts (24h format)
            weather_changes: List of (minute, weather) tuples
        """
        # Calculate time of day
        current_hour = (race_start_hour + race_minute / 60) % 24
        is_night = current_hour < 6 or current_hour >= 20
        
        # Check weather
        is_wet = False
        if weather_changes:
            for change_minute, weather in sorted(weather_changes, reverse=True):
                if race_minute >= change_minute:
                    is_wet = weather in ['wet', 'rain', 'heavy_rain']
                    break
        
        if is_wet and is_night:
            return DrivingCondition.WET_NIGHT
        elif is_wet:
            return DrivingCondition.WET_DAY
        elif is_night:
            return DrivingCondition.DRY_NIGHT
        else:
            return DrivingCondition.DRY_DAY
    
    def _select_best_driver(
        self,
        condition: DrivingCondition,
        stint_type: StintType,
        stint_duration_minutes: float
    ) -> Optional[Driver]:
        """
        Select the best available driver for given conditions.
        
        Args:
            condition: Current driving condition
            stint_type: Type of stint
            stint_duration_minutes: Expected stint duration
        """
        candidates = []
        
        for driver in self.drivers:
            # Check if driver can do this stint (regulations)
            remaining_allowed = self.max_driver_total_minutes - driver.total_drive_time_minutes
            if remaining_allowed < stint_duration_minutes * 0.5:  # Need at least half stint capacity
                continue
            
            # Check fatigue
            if driver.current_fatigue > 0.8:
                continue
            
            # Calculate score for this driver
            score = driver.get_effective_pace(condition)
            
            # Bonus for consistency in closing stint
            if stint_type == StintType.CLOSING:
                score *= (0.8 + 0.2 * driver.consistency)
            
            # Bonus for pace in opening stint
            if stint_type == StintType.OPENING:
                score *= (0.9 + 0.1 * driver.pace)
            
            # Penalty for high fatigue
            score *= (1 - driver.current_fatigue * 0.3)
            
            # Bonus for fresh driver (hasn't driven recently)
            minutes_since_last = (self.race_duration_minutes - driver.last_stint_end_minute) if driver.stints_completed > 0 else 999
            if minutes_since_last > 120:
                score *= 1.05
            
            candidates.append((driver, score))
        
        if not candidates:
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_optimal_stint_count(self) -> int:
        """Calculate optimal number of stints based on race duration."""
        num_drivers = len(self.drivers)
        
        # Each driver should do roughly equal time
        target_time_per_driver = self.race_duration_minutes / num_drivers
        
        # Optimal stint length is 60-90 minutes for most races
        optimal_stint_minutes = 75
        
        # Calculate stints needed
        stints_from_time = int(self.race_duration_minutes / optimal_stint_minutes)
        
        # At minimum, each driver should drive at least twice (fairness + regulations)
        min_stints = num_drivers * 2
        
        return max(min_stints, stints_from_time)
    
    def create_equal_time_plan(
        self,
        race_start_hour: int = 14,
        weather_changes: Optional[List[Tuple[float, str]]] = None,
        target_stint_minutes: float = 75
    ) -> StintPlan:
        """
        Create a plan where all drivers get roughly equal driving time.
        
        Args:
            race_start_hour: Hour of day when race starts
            weather_changes: List of (minute, weather) tuples
            target_stint_minutes: Target stint duration
        """
        plan = StintPlan(total_race_duration_minutes=self.race_duration_minutes)
        
        # Validate driver count
        if self.race_duration_hours > 6 and len(self.drivers) < 3:
            plan.warnings.append(f"WARNING: Races over 6h require minimum 3 drivers, only {len(self.drivers)} provided")
        
        # Calculate number of stints
        num_stints = max(
            len(self.drivers) * 2,
            int(self.race_duration_minutes / target_stint_minutes)
        )
        
        # Distribute time roughly equally
        base_stint_duration = self.race_duration_minutes / num_stints
        
        current_minute = 0
        stint_number = 0
        driver_index = 0
        
        while current_minute < self.race_duration_minutes:
            stint_number += 1
            
            # Determine stint duration (vary slightly for realism)
            remaining = self.race_duration_minutes - current_minute
            if remaining < base_stint_duration * 1.5:
                # Last stint - take whatever remains
                stint_duration = remaining
            else:
                # Normal stint with slight variation
                stint_duration = base_stint_duration * np.random.uniform(0.9, 1.1)
                stint_duration = min(stint_duration, self.MAX_CONTINUOUS_DRIVE_MINUTES)
            
            # Get condition
            condition = self._get_condition_at_time(current_minute, race_start_hour, weather_changes)
            
            # Determine stint type
            if stint_number == 1:
                stint_type = StintType.OPENING
            elif current_minute + stint_duration >= self.race_duration_minutes - 30:
                stint_type = StintType.CLOSING
            elif condition in [DrivingCondition.DRY_NIGHT, DrivingCondition.WET_NIGHT]:
                stint_type = StintType.NIGHT
            else:
                stint_type = StintType.STANDARD
            
            # Select driver (round-robin with condition adjustment)
            driver = self._select_best_driver(condition, stint_type, stint_duration)
            
            if driver is None:
                # Fallback to round-robin if no suitable driver
                driver = self.drivers[driver_index % len(self.drivers)]
                plan.warnings.append(f"Stint {stint_number}: No optimal driver available, using {driver.name}")
            
            # Calculate expected laps
            expected_laps = int((stint_duration * 60) / self.lap_time_seconds)
            
            # Create stint
            stint = PlannedStint(
                stint_number=stint_number,
                driver=driver,
                start_minute=current_minute,
                end_minute=current_minute + stint_duration,
                duration_minutes=stint_duration,
                condition=condition,
                stint_type=stint_type,
                expected_laps=expected_laps
            )
            
            # Add notes based on conditions
            notes = []
            if stint_type == StintType.OPENING:
                notes.append("Opening stint - prioritize track position")
            elif stint_type == StintType.CLOSING:
                notes.append("Final stint - consistency over pace")
            if condition in [DrivingCondition.WET_DAY, DrivingCondition.WET_NIGHT]:
                notes.append(f"Wet conditions - {driver.name} wet skill: {driver.wet_skill:.0%}")
            if condition in [DrivingCondition.DRY_NIGHT, DrivingCondition.WET_NIGHT]:
                notes.append(f"Night stint - {driver.name} night skill: {driver.night_skill:.0%}")
            stint.notes = "; ".join(notes)
            
            plan.stints.append(stint)
            
            # Update driver state
            driver.total_drive_time_minutes += stint_duration
            driver.last_stint_end_minute = current_minute + stint_duration
            driver.stints_completed += 1
            driver.update_fatigue(stint_duration)
            
            # Update rest for other drivers
            for other in self.drivers:
                if other != driver:
                    other.update_fatigue(0, stint_duration)
            
            current_minute += stint_duration
            driver_index += 1
        
        # Check for regulation violations
        for driver in self.drivers:
            if driver.total_drive_time_minutes > self.max_driver_total_minutes:
                plan.warnings.append(
                    f"REGULATION VIOLATION: {driver.name} exceeds max drive time "
                    f"({driver.total_drive_time_minutes/60:.1f}h > {self.max_driver_total_minutes/60:.1f}h)"
                )
        
        return plan
    
    def create_condition_optimized_plan(
        self,
        race_start_hour: int = 14,
        weather_changes: Optional[List[Tuple[float, str]]] = None,
        night_specialist: Optional[str] = None,
        wet_specialist: Optional[str] = None,
        closer: Optional[str] = None
    ) -> StintPlan:
        """
        Create a plan optimized for specific conditions.
        
        Args:
            race_start_hour: Hour of day when race starts
            weather_changes: List of (minute, weather) tuples
            night_specialist: Name of driver who excels at night
            wet_specialist: Name of driver who excels in wet
            closer: Name of driver who should finish the race
        """
        # Start with equal time plan
        plan = self.create_equal_time_plan(race_start_hour, weather_changes)
        
        # Apply specialist overrides
        specialists = {}
        if night_specialist:
            for d in self.drivers:
                if d.name == night_specialist:
                    specialists['night'] = d
                    break
        if wet_specialist:
            for d in self.drivers:
                if d.name == wet_specialist:
                    specialists['wet'] = d
                    break
        if closer:
            for d in self.drivers:
                if d.name == closer:
                    specialists['closer'] = d
                    break
        
        # Swap drivers for specialist conditions (simplified)
        for stint in plan.stints:
            if stint.stint_type == StintType.CLOSING and 'closer' in specialists:
                stint.driver = specialists['closer']
                stint.notes += f"; Designated closer"
            elif stint.condition in [DrivingCondition.DRY_NIGHT, DrivingCondition.WET_NIGHT] and 'night' in specialists:
                original = stint.driver.name
                stint.driver = specialists['night']
                stint.notes += f"; Night specialist (was {original})"
            elif stint.condition in [DrivingCondition.WET_DAY, DrivingCondition.WET_NIGHT] and 'wet' in specialists:
                original = stint.driver.name
                stint.driver = specialists['wet']
                stint.notes += f"; Wet specialist (was {original})"
        
        return plan
    
    def optimize_for_safety_car_probability(
        self,
        sc_probability_by_hour: Dict[int, float],
        race_start_hour: int = 14
    ) -> StintPlan:
        """
        Create a plan considering Safety Car probability.
        
        In high SC probability periods, use shorter stints with faster drivers
        to maximize position gains during SC pit stops.
        
        Args:
            sc_probability_by_hour: Dict mapping race hour to SC probability
            race_start_hour: Hour of day when race starts
        """
        plan = StintPlan(total_race_duration_minutes=self.race_duration_minutes)
        
        current_minute = 0
        stint_number = 0
        
        while current_minute < self.race_duration_minutes:
            stint_number += 1
            race_hour = int(current_minute / 60)
            
            # Get SC probability for this hour
            sc_prob = sc_probability_by_hour.get(race_hour, 0.15)
            
            # Adjust stint length based on SC probability
            if sc_prob > 0.25:  # High SC probability
                target_stint = 45  # Shorter stints
                notes = "High SC probability - short stint for flexibility"
            elif sc_prob > 0.15:  # Medium SC probability
                target_stint = 60
                notes = "Medium SC probability"
            else:  # Low SC probability
                target_stint = 90  # Longer stints
                notes = "Low SC probability - maximize track time"
            
            remaining = self.race_duration_minutes - current_minute
            stint_duration = min(target_stint, remaining, self.MAX_CONTINUOUS_DRIVE_MINUTES)
            
            condition = self._get_condition_at_time(current_minute, race_start_hour)
            
            # For high SC periods, prioritize fastest drivers
            if sc_prob > 0.25:
                stint_type = StintType.SC_RESPONSE
                # Sort drivers by pace
                available = [d for d in self.drivers 
                           if (self.max_driver_total_minutes - d.total_drive_time_minutes) > stint_duration * 0.5
                           and d.current_fatigue < 0.8]
                if available:
                    available.sort(key=lambda d: d.pace, reverse=True)
                    driver = available[0]
                else:
                    driver = self.drivers[stint_number % len(self.drivers)]
            else:
                stint_type = StintType.STANDARD
                driver = self._select_best_driver(condition, stint_type, stint_duration)
                if driver is None:
                    driver = self.drivers[stint_number % len(self.drivers)]
            
            expected_laps = int((stint_duration * 60) / self.lap_time_seconds)
            
            stint = PlannedStint(
                stint_number=stint_number,
                driver=driver,
                start_minute=current_minute,
                end_minute=current_minute + stint_duration,
                duration_minutes=stint_duration,
                condition=condition,
                stint_type=stint_type,
                expected_laps=expected_laps,
                notes=notes
            )
            
            plan.stints.append(stint)
            
            # Update driver
            driver.total_drive_time_minutes += stint_duration
            driver.last_stint_end_minute = current_minute + stint_duration
            driver.stints_completed += 1
            driver.update_fatigue(stint_duration)
            
            current_minute += stint_duration
        
        plan.total_race_duration_minutes = self.race_duration_minutes
        return plan
    
    def format_plan_table(self, plan: StintPlan) -> str:
        """Format plan as readable table."""
        lines = []
        lines.append("=" * 100)
        lines.append(f"DRIVER STINT PLAN - {self.race_duration_hours}h Race")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'#':<4} {'Driver':<15} {'Start':<8} {'End':<8} {'Duration':<10} {'Laps':<6} {'Condition':<12} {'Type':<12}")
        lines.append("-" * 100)
        
        for stint in plan.stints:
            start_time = f"{int(stint.start_minute//60)}:{int(stint.start_minute%60):02d}"
            end_time = f"{int(stint.end_minute//60)}:{int(stint.end_minute%60):02d}"
            duration = f"{stint.duration_minutes:.0f}min"
            
            lines.append(
                f"{stint.stint_number:<4} {stint.driver.name:<15} {start_time:<8} {end_time:<8} "
                f"{duration:<10} {stint.expected_laps:<6} {stint.condition.value:<12} {stint.stint_type.value:<12}"
            )
            if stint.notes:
                lines.append(f"     └─ {stint.notes}")
        
        lines.append("")
        lines.append("-" * 100)
        lines.append("DRIVER SUMMARY:")
        lines.append("-" * 100)
        
        summary = plan.get_driver_summary()
        for name, data in summary.items():
            lines.append(
                f"  {name}: {data['total_hours']:.1f}h total, "
                f"{data['stints']} stints, conditions: {', '.join(data['conditions'])}"
            )
        
        if plan.warnings:
            lines.append("")
            lines.append("⚠️  WARNINGS:")
            for warning in plan.warnings:
                lines.append(f"  - {warning}")
        
        lines.append("=" * 100)
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Create example drivers for a WEC team
    drivers = [
        Driver(
            name="Alex (Pro)",
            pace=0.95,
            consistency=0.90,
            wet_skill=0.85,
            night_skill=0.80,
            tire_management=0.85,
            fatigue_resistance=0.85
        ),
        Driver(
            name="Ben (Silver)",
            pace=0.80,
            consistency=0.85,
            wet_skill=0.75,
            night_skill=0.90,  # Night specialist
            tire_management=0.80,
            fatigue_resistance=0.80
        ),
        Driver(
            name="Carlos (Bronze)",
            pace=0.70,
            consistency=0.80,
            wet_skill=0.90,  # Wet specialist
            night_skill=0.70,
            tire_management=0.75,
            fatigue_resistance=0.90  # Can do longer stints
        )
    ]
    
    # Create planner for 6h race
    planner = DriverStintPlanner(
        drivers=drivers,
        race_duration_hours=6,
        lap_time_seconds=138  # ~2:18 lap at Spa
    )
    
    # Test equal time plan
    print("\n" + "="*50)
    print("TEST 1: Equal Time Plan (6h race)")
    print("="*50)
    plan = planner.create_equal_time_plan(race_start_hour=14)
    print(planner.format_plan_table(plan))
    
    # Test with weather changes
    print("\n" + "="*50)
    print("TEST 2: With Weather Changes")
    print("="*50)
    weather = [
        (0, 'dry'),
        (120, 'wet'),  # Rain at 2h
        (180, 'dry'),  # Dries at 3h
    ]
    plan2 = planner.create_condition_optimized_plan(
        race_start_hour=14,
        weather_changes=weather,
        wet_specialist="Carlos (Bronze)"
    )
    print(planner.format_plan_table(plan2))
    
    # Test 24h race
    print("\n" + "="*50)
    print("TEST 3: 24h Race at Le Mans")
    print("="*50)
    planner_24h = DriverStintPlanner(
        drivers=drivers,
        race_duration_hours=24,
        lap_time_seconds=234  # ~3:54 lap
    )
    plan_24h = planner_24h.create_equal_time_plan(race_start_hour=16)  # 4 PM start
    print(planner_24h.format_plan_table(plan_24h))
