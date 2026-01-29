"""
WEC Data Scraper
===============
Scrapes timing data from FIA World Endurance Championship sources.

Note: This module is designed to work with publicly available data.
Always respect terms of service and rate limits when scraping.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import re
from urllib.parse import urljoin, unquote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WECScraper:
    """
    Scraper for WEC timing data from Al Kamel Systems.
    
    Data available includes:
    - Lap times
    - Pit stops  
    - Sector times
    - Race analysis
    
    Important: Data is owned by Al Kamel Systems S.L.
    This scraper is for educational/personal analysis only.
    """
    
    BASE_URL = "https://fiawec.alkamelsystems.com"
    
    # Known URL patterns for data
    URL_PATTERNS = {
        'results_index': '/Results/',
        'analysis_csv': 'Analysis_Race',
        'lap_chart': 'Lap_Chart',
        'classification': 'Classification',
        'pit_stops': 'Pit_Stop_Summary'
    }
    
    def __init__(self, data_dir: str = None, rate_limit: float = 2.0):
        """
        Initialize the WEC scraper.
        
        Args:
            data_dir: Directory to save downloaded data
            rate_limit: Minimum seconds between requests
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / 'data' / 'raw' / 'wec'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self._last_request = 0
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (educational research)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
    
    def _rate_limit_wait(self):
        """Ensure rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a rate-limited request."""
        self._rate_limit_wait()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_available_seasons(self) -> List[Dict]:
        """
        Get list of available seasons from WEC results.
        
        Returns:
            List of dictionaries with season info
        """
        url = f"{self.BASE_URL}/Results/"
        response = self._make_request(url)
        
        if not response:
            return []
            
        soup = BeautifulSoup(response.text, 'lxml')
        seasons = []
        
        # Parse season links (format: XX_YYYY-YYYY or XX_YYYY)
        for link in soup.find_all('a', href=True):
            href = link['href']
            match = re.search(r'(\d{2})_(\d{4}(?:-\d{4})?)', href)
            if match:
                seasons.append({
                    'season_id': match.group(1),
                    'season_name': match.group(2),
                    'url': urljoin(url, href)
                })
        
        return seasons
    
    def get_season_events(self, season_url: str) -> List[Dict]:
        """
        Get list of events for a specific season.
        
        Args:
            season_url: URL of the season page
            
        Returns:
            List of event dictionaries
        """
        response = self._make_request(season_url)
        
        if not response:
            return []
            
        soup = BeautifulSoup(response.text, 'lxml')
        events = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Event folders usually have format: XX_VENUE
            if re.match(r'\d{2}_[A-Z]', href.split('/')[-2] if '/' in href else ''):
                event_name = unquote(href.split('/')[-2])
                events.append({
                    'event_name': event_name,
                    'url': urljoin(season_url, href)
                })
        
        return events
    
    def download_analysis_csv(
        self, 
        event_url: str,
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download race analysis CSV from an event.
        
        The analysis CSV contains:
        - Lap numbers
        - Lap times
        - Sector times
        - Driver info
        - Pit indicators
        
        Args:
            event_url: URL of the event page
            save: Whether to save the CSV locally
            
        Returns:
            DataFrame with lap analysis data
        """
        response = self._make_request(event_url)
        
        if not response:
            return None
            
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Look for race session
        race_links = []
        for link in soup.find_all('a', href=True):
            if 'Race' in link.get_text() or 'race' in link['href'].lower():
                race_links.append(urljoin(event_url, link['href']))
        
        if not race_links:
            logger.warning(f"No race session found at {event_url}")
            return None
        
        # Navigate to race session and find CSVs
        race_url = race_links[0]
        race_response = self._make_request(race_url)
        
        if not race_response:
            return None
            
        race_soup = BeautifulSoup(race_response.text, 'lxml')
        
        # Find analysis CSV links
        csv_links = []
        for link in race_soup.find_all('a', href=True):
            if link['href'].endswith('.CSV') and 'Analysis' in link['href']:
                csv_links.append(urljoin(race_url, link['href']))
        
        if not csv_links:
            logger.warning(f"No analysis CSV found at {race_url}")
            return None
        
        # Download the most complete analysis (usually the last one)
        csv_url = csv_links[-1]
        csv_response = self._make_request(csv_url)
        
        if not csv_response:
            return None
        
        # Parse CSV
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(csv_response.text))
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            if save:
                filename = self._generate_filename(csv_url)
                df.to_csv(self.data_dir / filename, index=False)
                logger.info(f"Saved: {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return None
    
    def _generate_filename(self, url: str) -> str:
        """Generate a clean filename from URL."""
        parts = url.split('/')
        # Extract meaningful parts
        relevant = [p for p in parts if p and not p.startswith('http')][-3:]
        clean = '_'.join(relevant).replace('%20', '_').replace('.CSV', '.csv')
        return clean
    
    def process_analysis_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process raw analysis CSV into structured DataFrames.
        
        Args:
            df: Raw analysis DataFrame
            
        Returns:
            Dictionary with processed DataFrames
        """
        processed = {}
        
        # Standardize column names
        column_mapping = {
            'NUMBER': 'car_number',
            'CAR_NUMBER': 'car_number',
            'LAP_NUMBER': 'lap_number',
            'LAP_TIME': 'lap_time',
            'LAP TIME': 'lap_time',
            'S1': 'sector_1',
            'S2': 'sector_2',
            'S3': 'sector_3',
            'PIT_TIME': 'pit_time',
            'DRIVER_NUMBER': 'driver_number',
            'CLASS': 'class',
            'TEAM': 'team'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert lap times to seconds if needed
        if 'lap_time' in df.columns:
            df['lap_time_seconds'] = df['lap_time'].apply(self._parse_time)
        
        for col in ['sector_1', 'sector_2', 'sector_3']:
            if col in df.columns:
                df[f'{col}_seconds'] = df[col].apply(self._parse_time)
        
        # Identify pit stops
        if 'pit_time' in df.columns:
            pit_stops = df[df['pit_time'].notna() & (df['pit_time'] != '')].copy()
            pit_stops['pit_time_seconds'] = pit_stops['pit_time'].apply(self._parse_time)
            processed['pit_stops'] = pit_stops
        
        processed['lap_times'] = df
        
        return processed
    
    def _parse_time(self, time_str) -> Optional[float]:
        """Parse time string to seconds."""
        if pd.isna(time_str) or time_str == '':
            return None
            
        time_str = str(time_str).strip()
        
        try:
            # Format: M:SS.mmm or MM:SS.mmm
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                # Just seconds
                return float(time_str)
        except (ValueError, IndexError):
            return None
    
    def download_season(
        self, 
        season: str = '2024',
        events: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all data for a season.
        
        Args:
            season: Season year/name
            events: Specific events to download (None = all)
            
        Returns:
            Dictionary with all downloaded data
        """
        seasons = self.get_available_seasons()
        
        season_info = next(
            (s for s in seasons if season in s['season_name']),
            None
        )
        
        if not season_info:
            logger.error(f"Season {season} not found")
            return {}
        
        season_events = self.get_season_events(season_info['url'])
        
        if events:
            season_events = [e for e in season_events if any(
                ev.lower() in e['event_name'].lower() for ev in events
            )]
        
        all_data = {'lap_times': [], 'pit_stops': []}
        
        for event in season_events:
            logger.info(f"Downloading: {event['event_name']}")
            
            df = self.download_analysis_csv(event['url'])
            
            if df is not None:
                processed = self.process_analysis_data(df)
                processed['lap_times']['event'] = event['event_name']
                processed['lap_times']['season'] = season
                
                all_data['lap_times'].append(processed['lap_times'])
                
                if 'pit_stops' in processed:
                    processed['pit_stops']['event'] = event['event_name']
                    all_data['pit_stops'].append(processed['pit_stops'])
        
        # Combine all events
        result = {}
        for key, dfs in all_data.items():
            if dfs:
                result[key] = pd.concat(dfs, ignore_index=True)
        
        return result
    
    def get_sample_url_structure(self) -> str:
        """
        Return example URL structure for manual data download.
        
        This is useful if automated scraping is blocked.
        """
        return """
WEC Data URL Structure (Al Kamel Systems)
=========================================

Base URL: https://fiawec.alkamelsystems.com/Results/

Season folders: XX_YYYY-YYYY/ (e.g., 13_2024/)
Event folders: XX_VENUE/ (e.g., 07_SPA/)

CSV files typically found in:
/Results/{Season}/{Event}/{Series}/Race/HourX/

Example full URL:
http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV

CSV columns typically include:
- NUMBER (car number)
- DRIVER_NUMBER
- LAP_NUMBER
- LAP_TIME
- S1, S2, S3 (sector times)
- PIT_TIME (if applicable)
- CLASS
- GROUP
- TEAM

Note: Data is owned by Al Kamel Systems S.L.
For commercial use, contact them for licensing.
"""


def main():
    """Demo scraper functionality."""
    print("WEC Data Scraper")
    print("=" * 50)
    print("\nNote: This scraper respects rate limits and TOS.")
    print("For production use, consider official data licensing.\n")
    
    scraper = WECScraper()
    
    # Show URL structure
    print(scraper.get_sample_url_structure())
    
    # Try to get available seasons
    print("\nAttempting to fetch available seasons...")
    seasons = scraper.get_available_seasons()
    
    if seasons:
        print(f"Found {len(seasons)} seasons:")
        for s in seasons[:5]:
            print(f"  - {s['season_name']}")
    else:
        print("Could not fetch seasons (site may require different access method)")
        print("\nAlternative: Download CSV files manually and place in data/raw/wec/")


if __name__ == '__main__':
    main()
