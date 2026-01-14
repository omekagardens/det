#!/usr/bin/env python3
"""
GPS Data Loader for DET Validation
===================================

Tools for downloading and parsing real GPS satellite clock data from IGS.

Data Sources:
- IGS (International GNSS Service): https://igs.org/products/
- CDDIS (NASA): https://cddis.nasa.gov/archive/gnss/products/

File Formats:
- SP3: Standard Product 3 format (orbits + clocks)
- CLK: RINEX Clock format (high-rate clocks)

Usage:
    from gps_data_loader import GPSDataLoader, parse_sp3_file

    # Download and parse IGS final products
    loader = GPSDataLoader()
    data = loader.download_sp3_for_date(2024, 1, 15)

    # Or parse a local file
    satellites = parse_sp3_file("igs23456.sp3")
"""

import numpy as np
import os
import gzip
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import ssl


# Physical constants
C_SI = 299792458.0  # m/s
GM_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.371e6  # m


@dataclass
class GPSSatellite:
    """Data for a single GPS satellite at one epoch."""
    prn: str  # e.g., "G01" for GPS PRN 01
    x: float  # ECEF X position (km)
    y: float  # ECEF Y position (km)
    z: float  # ECEF Z position (km)
    clock: float  # Clock offset (microseconds)
    epoch: datetime

    @property
    def radius_km(self) -> float:
        """Distance from Earth center in km."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def altitude_km(self) -> float:
        """Altitude above Earth surface in km."""
        return self.radius_km - R_EARTH/1000

    @property
    def clock_seconds(self) -> float:
        """Clock offset in seconds."""
        return self.clock * 1e-6

    @property
    def gravitational_potential(self) -> float:
        """Gravitational potential at satellite (m²/s²)."""
        r = self.radius_km * 1000  # Convert to meters
        return -GM_EARTH / r


@dataclass
class SP3Epoch:
    """Data for all satellites at one epoch."""
    epoch: datetime
    satellites: Dict[str, GPSSatellite] = field(default_factory=dict)


@dataclass
class SP3Data:
    """Parsed SP3 file data."""
    filename: str
    agency: str = ""
    version: str = ""
    start_epoch: datetime = None
    end_epoch: datetime = None
    num_epochs: int = 0
    interval_seconds: float = 900.0  # Default 15 min
    epochs: List[SP3Epoch] = field(default_factory=list)
    satellites_in_file: List[str] = field(default_factory=list)

    def get_satellite_data(self, prn: str) -> List[GPSSatellite]:
        """Get all epochs for a specific satellite."""
        data = []
        for epoch in self.epochs:
            if prn in epoch.satellites:
                data.append(epoch.satellites[prn])
        return data

    def get_clock_offsets_per_day(self, prn: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get clock offsets over time for a satellite.

        Returns:
            times: Array of hours from start
            clocks: Array of clock offsets in microseconds
        """
        data = self.get_satellite_data(prn)
        if not data:
            return np.array([]), np.array([])

        t0 = data[0].epoch
        times = np.array([(d.epoch - t0).total_seconds() / 3600 for d in data])
        clocks = np.array([d.clock for d in data])
        return times, clocks

    def summary(self) -> str:
        """Return summary of the data."""
        lines = [
            f"SP3 Data: {self.filename}",
            f"Agency: {self.agency}",
            f"Time span: {self.start_epoch} to {self.end_epoch}",
            f"Epochs: {self.num_epochs}",
            f"Interval: {self.interval_seconds}s",
            f"Satellites: {len(self.satellites_in_file)}",
        ]
        if self.epochs:
            # Sample one satellite
            prn = self.satellites_in_file[0] if self.satellites_in_file else None
            if prn:
                data = self.get_satellite_data(prn)
                if data:
                    lines.append(f"\nSample ({prn}):")
                    lines.append(f"  Altitude: {data[0].altitude_km:.0f} km")
                    lines.append(f"  Clock offset: {data[0].clock:.3f} μs")
        return "\n".join(lines)


def parse_sp3_file(filepath: str) -> SP3Data:
    """
    Parse an SP3 format file.

    SP3 format reference: https://files.igs.org/pub/data/format/sp3c.txt

    The format has:
    - Header lines starting with # or +
    - Epoch lines starting with *
    - Position/clock lines starting with P (or V for velocity)
    """
    result = SP3Data(filename=os.path.basename(filepath))

    # Handle gzipped files
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            lines = f.readlines()
    else:
        with open(filepath, 'r') as f:
            lines = f.readlines()

    current_epoch = None
    satellites = set()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Header line 1: version, mode, start epoch
        if line.startswith('#'):
            if line[1] in 'acdP':  # Version indicator
                result.version = line[1]
                # Parse start epoch from first header line
                try:
                    year = int(line[3:7])
                    month = int(line[8:10])
                    day = int(line[11:13])
                    hour = int(line[14:16])
                    minute = int(line[17:19])
                    second = float(line[20:31])
                    result.start_epoch = datetime(year, month, day, hour, minute, int(second))
                except (ValueError, IndexError):
                    pass

        # Agency identifier (line 2)
        elif line.startswith('##'):
            try:
                result.interval_seconds = float(line[24:38])
            except (ValueError, IndexError):
                pass

        # Satellite list
        elif line.startswith('+'):
            # Lines starting with + contain satellite PRNs
            parts = line[9:].split()
            for part in parts:
                if part and part[0] in 'GREJCIS':
                    satellites.add(part)

        # Epoch line
        elif line.startswith('*'):
            try:
                year = int(line[3:7])
                month = int(line[8:10])
                day = int(line[11:13])
                hour = int(line[14:16])
                minute = int(line[17:19])
                second = float(line[20:31])
                epoch_time = datetime(year, month, day, hour, minute, int(second))

                current_epoch = SP3Epoch(epoch=epoch_time)
                result.epochs.append(current_epoch)
                result.num_epochs += 1

                if result.end_epoch is None or epoch_time > result.end_epoch:
                    result.end_epoch = epoch_time
            except (ValueError, IndexError):
                pass

        # Position/clock line
        elif line.startswith('P') and current_epoch is not None:
            try:
                prn = line[1:4].strip()
                x = float(line[4:18])  # km
                y = float(line[18:32])  # km
                z = float(line[32:46])  # km
                clock = float(line[46:60])  # microseconds

                # Skip bad values (999999.999999 indicates no data)
                if abs(x) > 100000 or abs(y) > 100000 or abs(z) > 100000:
                    continue
                if abs(clock) > 1e6:  # Bad clock value
                    continue

                sat = GPSSatellite(
                    prn=prn,
                    x=x, y=y, z=z,
                    clock=clock,
                    epoch=current_epoch.epoch
                )
                current_epoch.satellites[prn] = sat
                satellites.add(prn)
            except (ValueError, IndexError):
                pass

    result.satellites_in_file = sorted(list(satellites))
    return result


def gps_week_from_date(date: datetime) -> Tuple[int, int]:
    """
    Convert a date to GPS week and day of week.

    GPS epoch: January 6, 1980 00:00:00 UTC
    """
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    weeks = delta.days // 7
    day_of_week = delta.days % 7
    return weeks, day_of_week


def date_from_gps_week(week: int, day: int = 0) -> datetime:
    """Convert GPS week and day to date."""
    gps_epoch = datetime(1980, 1, 6)
    return gps_epoch + timedelta(days=week*7 + day)


class GPSDataLoader:
    """
    Loader for GPS data from IGS archives.

    Can download:
    - Final products (highest quality, ~2 week delay)
    - Rapid products (~17 hour delay)
    - Ultra-rapid products (real-time)
    """

    # IGS data mirrors
    MIRRORS = [
        "https://noaa-cors-pds.s3.amazonaws.com/rinex",  # NOAA (no auth)
        "https://igs.bkg.bund.de/root_ftp/IGS/products",
        "https://igs.ign.fr/pub/igs/products",
    ]

    def __init__(self, cache_dir: str = None):
        """
        Initialize GPS data loader.

        Args:
            cache_dir: Directory for caching downloaded files
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '.gps_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_sp3_urls(self, week: int, day: int, product: str = "final",
                      date: datetime = None) -> List[str]:
        """
        Get URLs for SP3 file from multiple mirrors.

        Args:
            week: GPS week number
            day: Day of week (0=Sunday)
            product: "final", "rapid", or "ultra"
            date: Date object (needed for NOAA format)

        Returns:
            List of URLs to try
        """
        urls = []

        # NOAA format: YYYY/DDD/igsWWWWD.sp3.gz
        if date:
            doy = date.timetuple().tm_yday
            noaa_url = f"https://noaa-cors-pds.s3.amazonaws.com/rinex/{date.year}/{doy:03d}/igs{week:04d}{day}.sp3.gz"
            urls.append(noaa_url)

        # IGS format
        if product == "final":
            filename = f"igs{week:04d}{day}.sp3.Z"
        elif product == "rapid":
            filename = f"igr{week:04d}{day}.sp3.Z"
        else:
            filename = f"igu{week:04d}{day}_00.sp3.Z"

        for mirror in self.MIRRORS[1:]:  # Skip NOAA (already handled)
            urls.append(f"{mirror}/{week:04d}/{filename}")

        return urls

    def get_sp3_url(self, week: int, day: int, product: str = "final") -> str:
        """Legacy method - returns first URL."""
        urls = self.get_sp3_urls(week, day, product)
        return urls[0] if urls else ""

    def download_file(self, url: str, local_path: str) -> bool:
        """
        Download a file from URL.

        Returns:
            True if successful, False otherwise
        """
        print(f"  Downloading: {url}")

        # Create SSL context that doesn't verify (for some mirrors)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'DET-Validation/1.0')

            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                data = response.read()

            with open(local_path, 'wb') as f:
                f.write(data)

            return True

        except urllib.error.HTTPError as e:
            print(f"  HTTP Error {e.code}: {e.reason}")
            return False
        except urllib.error.URLError as e:
            print(f"  URL Error: {e.reason}")
            return False
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def download_sp3_for_date(self, year: int, month: int, day: int,
                              product: str = "final") -> Optional[SP3Data]:
        """
        Download and parse SP3 data for a specific date.

        Args:
            year, month, day: Date to download
            product: "final", "rapid", or "ultra"

        Returns:
            Parsed SP3Data or None if download failed
        """
        date = datetime(year, month, day)
        week, dow = gps_week_from_date(date)

        print(f"Downloading GPS data for {date.date()} (week {week}, day {dow})")

        # Check cache first
        cache_file = os.path.join(
            self.cache_dir,
            f"{product}_{week:04d}_{dow}.sp3"
        )

        if os.path.exists(cache_file):
            print(f"  Using cached file: {cache_file}")
            return parse_sp3_file(cache_file)

        # Try multiple URLs
        urls = self.get_sp3_urls(week, dow, product, date)

        for url in urls:
            # Determine extension
            if url.endswith('.gz'):
                compressed_path = cache_file + ".gz"
            else:
                compressed_path = cache_file + ".Z"

            if self.download_file(url, compressed_path):
                # Decompress
                try:
                    # Try gzip first (most common)
                    try:
                        with gzip.open(compressed_path, 'rb') as f_in:
                            with open(cache_file, 'wb') as f_out:
                                f_out.write(f_in.read())
                        if os.path.exists(cache_file):
                            return parse_sp3_file(cache_file)
                    except:
                        pass

                    # Try Unix compress format
                    try:
                        import subprocess
                        subprocess.run(['uncompress', '-f', compressed_path],
                                     check=True, capture_output=True)
                        if os.path.exists(cache_file):
                            return parse_sp3_file(cache_file)
                    except:
                        pass

                    # Just try to parse the compressed file directly
                    return parse_sp3_file(compressed_path)

                except Exception as e:
                    print(f"  Decompression error: {e}")
                    # Try parsing directly
                    try:
                        return parse_sp3_file(compressed_path)
                    except:
                        continue

        return None

    def get_sample_data(self) -> SP3Data:
        """
        Return sample GPS data for testing (no download required).

        Based on typical GPS satellite parameters.
        """
        # Create synthetic data based on real GPS parameters
        # GPS orbital radius: ~26,560 km
        # Orbital period: ~11.97 hours
        # Typical clock offset: varies, but ~microseconds

        data = SP3Data(
            filename="synthetic_sample.sp3",
            agency="DET-TEST",
            version="c",
            start_epoch=datetime(2024, 1, 1, 0, 0, 0),
            interval_seconds=900.0
        )

        # Generate 24 hours of data at 15-min intervals
        R_orbit = 26560.0  # km (GPS orbital radius)
        T_orbit = 11.97 * 3600  # seconds (orbital period)
        omega = 2 * np.pi / T_orbit  # rad/s

        # GPS satellites (typical PRNs)
        prns = [f"G{i:02d}" for i in range(1, 32)]

        for epoch_num in range(96):  # 96 epochs = 24 hours at 15-min
            t = epoch_num * 900  # seconds from start
            epoch_time = data.start_epoch + timedelta(seconds=t)

            epoch = SP3Epoch(epoch=epoch_time)

            for i, prn in enumerate(prns):
                # Different orbital phases for each satellite
                phase = (2 * np.pi * i / len(prns)) + omega * t

                # Simplified circular orbit
                x = R_orbit * np.cos(phase)
                y = R_orbit * np.sin(phase) * 0.9  # Slight inclination effect
                z = R_orbit * np.sin(phase) * 0.1

                # Clock offset: systematic + periodic + noise
                # Systematic: ~+38 μs/day from GR effects (we'll add this as a trend)
                # Periodic: from eccentricity
                base_clock = 100.0 + i * 10  # Base offset varies by satellite
                trend = (t / 86400) * 38.5  # ~38.5 μs/day systematic
                periodic = 0.05 * np.sin(omega * t)  # Small periodic term
                clock = base_clock + trend + periodic

                sat = GPSSatellite(
                    prn=prn,
                    x=x, y=y, z=z,
                    clock=clock,
                    epoch=epoch_time
                )
                epoch.satellites[prn] = sat

            data.epochs.append(epoch)
            data.num_epochs += 1

        data.end_epoch = data.epochs[-1].epoch
        data.satellites_in_file = prns

        return data


def compute_clock_rate_from_data(data: SP3Data, prn: str) -> Dict:
    """
    Analyze clock offset data to extract clock rate (drift).

    The clock offset includes:
    1. Initial offset (arbitrary)
    2. Clock drift (can compare to DET prediction)
    3. Periodic terms (from orbital motion)

    Returns dict with:
        - drift_per_day: Linear drift rate (μs/day)
        - periodic_amplitude: Amplitude of periodic terms (μs)
        - residual_rms: RMS of residuals after removing drift
    """
    times, clocks = data.get_clock_offsets_per_day(prn)

    if len(times) < 2:
        return {'error': 'Insufficient data'}

    # Convert times to days
    times_days = times / 24.0

    # Linear fit for drift
    coeffs = np.polyfit(times_days, clocks, 1)
    drift_per_day = coeffs[0]  # μs/day
    offset = coeffs[1]  # μs

    # Remove linear trend
    residuals = clocks - (drift_per_day * times_days + offset)

    # Estimate periodic amplitude
    periodic_amplitude = (np.max(residuals) - np.min(residuals)) / 2

    # RMS of residuals
    residual_rms = np.sqrt(np.mean(residuals**2))

    # Get satellite info
    sat_data = data.get_satellite_data(prn)
    if sat_data:
        altitude_km = sat_data[0].altitude_km
        potential = sat_data[0].gravitational_potential
    else:
        altitude_km = 0
        potential = 0

    return {
        'prn': prn,
        'drift_per_day_us': drift_per_day,
        'drift_per_day_s': drift_per_day * 1e-6,
        'periodic_amplitude_us': periodic_amplitude,
        'residual_rms_us': residual_rms,
        'altitude_km': altitude_km,
        'potential_m2_s2': potential,
        'num_epochs': len(times)
    }


def validate_gps_against_det(data: SP3Data, verbose: bool = True) -> Dict:
    """
    Validate DET predictions against real GPS clock data.

    DET predicts:
    - Gravitational time dilation: ~+45.7 μs/day (clocks run fast at altitude)
    - Kinematic time dilation: ~-7.2 μs/day (velocity effect)
    - Net effect: ~+38.5 μs/day

    Returns validation results.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GPS DATA VALIDATION AGAINST DET")
        print("=" * 60)

    results = {
        'satellites': [],
        'summary': {}
    }

    # Expected from GR/DET
    expected_grav_drift = 45.7  # μs/day (gravitational)
    expected_kin_drift = -7.2  # μs/day (kinematic)
    expected_net_drift = 38.5  # μs/day

    drift_rates = []

    for prn in data.satellites_in_file[:10]:  # Analyze first 10 satellites
        if not prn.startswith('G'):  # Only GPS satellites
            continue

        analysis = compute_clock_rate_from_data(data, prn)

        if 'error' not in analysis:
            drift_rates.append(analysis['drift_per_day_us'])
            results['satellites'].append(analysis)

            if verbose:
                print(f"\n{prn}:")
                print(f"  Altitude: {analysis['altitude_km']:.0f} km")
                print(f"  Clock drift: {analysis['drift_per_day_us']:.2f} μs/day")
                print(f"  Periodic amplitude: {analysis['periodic_amplitude_us']:.3f} μs")

    if drift_rates:
        mean_drift = np.mean(drift_rates)
        std_drift = np.std(drift_rates)

        # Note: The raw drift includes the satellite clock's intrinsic drift
        # which varies by satellite. The relativistic effect is a systematic
        # offset that's removed by the control segment.

        results['summary'] = {
            'num_satellites': len(drift_rates),
            'mean_drift_us_day': mean_drift,
            'std_drift_us_day': std_drift,
            'expected_grav_us_day': expected_grav_drift,
            'expected_net_us_day': expected_net_drift
        }

        if verbose:
            print("\n" + "-" * 60)
            print("SUMMARY")
            print("-" * 60)
            print(f"Satellites analyzed: {len(drift_rates)}")
            print(f"Mean observed drift: {mean_drift:.2f} ± {std_drift:.2f} μs/day")
            print(f"Expected (GR/DET gravitational): +{expected_grav_drift:.2f} μs/day")
            print(f"Expected (net with kinematic): +{expected_net_drift:.2f} μs/day")
            print()
            print("Note: Raw SP3 clock data includes satellite-specific offsets")
            print("and drifts. The relativistic correction is applied by the GPS")
            print("control segment before upload to satellites.")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='GPS Data Loader for DET Validation')
    parser.add_argument('--date', type=str, help='Date to download (YYYY-MM-DD)')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    parser.add_argument('--file', type=str, help='Parse local SP3 file')

    args = parser.parse_args()

    if args.file:
        print(f"Parsing: {args.file}")
        data = parse_sp3_file(args.file)
        print(data.summary())
        validate_gps_against_det(data)

    elif args.sample:
        print("Using synthetic sample data...")
        loader = GPSDataLoader()
        data = loader.get_sample_data()
        print(data.summary())
        validate_gps_against_det(data)

    elif args.date:
        parts = args.date.split('-')
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

        loader = GPSDataLoader()
        data = loader.download_sp3_for_date(year, month, day, product="final")

        if data:
            print(data.summary())
            validate_gps_against_det(data)
        else:
            print("Failed to download data. Try --sample for synthetic data.")

    else:
        # Default: use sample data
        print("No input specified. Using synthetic sample data...")
        print("(Use --date YYYY-MM-DD to download real data)")
        print()

        loader = GPSDataLoader()
        data = loader.get_sample_data()
        print(data.summary())
        validate_gps_against_det(data)
