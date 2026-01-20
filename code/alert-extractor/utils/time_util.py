"""
Time Utilities for Alert Extraction
"""
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

def convert_datetime_to_epoch(ctime: str, timezone: str = "UTC") -> int:
    """
    Converts a datetime string to a Unix timestamp in milliseconds for the given timezone.

    Accepts formats:
    - 'YYYY-MM-DD HH:MM:SS.%f'
    - 'YYYY-MM-DD HH:MM:SS'
    - 'YYYY-MM-DD'

    Args:
        ctime (str): The input datetime string
        
    Returns:
        int: Unix timestamp in milliseconds since epoch.
    """
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            dt = datetime.strptime(ctime, fmt)
            dt = dt.replace(tzinfo=ZoneInfo(timezone))
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {ctime}")

def convert_epoch_to_datetime_str(epoch_ms: int, timezone: str = "UTC") -> str:
    """
    Converts an epoch timestamp in milliseconds to a datetime string 
    (e.g., '2022-03-20 14:32:00.123') in the given timezone.

    Args:
        epoch_ms (int): Epoch timestamp in milliseconds.
        timezone (str): Timezone name (e.g., 'UTC' or'Asia/Brunei').

    Returns:
        str: Formatted datetime string with milliseconds.
    """
    tz = pytz.timezone(timezone)
    dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # trim to 3 decimal places (ms)
