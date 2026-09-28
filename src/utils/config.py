"""
Central configuration loader.
Every module imports `get_config()` to read settings from config/config.yaml.
Never hardcode paths or parameters in other modules.
"""
from pathlib import Path
from functools import lru_cache
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache config.yaml.
    
    @lru_cache ensures the file is read only once per Python process —
    subsequent calls return the cached dict instantly.
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Are you running from the project root?"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str) -> Path:
    """
    Return an absolute Path for a path defined in config.paths.
    Example: get_path('data_raw') -> <project>/data/raw
    """
    cfg = get_config()
    relative = cfg["paths"][key]
    return PROJECT_ROOT / relative