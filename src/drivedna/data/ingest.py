"""Download nflverse play-by-play and cache it locally as Parquet."""
from pathlib import Path

import nflreadpy as nfl

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
DEFAULT_SEASONS = list(range(2016, 2026))


def pbp_path(season: int) -> Path:
    return RAW_DIR / f"pbp_{season}.parquet"


def participation_path(season: int) -> Path:
    return RAW_DIR / f"participation_{season}.parquet"


def ingest_participation(seasons=DEFAULT_SEASONS, force: bool = False) -> list[Path]:
    """Defenders in box, pass rushers, pressure, man/zone and coverage (man/zone and coverage from 2018)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["nflverse_game_id", "play_id", "offense_formation", "defenders_in_box", "number_of_pass_rushers",
            "was_pressure", "time_to_throw", "defense_man_zone_type", "defense_coverage_type"]
    paths = []
    for season in seasons:
        path = participation_path(season)
        if force or not path.exists():
            print(f"downloading participation {season} ...")
            nfl.load_participation([season]).select(keep).write_parquet(path)
        paths.append(path)
    return paths


def ingest_pbp(seasons=DEFAULT_SEASONS, force: bool = False) -> list[Path]:
    """Fetch regular-season + playoff pbp per season; skip seasons already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        path = pbp_path(season)
        if force or not path.exists():
            print(f"downloading {season} ...")
            nfl.load_pbp([season]).write_parquet(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    ingest_pbp()
    ingest_participation()
