"""Central config: which cities we track, and where to find their coordinates
and AQICN station IDs. Add a new city by adding one entry to CITIES."""

from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class CityConfig:
    slug: str
    display_name: str
    latitude: float
    longitude: float
    aqicn_station_uid: int


CITIES = {
    "islamabad": CityConfig(
        slug="islamabad",
        display_name="Islamabad, Pakistan",
        latitude=33.6844,
        longitude=73.0479,
        aqicn_station_uid=11739,  # "Islamabad US Embassy" station on aqicn.org
    ),
}

DEFAULT_CITY = "islamabad"
