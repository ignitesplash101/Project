"""Convenience helpers for the MLIT Real Estate Information Library demo notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

import os

import requests
from dotenv import load_dotenv

BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external"
_API_KEY_CACHE: str | None = None


@dataclass(frozen=True)
class EndpointSpec:
    """Metadata that describes one MLIT endpoint."""

    code: str
    name: str
    description: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def with_defaults(self, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(self.defaults)
        if overrides:
            params.update(overrides)
        return params


# Shared defaults used by many tile/mesh endpoints.
_TILE_DEFAULTS: Dict[str, Any] = {
    "response_format": "geojson",
    "z": 13,
    "x": 7301,
    "y": 3152,
}


_SPEC_LIST: tuple[EndpointSpec, ...] = (
    EndpointSpec(
        code="XIT001",
        name="Real Estate Transaction Prices",
        description="Quarterly transaction and contract price records.",
        required=("year",),
        optional=("priceClassification", "quarter", "area", "city", "station", "language"),
        defaults={"year": 2024, "priceClassification": "01", "area": "04", "quarter": 1, "language": "en"},
    ),
    EndpointSpec(
        code="XIT002",
        name="Prefecture Municipality Directory",
        description="Lists municipalities for a prefecture.",
        required=("area",),
        optional=("language",),
        defaults={"area": "04", "language": "en"},
    ),
    EndpointSpec(
        code="XCT001",
        name="Appraisal Report Information",
        description="Land appraisal sheets published by MLIT.",
        required=("year", "area", "division"),
        defaults={"year": 2024, "area": "04", "division": "00"},
    ),
    EndpointSpec(
        code="XPT001",
        name="Transaction Price Points",
        description="Real estate transaction points (tile feed).",
        required=("response_format", "z", "x", "y", "from", "to"),
        optional=("priceClassification", "landTypeCode"),
        defaults={
            **_TILE_DEFAULTS,
            "from": "20231",
            "to": "20234",
            "priceClassification": "01",
        },
    ),
    EndpointSpec(
        code="XPT002",
        name="Land Price Survey Points",
        description="Land price public survey points (tile feed).",
        required=("response_format", "z", "x", "y", "year"),
        optional=("priceClassification", "useCategoryCode"),
        defaults={**_TILE_DEFAULTS, "year": 2024},
    ),
    EndpointSpec(
        code="XKT001",
        name="City Planning GIS – Planning Areas",
        description="City planning areas / classifications.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT002",
        name="City Planning GIS – Zoning",
        description="City planning zoning boundaries.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT003",
        name="City Planning GIS – Location Optimization Plan",
        description="Location optimization plan layers.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT004",
        name="National Land Info – Elementary School Districts",
        description="Elementary school catchment polygons.",
        required=("response_format", "z", "x", "y"),
        optional=("administrativeAreaCode",),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT005",
        name="National Land Info – Junior High School Districts",
        description="Junior high school catchment polygons.",
        required=("response_format", "z", "x", "y"),
        optional=("administrativeAreaCode",),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT006",
        name="National Land Info – Schools",
        description="School facility point features.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT007",
        name="National Land Info – Nursery & Kindergarten Facilities",
        description="Nursery, kindergarten, and daycare facilities.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT010",
        name="National Land Info – Medical Institutions",
        description="Hospital and clinic facilities.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT011",
        name="National Land Info – Welfare Facilities",
        description="Welfare facilities with optional facility classes.",
        required=("response_format", "z", "x", "y"),
        optional=(
            "administrativeAreaCode",
            "welfareFacilityClassCode",
            "welfareFacilityMiddleClassCode",
            "welfareFacilityMinorClassCode",
        ),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT013",
        name="National Land Info – Future Population 250m Mesh",
        description="Future population estimates at 250m resolution.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT014",
        name="City Planning GIS – Fire / Quasi-Fire Zones",
        description="Fire-proof and quasi fire-proof designation areas.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT015",
        name="National Land Info – Station Passenger Volume",
        description="Passenger counts by station.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT016",
        name="National Land Info – Disaster Hazard Areas",
        description="Designated disaster hazard polygons.",
        required=("response_format", "z", "x", "y"),
        optional=("administrativeAreaCode",),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT017",
        name="National Land Info – Libraries",
        description="Public library facilities.",
        required=("response_format", "z", "x", "y"),
        optional=("administrativeAreaCode",),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT018",
        name="National Land Info – Municipal Offices & Community Centers",
        description="Municipal offices and community facilities.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT019",
        name="National Land Info – Natural Park Areas",
        description="National/natural park zoning.",
        required=("response_format", "z", "x", "y"),
        optional=("prefectureCode", "districtCode"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT020",
        name="National Land Info – Large-Scale Landfill Risk",
        description="Large-scale fill / excavation hazard map.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT021",
        name="National Land Info – Landslide Prevention Districts",
        description="Landslide prevention districts (polygons).",
        required=("response_format", "z", "x", "y"),
        optional=("prefectureCode", "administrativeAreaCode"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT022",
        name="National Land Info – Steep Slope Collapse Hazard Areas",
        description="Steep slope collapse hazard polygons.",
        required=("response_format", "z", "x", "y"),
        optional=("prefectureCode", "administrativeAreaCode"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT023",
        name="City Planning GIS – District Plans",
        description="Detailed district plan areas.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT024",
        name="City Planning GIS – High-Use Districts",
        description="High-use districts (高度利用地区).",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
    EndpointSpec(
        code="XKT025",
        name="Liquefaction Propensity Map",
        description="Liquefaction risk map by the MLIT Urban Bureau.",
        required=("response_format", "z", "x", "y"),
        defaults=dict(_TILE_DEFAULTS),
    ),
)

ENDPOINT_SPECS: dict[str, EndpointSpec] = {spec.code: spec for spec in _SPEC_LIST}
ORDERED_SPECS: tuple[EndpointSpec, ...] = _SPEC_LIST


_WRAPPER_DEFINITIONS: dict[str, tuple[str, str]] = {
    "XIT001": ("fetch_transaction_prices", "Real estate transaction prices (XIT001)."),
    "XIT002": ("fetch_municipalities", "Prefecture municipality directory (XIT002)."),
    "XCT001": ("fetch_appraisal_reports", "Appraisal report information (XCT001)."),
    "XPT001": ("fetch_transaction_points", "Transaction price points (XPT001)."),
    "XPT002": ("fetch_land_price_points", "Land price survey points (XPT002)."),
    "XKT001": ("fetch_city_planning_areas", "City planning areas / classifications (XKT001)."),
    "XKT002": ("fetch_zoning_layers", "City planning zoning (XKT002)."),
    "XKT003": ("fetch_location_optimization_layers", "Location optimization plan layers (XKT003)."),
    "XKT004": ("fetch_elementary_school_districts", "Elementary school districts (XKT004)."),
    "XKT005": ("fetch_junior_high_school_districts", "Junior high school districts (XKT005)."),
    "XKT006": ("fetch_school_facilities", "School facilities (XKT006)."),
    "XKT007": ("fetch_childcare_facilities", "Nursery / kindergarten facilities (XKT007)."),
    "XKT010": ("fetch_medical_facilities", "Medical institutions (XKT010)."),
    "XKT011": ("fetch_welfare_facilities", "Welfare facilities (XKT011)."),
    "XKT013": ("fetch_future_population_mesh", "Future population 250m mesh (XKT013)."),
    "XKT014": ("fetch_fire_zones", "Fire and quasi fire zones (XKT014)."),
    "XKT015": ("fetch_station_passenger_volume", "Station passenger volumes (XKT015)."),
    "XKT016": ("fetch_disaster_hazard_areas", "Disaster hazard areas (XKT016)."),
    "XKT017": ("fetch_library_facilities", "Libraries (XKT017)."),
    "XKT018": ("fetch_municipal_offices", "Municipal offices & community facilities (XKT018)."),
    "XKT019": ("fetch_natural_parks", "Natural park areas (XKT019)."),
    "XKT020": ("fetch_landfill_risk", "Large-scale landfill risk (XKT020)."),
    "XKT021": ("fetch_landslide_prevention_districts", "Landslide prevention districts (XKT021)."),
    "XKT022": ("fetch_steep_slope_hazards", "Steep slope collapse hazard areas (XKT022)."),
    "XKT023": ("fetch_district_plans", "District plan polygons (XKT023)."),
    "XKT024": ("fetch_high_use_districts", "High-use districts (XKT024)."),
    "XKT025": ("fetch_liquefaction_map", "Liquefaction propensity map (XKT025)."),
}


__all__ = [
    "EndpointSpec",
    "ENDPOINT_SPECS",
    "ORDERED_SPECS",
    "call_endpoint",
    "call_mlit_api",
    "get_default_params",
    "list_endpoints",
    "resolve_api_key",
]


def resolve_api_key(*, env_var: str = "MLIT_API_KEY", env_path: str | None = ".env") -> str:
    """Return the MLIT API key from the environment or a .env file."""

    global _API_KEY_CACHE
    if _API_KEY_CACHE:
        return _API_KEY_CACHE

    key = os.getenv(env_var)
    if not key and env_path:
        load_dotenv(env_path)
        key = os.getenv(env_var)

    if not key:
        raise RuntimeError(
            f"Missing API key: set the environment variable '{env_var}' or add it to {env_path}."
        )

    _API_KEY_CACHE = key
    return key


def _stringify(params: Mapping[str, Any]) -> Dict[str, str]:
    """Convert query parameters to strings and drop None values."""

    cleaned: Dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = str(value).lower()
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            cleaned[key] = ",".join(str(item) for item in value)
        else:
            cleaned[key] = str(value)
    return cleaned


def call_mlit_api(endpoint: str, params: Mapping[str, Any], *, timeout: int = 30) -> Any:
    """Issue a GET request against the MLIT API and return the decoded payload."""

    headers = {"Ocp-Apim-Subscription-Key": resolve_api_key()}
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=_stringify(params),
        headers=headers,
        timeout=timeout,
    )

    if response.status_code in (204, 404):
        return []

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type.startswith("application/json") or response.text.strip().startswith(("[", "{")):
        return response.json()
    return response.text


def _prepare_params(spec: EndpointSpec, overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    params = spec.with_defaults(overrides)
    missing = [
        name
        for name in spec.required
        if name not in params or params[name] is None or (isinstance(params[name], str) and params[name] == "")
    ]
    if missing:
        raise ValueError(f"Missing required parameters for {spec.code}: {', '.join(missing)}")
    return params


def call_endpoint(code: str, /, **params: Any) -> Any:
    """Validate parameters and call the requested MLIT endpoint."""

    try:
        spec = ENDPOINT_SPECS[code]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown endpoint code '{code}'. Known codes: {sorted(ENDPOINT_SPECS)}") from exc

    payload = _prepare_params(spec, params)
    return call_mlit_api(code, payload)


def get_default_params(code: str) -> Dict[str, Any]:
    """Return a copy of the documented default parameters for *code*."""

    params = dict(ENDPOINT_SPECS[code].defaults)
    return params


def list_endpoints() -> tuple[EndpointSpec, ...]:
    """Return the ordered list of endpoint specifications."""

    return ORDERED_SPECS


for _code, ( _func_name, _summary) in _WRAPPER_DEFINITIONS.items():
    spec = ENDPOINT_SPECS[_code]

    def _factory(code: str, func_name: str, summary: str, spec: EndpointSpec):
        def wrapper(**params: Any) -> Any:
            return call_endpoint(code, **params)

        wrapper.__name__ = func_name
        required = ", ".join(spec.required) if spec.required else "None"
        optional = ", ".join(spec.optional) if spec.optional else "None"
        wrapper.__doc__ = (
            f"{summary}\n\n"
            f"Endpoint: {code} — {spec.name}.\n"
            f"Required parameters: {required}. Optional parameters: {optional}."
        )
        return wrapper

    globals()[_func_name] = _factory(_code, _func_name, _summary, spec)
    __all__.append(_func_name)

