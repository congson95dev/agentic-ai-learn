"""Local MCP server that exposes OpenWeather tools over stdio."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# The MCP process is separate from the FastAPI process, so load the repository
# configuration here as well instead of relying on inherited environment state.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

mcp = FastMCP(
    "Local Weather",
    instructions="Use these tools for current conditions and OpenWeather's five-day forecast.",
)


def _find_location(location: str, api_key: str) -> dict | None:
    response = requests.get(
        "https://api.openweathermap.org/geo/1.0/direct",
        params={"q": location, "limit": 1, "appid": api_key},
        timeout=15,
    )
    response.raise_for_status()
    places = response.json()
    return places[0] if places else None


def _weather_api_key() -> str | None:
    return os.getenv("OPENWEATHER_API_KEY")


@mcp.tool()
def get_current_weather(location: str) -> str:
    """Get current weather for a city or location."""
    api_key = _weather_api_key()
    if not api_key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY."

    try:
        place = _find_location(location, api_key)
        if not place:
            return f"Could not find the location: {location}"
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": place["lat"], "lon": place["lon"], "appid": api_key, "units": "metric"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return (
            f"Current weather in {place['name']}: {data['weather'][0]['description'].title()}, "
            f"{data['main']['temp']}°C (feels like {data['main']['feels_like']}°C), "
            f"humidity {data['main']['humidity']}%."
        )
    except (requests.RequestException, KeyError, IndexError, TypeError) as error:
        return f"Could not retrieve weather: {error}"


@mcp.tool()
def get_weather_forecast(location: str) -> str:
    """Get OpenWeather's five-day forecast in three-hour intervals for a city or location."""
    api_key = _weather_api_key()
    if not api_key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY."

    try:
        place = _find_location(location, api_key)
        if not place:
            return f"Could not find the location: {location}"
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": place["lat"], "lon": place["lon"], "appid": api_key, "units": "metric"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        entries = data["list"]
        city_timezone = timezone(timedelta(seconds=data.get("city", {}).get("timezone", 0)))
        forecast = []
        for entry in entries:
            timestamp = datetime.fromtimestamp(entry["dt"], tz=city_timezone).strftime("%a %Y-%m-%d %H:%M")
            forecast.append(
                f"{timestamp}: {entry['weather'][0]['description'].title()}, "
                f"{entry['main']['temp']}°C (feels like {entry['main']['feels_like']}°C), "
                f"humidity {entry['main']['humidity']}%"
            )
        return f"Five-day forecast for {place['name']} (local time):\n" + "\n".join(forecast)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        return f"Could not retrieve weather forecast: {error}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
