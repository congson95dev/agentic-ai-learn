from __future__ import annotations

import math
import os
from typing import Literal

import requests
from langchain_core.tools import tool
from langgraph.types import interrupt

from database import LongTermMemory, SessionLocal
from rag import retrieve

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulate buying stock. Always asks the user to approve before placing the simulated order."""
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {"status": "success", "message": f"Purchase order placed for {quantity} shares of {symbol}."}
    return {"status": "cancelled", "message": f"Purchase of {quantity} shares of {symbol} was declined."}


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression, for example `math.sqrt(16)` or `10 * 5`."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {"math": math, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}))
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> dict | str:
    """Get the current stock quote for a ticker such as AAPL or TSLA."""
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        return "Alphavantage API key is missing. Set ALPHAVANTAGE_API_KEY."
    try:
        response = requests.get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key}, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return f"Could not retrieve a stock quote: {error}"


@tool
def get_current_weather(location: str) -> str:
    """Get current weather for a city or location."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY."
    try:
        geo = requests.get("https://api.openweathermap.org/geo/1.0/direct", params={"q": location, "limit": 1, "appid": key}, timeout=15)
        geo.raise_for_status()
        places = geo.json()
        if not places:
            return f"Could not find the location: {location}"
        place = places[0]
        weather = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"lat": place["lat"], "lon": place["lon"], "appid": key, "units": "metric"}, timeout=15)
        weather.raise_for_status()
        data = weather.json()
        return f"Current weather in {place['name']}: {data['weather'][0]['description'].title()}, {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C), humidity {data['main']['humidity']}%."
    except (requests.RequestException, KeyError, IndexError) as error:
        return f"Could not retrieve weather: {error}"


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant passages from files uploaded by the user."""
    return retrieve(query)


@tool
def save_memory(memory: str, action: Literal["save", "retrieve"] = "save") -> str:
    """Save a user preference/fact for future chats. Use action='retrieve' to find remembered facts relevant to a query."""
    db = SessionLocal()
    try:
        if action == "save":
            if not db.query(LongTermMemory).filter(LongTermMemory.content == memory.strip()).first():
                db.add(LongTermMemory(content=memory.strip()))
                db.commit()
            return "Memory saved."
        records = db.query(LongTermMemory).filter(LongTermMemory.content.ilike(f"%{memory.strip()}%")).limit(8).all()
        return "\n".join(item.content for item in records) or "No relevant saved memory found."
    finally:
        db.close()
