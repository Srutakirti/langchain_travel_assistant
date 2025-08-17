#!/usr/bin/env python3
"""
Travel Assistant AI — LangChain Tool-Calling Agent

Features
- Custom Weather tool (WeatherAPI.com)
- DuckDuckGo search tool for top attractions
- Tool-calling agent (create_tool_calling_agent + AgentExecutor)

Setup
1) Install deps (recommended new virtualenv):
   pip install -U langchain langchain-community langchain-openai duckduckgo-search requests python-dotenv

2) Env vars:
   export GEMINI_API_KEY="sk-..."
   export WEATHER_API_KEY="your_weatherapi_key"

(Optionally load from .env alongside this file.)

Run:
   python travel_assistant_agent.py "Paris"
   # or interactive mode (no arg)
"""

import json
import os
import sys
from typing import Optional

import requests

# LangChain core/agents
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI


# DuckDuckGo search tool (no API key required)
from langchain_community.tools import DuckDuckGoSearchResults

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# -----------------------------
# Custom Weather Tool
# -----------------------------
@tool("get_weather", return_direct=False)
def get_weather(destination: str) -> str:
    """
    Fetch the current weather (and multi-day forecast) for a destination.
    Uses WeatherAPI.com (requires WEATHER_API_KEY env variable).

    Args:
        destination: e.g. "Paris", "San Francisco, USA", or "48.8566,2.3522"
    Returns:
        A concise JSON string with key weather fields suitable for LLM consumption.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "Missing WEATHER_API_KEY. Set the environment variable to your WeatherAPI.com key."
        })

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": destination,
        "days": 7,   # ⬅️ extend up to 7 days (max 10)
        "aqi": "no",
        "alerts": "no",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        location = data.get("location", {})
        current = data.get("current", {})
        forecasts = data.get("forecast", {}).get("forecastday", [])

        summary = {
            "location": {
                "name": location.get("name"),
                "region": location.get("region"),
                "country": location.get("country"),
                "localtime": location.get("localtime"),
                "lat": location.get("lat"),
                "lon": location.get("lon"),
            },
            "current": {
                "condition": current.get("condition", {}).get("text"),
                "temp_c": current.get("temp_c"),
                "feelslike_c": current.get("feelslike_c"),
                "wind_kph": current.get("wind_kph"),
                "humidity": current.get("humidity"),
                "cloud": current.get("cloud"),
            },
            "forecast": []
        }

        for day in forecasts:
            summary["forecast"].append({
                "date": day.get("date"),
                "maxtemp_c": day.get("day", {}).get("maxtemp_c"),
                "mintemp_c": day.get("day", {}).get("mintemp_c"),
                "daily_chance_of_rain": day.get("day", {}).get("daily_chance_of_rain"),
                "condition": (day.get("day", {}).get("condition") or {}).get("text"),
            })

        return json.dumps(summary)

    except requests.HTTPError as e:
        return json.dumps({"error": f"HTTP error from WeatherAPI: {str(e)}", "details": resp.text if 'resp' in locals() else None})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# -----------------------------
# DuckDuckGo Attractions Tool
# -----------------------------
def build_ddg_tool(k: int = 8):
    """
    Returns a DuckDuckGo search tool configured to fetch several results.
    We'll prime the query so the agent gets 'top attractions' results.
    """
    ddg = DuckDuckGoSearchResults(
        name="search_attractions",
        description=(
            "Searches the web (DuckDuckGo) for top attractions in a city. "
            "Input should be the destination name, e.g., 'Paris' or 'Tokyo'. "
            "Returns up to k results with title, snippet, and link."
        ),
        num_results=k,
    )
    return ddg


# -----------------------------
# Agent wiring
# -----------------------------
SYSTEM_PROMPT = (
     """
You are an intelligent Travel Assistant.
Given a user's destination, ALWAYS do the following:
1) Call get_weather to fetch the current weather and today's forecast.
2) Call search_attractions to fetch top attractions for the city.

Then synthesize a friendly, concise answer that takes into account the weahter and attractions.
Use the weather data to suggest appropriate clothing or activities.
If the weather is bad, suggest indoor attractions.Give at least 3-5 top attractions with brief notes.

Keep it practical and avoid redundant chatter.
If a tool returns an error, explain it briefly and continue with what you have.
"""
)


def build_agent(llm, k_results: int = 8) -> AgentExecutor:
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # free + fast
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    ddg_tool = build_ddg_tool(k=k_results)
    tools = [get_weather, ddg_tool]

    # Include agent_scratchpad so the agent can "think"
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),   
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def run_query(destination: str, llm = None) -> dict:
    """
    Invoke the agent on a destination string.
    Returns LangChain's standardized response dict.
    """
    executor = build_agent(llm=llm)
    # Nudge the agent so it knows the user's intent clearly.
    user_input = (
        f"Destination: {destination}. "
        f"Please provide the weather and a list of top attractions with brief notes."
    )
    return executor.invoke({"input": user_input})


def main():
    if len(sys.argv) > 1:
        destination = " ".join(sys.argv[1:]).strip()
    else:
        destination = input("Enter a destination city (e.g., 'Paris'): ").strip()

    if not destination:
        print("No destination provided.")
        sys.exit(1)

    # Minimal check for gemini key
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set. Set it to use the gemini model.\n"
              "export GEMINI_API_KEY='sk-...'\n")

    result = run_query(destination)
    print("\n=== Travel Assistant Result ===\n")
    # 'output' key contains the final string from the agent
    print(result.get("output", result))

    # Optional: save raw JSON to a file for debugging/turn-in
    out_path = f"travel_assistant_output_{destination.replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved raw agent output to: {out_path}")


if __name__ == "__main__":
    main()
