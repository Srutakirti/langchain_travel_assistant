# Travel Assistant AI

An AI-powered **Travel Assistant** built with LangChain, Gemini, and UV.  
The assistant fetches **weather forecasts** and **top attractions** for any destination, then synthesizes the results into a concise travel summary.  

---

## i. How the LLM is Used for Reasoning

The program uses a **tool-calling agent** powered by a Large Language Model (LLM).  

- The **LLM’s role** is not just to generate text, but also to decide when and how to call tools (`get_weather` and `search_attractions`).  
- The **reasoning step** happens inside the agent’s “scratchpad”:  
  - The LLM is given a **system prompt** that clearly tells it the rules:  
    1. Always call `get_weather` for the forecast.  
    2. Always call `search_attractions` for top attractions.  
    3. Then synthesize the outputs into a single helpful answer.  
  - The **agent_scratchpad** allows the LLM to “think aloud” and plan its next step. For example:  
    - Step 1: “User asked about Paris, I need weather → call `get_weather`.”  
    - Step 2: “I also need attractions → call `search_attractions`.”  
    - Step 3: “Now I combine weather + attractions into a concise travel summary.”  
- In short:  
  - **Tools fetch real data** (weather + search results).  
  - **The LLM reasons over this data** and transforms it into natural language guidance (like clothing advice, indoor/outdoor activity suggestions, and concise attraction summaries).  

This reasoning loop ensures answers are both **grounded in factual data** (from APIs) and **human-friendly** (summarized by the LLM).  

---

## ii. Code and Flow of the Program

The code implements a **travel assistant agent** that can be run from the command line.  

### Key Components

1. **Weather Tool (`get_weather`)**  
   - Uses [WeatherAPI](https://www.weatherapi.com/) to fetch:  
     - Current conditions (temperature, humidity, etc.).  
     - A **7-day forecast**.  
   - Returns structured JSON for the LLM to interpret.  

2. **Attractions Tool (`search_attractions`)**  
   - Uses the DuckDuckGo search integration from `langchain_community`.  
   - Searches for `"top attractions in {city}"`.  
   - Returns multiple results with titles, snippets, and links.  

3. **Agent Setup**  
   - The **system prompt** defines strict behavior (always fetch weather + attractions, then summarize).  
   - `ChatPromptTemplate` wires:  
     - **System message** (rules).  
     - **Human input** (destination).  
     - **Agent scratchpad** (LLM’s reasoning workspace).  
   - `create_tool_calling_agent` + `AgentExecutor` build the agent loop.  

4. **LLM (Gemini)**  
   - Default model: `gemini-1.5-flash` via `ChatGoogleGenerativeAI`.  
   - Free, lightweight, and fast.  

5. **Main Program Flow**  
   ```text
   User runs → program reads destination
          ↓
   Run `run_query(destination)`
          ↓
   Agent executes:
      1) Calls `get_weather(destination)`
      2) Calls `search_attractions(destination)`
      3) Synthesizes both results
          ↓
   Final travel summary returned
   ```  

6. **Output**  
   - Displays the **assistant’s summary** in the terminal.  
   - Saves the **raw JSON agent output** to a file for debugging (e.g., `travel_assistant_output_Paris.json`).  

---

✅ In summary:  
- **Tools = data retrieval** (weather, attractions).  
- **LLM = reasoning + synthesis**.  
- **Agent = orchestration loop** between tools and reasoning.  

---

## iii. Running the Project with UV

This project uses [UV](https://docs.astral.sh/uv/) for dependency and environment management.  

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies
```bash
uv sync
```

This will install all dependencies defined in `pyproject.toml` and `uv.lock`.  

### 3. Set up API keys
Create a `.env` file in the project root:  
```bash
OPENAI_API_KEY=your_gemini_api_key_here
WEATHER_API_KEY=your_weatherapi_key_here
```

### 4. Run the program
```bash
uv run travel_assistant_agent.py
```

You’ll be prompted to enter a destination (e.g., **Paris**), and the program will fetch the **weather forecast** and **top attractions**, then display a friendly travel guide.

---

## iv. Example Output

```text
Travel Summary for Paris 🇫🇷

Weather: Mostly sunny, 23–28°C, low chance of rain.  
Great for outdoor walking and evening dining.

Top Attractions:
1. Eiffel Tower – Iconic landmark, best at sunset.  
2. Louvre Museum – World’s largest art museum.  
3. Notre-Dame Cathedral – Gothic architecture.  
4. Montmartre – Historic artists’ district.  
5. Seine River Cruise – Scenic boat rides.  

Tip: Carry a light jacket for evenings.
```

---

✨ That’s it! With **UV + Gemini + LangChain**, you now have a fully functional **AI Travel Assistant**.