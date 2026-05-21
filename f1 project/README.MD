# F1 Plotter

A Flask web application for visualising Formula 1 data. Built with [FastF1](https://docs.fastf1.dev/) for data fetching and [Plotly](https://plotly.com/python/) for interactive charts, presented through a dark, F1-themed dashboard UI.

---

## Features

F1 Plotter supports five graph types:

- **Position Changes Graph** — tracks each driver's position lap-by-lap through a race or sprint race.
- **Qualifying Timings Graph** — plots every driver's Q1, Q2, and Q3 lap times for a qualifying or sprint qualifying session, with times displayed in `m:ss.mmm` format.
- **Cumulative Qualifying Average Gap Graph** — shows the running average qualifying time gap between a pair of teammates across an entire season.
- **Single Session Qualifying Average Gap Graph** — shows the qualifying time gap between a pair of teammates for a single race weekend.
- **Championship Progression Graph** — plots each driver's cumulative championship points race-by-race for a selected season, with support for any historical F1 scoring system and sprint scoring variant.

---

## Project Structure

```
f1Project.py              # Main Flask application — routes and graph logic
data/
    raceMapping.py        # Race names and numbers per season (2018–2025)
    teammateMapping.py    # Teammate pairings per season (2018–2025)
    seasonMapping.py      # Season ranges for dropdown menus
    scoringMapping.py     # Human-readable names for scoring systems
    scoringSystems.py     # Points awarded per position for every historical scoring system
templates/
    index.html            # Main page template (Jinja2 + Plotly.js)
static/
    css/
        style.css         # Dark carbon-fibre dashboard stylesheet
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install flask flask-wtf flask-bootstrap fastf1 plotly pandas numpy
```

### Run the app

```bash
python f1Project.py
```

The app runs on `http://localhost:5050` by default.

> **Note:** FastF1 caches session data locally on first load. Generating a chart for a session you haven't loaded before will take longer as data is downloaded from the F1 API.

---

## Usage

1. Open the app in your browser.
2. Select a **Graph Type** from the top bar.
3. Select a **Season** (2018–2025).
4. The third dropdown updates dynamically based on your selections:
   - Most graph types show a list of **races** for that season.
   - Qualifying average graphs show **teammate pairings**.
   - The Championship Progression graph shows **scoring systems**.
5. If you selected the Championship Progression graph, also choose a **Sprint Scoring System**.
6. Click **Submit** to generate the chart. Charts are interactive — you can hover, zoom, and toggle drivers on/off via the legend.

---

## Data Coverage

| Data | Coverage |
|---|---|
| Races & sessions | 2018–2025 (via FastF1) |
| Championship Progression | 1961–2025 (historical scoring systems included) |
| Teammate pairings | 2018–2025 (including mid-season changes) |
| Sprint races | 2021–2025 |

### Scoring systems

Twenty historical F1 scoring systems are included, from the 1950–1953 system through to the current 2025 system. Sprint scoring variants for 2021 and 2022–2025 are also supported.

Notable edge cases handled:

- **Half points races** — the 2021 Belgian Grand Prix is flagged and points are halved accordingly.
- **Fastest lap point** — correctly applied per the rules of each era, including the 2019–2024 top-10 restriction.
- **Results-counted scoring** — older systems where only the best N results counted are fully implemented.
- **Tie-breaking** — ties are resolved by countback (number of 1st places, then 2nd places, and so on).

---

## Notes

- The app uses FastF1's session caching. If you want to specify a cache directory, add `fastf1.Cache.enable_cache('path/to/cache')` near the top of `f1Project.py`.
- The secret key in `app.config['SECRET_KEY']` should be changed to a secure random value before any public deployment.
- Generating charts — particularly the Qualifying Averages and Championship Progression graphs — can take some time on first run as they load multiple sessions sequentially.