
import fastf1
import matplotlib
import matplotlib.pyplot as plt
import fastf1.plotting
from flask import Flask, send_file, jsonify, render_template, request, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, HiddenField
from wtforms.validators import DataRequired, Length
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from raceMapping import raceMapping
import io
from datetime import timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from teammateMapping import teammateMapping
import numpy as np
import pandas as pd
from seasonMapping import seasonMapping
from scoringSystems import scoringSystems, sprintScoringSystems
from scoringMapping import scoringMapping, sprintScoringMapping
from collections import defaultdict


app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.sqlite3'
bootstrap = Bootstrap(app)

class optionForm(FlaskForm):
    graphType = SelectField("graphType", choices = [
        ("", "--Graph Type--"),
        ("positionChanges", "Position Changes Graph"),
        ("qualifyingTimings", "Qualifying Timings Graph"),
        ("qualifyingAverages", "Cumulative Qualifying Average Gap Graph"),
        ("singleQualifyingAverages", "Single Session Qualifying Average Gap Graph"),
        ("championshipProgressionGraph", "Championship Progression Graph")
    ], validators = [DataRequired()])

    season = SelectField("season", choices = [
        ("", "--Season--"),
        ("2018", "2018"),
        ("2019", "2019"),
        ("2020", "2020"),
        ("2021", "2021"),
        ("2022", "2022"),
        ("2023", "2023"),
        ("2024", "2024"),
        ("2025", "2025"),
    ], validators = [DataRequired()])

    races = SelectField("races", choices = [
        ("", "--Race--")
    ], validators = [DataRequired()])

    sprintScoring = SelectField("sprintScoring", choices = [
        ("", "--Sprint Scoring System--"),
        ("1", "Sprint Scoring System Used in 2021"),
        ("2", "Sprint Scoring System Used in 2022 - 2025")
    ])

    submit = SubmitField("Submit")

def to_seconds(t):
    if isinstance(t, timedelta):
        return t.total_seconds()
    elif isinstance(t, str):
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    elif isinstance(t, (float, int)):
        return t
    return None

def format_time(t, _):
    minutes = int(t // 60)
    seconds = t % 60
    return f"{minutes}:{seconds:06.3f}"

def positionChanges(season, race):
    try:
        matplotlib.use('Agg')
        
        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')

        if race.endswith("S"):
            race = race[:-1]
            session = fastf1.get_session(int(season), int(race), "S")
        else:
            session = fastf1.get_session(int(season), int(race), "R")
        session.load(telemetry=False, weather=False)

        fig, ax = plt.subplots(figsize=(12.0, 4.9))

        for drv in session.drivers:
            drv_laps = session.laps.pick_drivers(drv)
            if not drv_laps.empty:
                abb = drv_laps['Driver'].iloc[0]
                style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
                ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, **style)

        ax.set_ylim([20.5, 0.5])
        ax.set_yticks([1, 5, 10, 15, 20])
        ax.set_xlabel('Lap')
        ax.set_ylabel('Position')

        ax.legend(bbox_to_anchor=(1.0, 1.02))
        plt.tight_layout()

        print("Plot generated successfully")
        return fig
        
    except Exception as e:
        print(f"Error in positionChanges: {e}")
        fig, ax = plt.subplots(figsize=(8.0, 4.9))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Error generating plot')
        return fig
    
def qualifyingTimings(season, race):
    try:
        matplotlib.use('Agg')
        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')
        if race.endswith("S"):
            newRace = race[:-1]
            session = fastf1.get_session(int(season), int(newRace), "SQ")
        else:
            session = fastf1.get_session(int(season), int(race), "Q")
        session.load(telemetry=False, weather=False)
        results = session.results
        fig, ax = plt.subplots(figsize=(12.0, 4.9))
        all_times = []

        for _, row in results.iterrows():
            abb = row['Abbreviation']
            q1 = row['Q1']
            q2 = row['Q2']
            q3 = row['Q3']
            style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=session)
            if q3 != "NaT":
                y = [to_seconds(q1), to_seconds(q2), to_seconds(q3)]
            elif q2 != "NaT":
                y = [to_seconds(q1), to_seconds(q2)]
            elif q1 != "NaT":
                y = [to_seconds(q1)]
            else:
                y = []
            all_times.extend(y)
            x = ["Q1", "Q2", "Q3"]
            ax.plot(x, y, label = abb, **style, marker = "o")
        
        all_times = [t for t in all_times if t is not None]
        if all_times:
            min_y = min(all_times)
            max_y = max(all_times)
            padding = 0.5
            ax.set_ylim(min_y - padding, max_y + padding)
            ax.grid(True, axis='y')
            ax.yaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(mdates.DateFormatter('%M:%S.%f'))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
        
        ax.legend(bbox_to_anchor=(1.0, 1.02))
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in qualifyingTimings: {e}")
        fig, ax = plt.subplots(figsize=(8.0, 4.9))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Error generating plot')
        return fig


def qualifyingAverages(season, pairing):
    try:
        matplotlib.use("Agg")
        noOfRaces = {2018 : 21, 2019 : 21, 2020 : 17, 2021 : 22, 2022 : 19, 2023 : 22, 2024 : 24, 2025 : 13}
        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')
        fig, ax = plt.subplots(figsize=(12.0, 4.9))
        if " - " not in pairing:
            raise ValueError(f"Invalid pairing format: '{pairing}'. Expected format: 'Driver A - Driver B'")
        driverA, driverB = pairing.split(" - ")
        differences = []
        counter = 0
        xAxis= []
        yAxis = []

        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            print(event["EventFormat"])
            if event["EventFormat"] == "sprint_qualifying":
                sprintSession = fastf1.get_session(int(season), int(raceNo), "SQ")
                sprintSession.load(telemetry=False, weather=False)
                results = session.results
                qualiNames = results['FullName'].tolist()
                if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli":
                    driverB = "Kimi Antonelli"
                if driverA in qualiNames and driverB in qualiNames:
                    driverATeam = results.loc[results['FullName'] == driverA, 'TeamName'].values[0]
                    driverBTeam = results.loc[results['FullName'] == driverB, 'TeamName'].values[0]
                    if driverATeam == driverBTeam:
                        counter, differences = qualifyingAverageCalculator(results, counter, differences, driverA, driverB)
                        xAxis.append(counter)
                        yAxis.append(sum(differences) / len(differences))
            session = fastf1.get_session(int(season), int(raceNo), "Q")
            session.load(telemetry=False, weather=False)
            results = session.results
            qualiNames = results['FullName'].tolist()
            if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli":
                driverB = "Kimi Antonelli"
            if driverA in qualiNames and driverB in qualiNames:
                driverATeam = results.loc[results['FullName'] == driverA, 'TeamName'].values[0]
                driverBTeam = results.loc[results['FullName'] == driverB, 'TeamName'].values[0]
                if driverATeam == driverBTeam:
                    counter, differences = qualifyingAverageCalculator(results, counter, differences, driverA, driverB)
                    xAxis.append(counter)
                    yAxis.append(sum(differences) / len(differences))

        ax.bar(xAxis, yAxis)

        for i, v in enumerate(yAxis):
            sign = "+" if v > 0 else ""
            ax.text(xAxis[i], v + 0.05 if v >= 0 else v - 0.1, f"{sign}{v:.3f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=6)

        ax.set_yticks(np.arange(-2, 2.25, 0.25))
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_title(f"Cumulative Qualifying Average Gap for {pairing} in the {season} F1 season.")
        ax.set_xlabel("Number of Races as teamates.")
        ax.set_ylabel("Average Qualifying Gap (seconds).")
        ax.set_ylim(-2, 2)
        ax.set_xticks(range(1, counter + 1))
        description = ("If the bar is negative, the first driver is faster on average; if positive, the second driver is faster.")
        plt.figtext(0.5, -0.1, description, wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Error in qualifyingAverages: {e}")
        fig, ax = plt.subplots(figsize=(8.0, 4.9))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Error generating plot')
        return fig 

def championshipProgression(season, system, sprintsystem = None):
    try:
        matplotlib.use("Agg") 
        noOfRaces = {2018 : 21, 2019 : 21, 2020 : 17, 2021 : 22, 2022 : 19, 2023 : 22, 2024 : 24, 2025 : 14}
        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')
        fig, ax = plt.subplots(figsize=(16.0, 8.0))

        drivers = {}
        driversResults = {}
        driver_points_progression = {}
        driversRacesRaced = {}
        driversPositions = {}
        reference_session = None

        scoring = dict(scoringSystems)[int(system)]

        sprint_scoring = None
        if sprintsystem:
            sprint_system_id = int(sprintsystem)
            if sprint_system_id == 1:
                sprint_scoring = {"1": 3, "2": 2, "3": 1}
            elif sprint_system_id == 2:
                sprint_scoring = {"1": 8, "2": 7, "3": 6, "4": 5, "5": 4, "6": 3, "7": 2, "8": 1}


        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            if event["EventFormat"] == "sprint_qualifying" and sprint_scoring:
                if int(season) == 2021:
                    session = fastf1.get_session(int(season), raceNo, 'SQ')
                else:
                    session = fastf1.get_session(int(season), raceNo, 'S')
                session.load()
                results = session.results.loc[:, ['Abbreviation', 'FirstName', 'LastName', 'ClassifiedPosition']].copy()
                results['DriverName'] = results['FirstName'] + ' ' + results['LastName']
                for _, row in results.iterrows():
                    abb = row['Abbreviation']
                    driver_name = row['DriverName']
                    pos = row['ClassifiedPosition']

                    if driver_name not in drivers:
                        drivers[driver_name] = 0
                        driversResults[driver_name] = []
                        driver_points_progression[abb] = []
                        
                    points = sprint_scoring.get(str(pos), 0)
                    if driver_name not in driversResults:
                        driversResults[driver_name] = []
                    driversResults[driver_name].append(points)
                    drivers[driver_name] += points

                
            session = fastf1.get_session(int(season), raceNo, 'R')
            session.load()
            reference_session = session

            results = session.results.loc[:, ['Abbreviation', 'FirstName', 'LastName', 'ClassifiedPosition']].copy()
            results['DriverName'] = results['FirstName'] + ' ' + results['LastName']

            fastest_lap = session.laps.pick_fastest()
            fastest_driver = fastest_lap['Driver'] if fastest_lap is not None else None
            results['GotFastestLap'] = results['Abbreviation'] == fastest_driver
            
            for _, row in results.iterrows():
                abb = row['Abbreviation']
                driver_name = row['DriverName']
                pos = row['ClassifiedPosition']
                gotFL = row['GotFastestLap']

                if driver_name not in drivers:
                    drivers[driver_name] = 0
                    driversResults[driver_name] = []
                    driversPositions[driver_name] = {}
                    driver_points_progression[abb] = []

                if pos not in driversPositions[driver_name]:
                    driversPositions[driver_name][pos] = 1
                
                driversPositions[driver_name][pos] += 1
                points = scoring.get(pos, 0)
                flRule = scoring.get("FL", False)
                if flRule:
                    if isinstance(flRule, bool) and flRule:
                        if gotFL:
                            points += 1
                    elif isinstance(flRule, int):
                        if gotFL and int(pos) <= int(flRule):
                            points += 1
                            
                driversResults[driver_name].append(points)
                countedRaces = scoring.get("Counted")
                if isinstance(countedRaces, int):
                    driversResults[driver_name].sort(reverse=True)
                    drivers[driver_name] = sum(driversResults[driver_name][:countedRaces])
                else:
                    drivers[driver_name] += points
                    
                driver_points_progression[abb].append(drivers[driver_name])

                if abb in driversRacesRaced:
                    driversRacesRaced[abb].append(raceNo)
                else:
                    driversRacesRaced[abb] = []
                    driversRacesRaced[abb].append(raceNo)

        final_points_dict = {
            abb: totals[-1] for abb, totals in driver_points_progression.items()
        }
        sorted_drivers = sorted(final_points_dict.items(), key=lambda x: x[1], reverse=True)
        fallback_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        color_index = 0

        abb_to_name = {}
        for abb, totals in driver_points_progression.items():
            for name in drivers.keys():
                # Match by checking if this driver has data
                if name not in abb_to_name.values():
                    abb_to_name[abb] = name
                    break

        final_sorted_drivers = []
        i = 0
        while i < len(sorted_drivers):
            current_points = sorted_drivers[i][1]
            tied_drivers = [sorted_drivers[i]]

            j = i + 1
            while j < len(sorted_drivers) and sorted_drivers[j][1] == current_points:
                tied_drivers.append(sorted_drivers[j])
                j += 1

            if len(tied_drivers) > 1:
                def get_countback_key(driver_tuple):
                    abb = driver_tuple[0]
                    driver_name = abb_to_name.get(abb)
                    if driver_name and driver_name in driversPositions:
                        countback = []
                        for pos in range(1, 21):
                            count = driversPositions[driver_name].get(pos, 0)
                            countback.append(-count)
                        return tuple(countback)
                    return tuple([0] * 20)
                
                tied_drivers.sort(key=lambda x: (get_countback_key(x), x[0]))

            final_sorted_drivers.extend(tied_drivers)
            i = j

        sorted_drivers = final_sorted_drivers

        for abb, _ in sorted_drivers:
            totals = driver_points_progression[abb]
            races_participated = driversRacesRaced[abb]
            final_points = totals[-1]
            label = f"{abb}-{final_points}"

            try:
                style = fastf1.plotting.get_driver_style(
                    identifier=abb, 
                    style=['color', 'linestyle'], 
                    session=reference_session
                )
                ax.plot(races_participated, totals, label=label, **style)
            except Exception as style_error:
                print(f"Warning: Could not get style for driver {abb}: {style_error}")
                fallback_color = fallback_colors[color_index % len(fallback_colors)]
                ax.plot(races_participated, totals, label=label, color=fallback_color, linewidth=2)
                color_index += 1

        ax.set_xlabel('Race Number')
        ax.set_ylabel('Championship Points')
        
        ax.set_xticks(range(1, noOfRaces[int(season)] + 1))
        ax.grid(True, alpha=0.3)
        systems = dict(scoringMapping)
        sprint_text = ""
        if sprintsystem and sprint_scoring:
            sprint_systems = dict(sprintScoringMapping)
            sprint_text = f" with {sprint_systems[int(sprintsystem)]}"
        ax.set_title(f"Championship progression for the {season} season using the {systems[int(system)]}{sprint_text}.")

        handles, labels = ax.get_legend_handles_labels()
        handles_labels = list(zip(labels, handles))
        handles_labels.sort(key=lambda x: int(x[0].split('-')[-1]), reverse=True)
        labels, handles = zip(*handles_labels)
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig
               
    except Exception as e:
        print(f"Error in championshipProgression: {e}")
        fig, ax = plt.subplots(figsize=(8.0, 4.9))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Error generating plot')
        return fig

def singleQualifyingAverages(season, pairing):
    try:
        matplotlib.use("Agg")
        noOfRaces = {2018 : 21, 2019 : 21, 2020 : 17, 2021 : 22, 2022 : 19, 2023 : 22, 2024 : 24, 2025 : 14}
        fastf1.plotting.setup_mpl(mpl_timedelta_support=False, misc_mpl_mods=False, color_scheme='fastf1')
        fig, ax = plt.subplots(figsize=(12.0, 4.9))
        if " - " not in pairing:
            raise ValueError(f"Invalid pairing format: '{pairing}'. Expected format: 'Driver A - Driver B'")
        driverA, driverB = pairing.split(" - ")
        counter = 0
        differences = []
        xAxis= []
        yAxis = []

        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            if event["EventFormat"] == "sprint_qualifying":
                counter += 1
                sprintSession = fastf1.get_session(int(season), int(raceNo), "SQ")
                sprintSession.load(telemetry=False, weather=False)
                results = session.results
                qualiNames = results['FullName'].tolist()
                if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli":
                    driverB = "Kimi Antonelli"
                if driverA in qualiNames and driverB in qualiNames:
                    driverATeam = results.loc[results['FullName'] == driverA, 'TeamName'].values[0]
                    driverBTeam = results.loc[results['FullName'] == driverB, 'TeamName'].values[0]
                    if driverATeam == driverBTeam:
                        average = singleQualifyingAverageCalculator(results, differences, driverA, driverB)
                        differences = []
                        xAxis.append(f"Sprint Race {counter}")
                        yAxis.append(average)
            session = fastf1.get_session(int(season), int(raceNo), "Q")
            session.load(telemetry=False, weather=False)
            results = session.results
            qualiNames = results['FullName'].tolist()
            if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli":
                driverB = "Kimi Antonelli"
            if driverA in qualiNames and driverB in qualiNames:
                driverATeam = results.loc[results['FullName'] == driverA, 'TeamName'].values[0]
                driverBTeam = results.loc[results['FullName'] == driverB, 'TeamName'].values[0]
                if driverATeam == driverBTeam:
                    average = singleQualifyingAverageCalculator(results, differences, driverA, driverB)
                    differences = []
                    xAxis.append(f"Race {raceNo}")
                    yAxis.append(average)
        
        clippedYAxis = [max(min(v, 2), -2) for v in yAxis]
        ax.bar(xAxis, clippedYAxis)

        for i, (orig, clipped) in enumerate(zip(yAxis, clippedYAxis)):
            sign = "+" if orig > 0 else ""

            if orig > 2:
                ax.text(xAxis[i], 2 - 0.05, f"{sign}{orig:.3f}", ha='center', va='top', fontsize=6, fontweight='bold')
            elif orig < -2:
                ax.text(xAxis[i], -2 + 0.05, f"{sign}{orig:.3f}", ha='center', va='bottom', fontsize=6, fontweight='bold')
            else:
                ax.text(xAxis[i], orig + 0.05 if orig >= 0 else orig - 0.1, f"{sign}{orig:.3f}", ha='center', va='bottom' if orig >= 0 else 'top', fontsize=6,  fontweight='bold')

        ax.set_yticks(np.arange(-2, 2.25, 0.25))
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        ax.set_title(f"Qualifying Average Gap in each race for {pairing} in the {season} F1 season.")
        ax.set_xlabel("Race/Sprint Race Number ")
        ax.set_ylabel("Average Qualifying Gap (seconds).")
        ax.set_ylim(-2, 2)
        ax.set_xticklabels(xAxis, rotation=90, ha='right', fontsize=7)
        description = ("If the bar is negative, the first driver is faster on average; if positive, the second driver is faster.")
        plt.figtext(0.5, -0.1, description, wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in singleQualifyingAverages: {e}")
        fig, ax = plt.subplots(figsize=(8.0, 4.9))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Error generating plot')
        return fig   

    
def qualifyingAverageCalculator(results, counter, differences, driverA, driverB):
    counter += 1
    driverAQ1 = results.loc[results['FullName'] == driverA, 'Q1'].values[0]
    driverBQ1 = results.loc[results['FullName'] == driverB, 'Q1'].values[0]
    driverAQ2 = results.loc[results['FullName'] == driverA, 'Q2'].values[0]
    driverBQ2 = results.loc[results['FullName'] == driverB, 'Q2'].values[0]
    driverAQ3 = results.loc[results['FullName'] == driverA, 'Q3'].values[0]
    driverBQ3 = results.loc[results['FullName'] == driverB, 'Q3'].values[0]

    if not pd.isna(driverAQ1) and not pd.isna(driverBQ1):
        difference = (driverAQ1 - driverBQ1) / np.timedelta64(1, 's')
        differences.append(difference)
                
    if not pd.isna(driverAQ2) and not pd.isna(driverBQ2):
        difference = (driverAQ2 - driverBQ2) / np.timedelta64(1, 's')
        differences.append(difference)

    if not pd.isna(driverAQ3) and not pd.isna(driverBQ3):
        difference = (driverAQ3 - driverBQ3) / np.timedelta64(1, 's')
        differences.append(difference)

    return counter, differences

def singleQualifyingAverageCalculator(results, differences, driverA, driverB):
    driverAQ1 = results.loc[results['FullName'] == driverA, 'Q1'].values[0]
    driverBQ1 = results.loc[results['FullName'] == driverB, 'Q1'].values[0]
    driverAQ2 = results.loc[results['FullName'] == driverA, 'Q2'].values[0]
    driverBQ2 = results.loc[results['FullName'] == driverB, 'Q2'].values[0]
    driverAQ3 = results.loc[results['FullName'] == driverA, 'Q3'].values[0]
    driverBQ3 = results.loc[results['FullName'] == driverB, 'Q3'].values[0]

    if not pd.isna(driverAQ1) and not pd.isna(driverBQ1):
        difference = (driverAQ1 - driverBQ1) / np.timedelta64(1, 's')
        differences.append(difference)
                
    if not pd.isna(driverAQ2) and not pd.isna(driverBQ2):
        difference = (driverAQ2 - driverBQ2) / np.timedelta64(1, 's')
        differences.append(difference)

    if not pd.isna(driverAQ3) and not pd.isna(driverBQ3):
        difference = (driverAQ3 - driverBQ3) / np.timedelta64(1, 's')
        differences.append(difference)

    average = sum(differences)/ len(differences)
    return average





@app.route('/', methods = ['GET', 'POST'])
def homePage():
    form = optionForm()
    img_url = None

    print(f"Request method: {request.method}")
    print(f"Form data: {request.form}")

    season_value = request.form.get('season') if request.method == "POST" else None
    graph_type = request.form.get('graphType') if request.method == "POST" else None

    if season_value:
        if graph_type == 'qualifyingAverages' or graph_type == 'singleQualifyingAverages':
            race_choices = teammateMapping.get(season_value, [])
            race_choices.insert(0, ("", "--Driver Pairings--"))

        elif graph_type == "championshipProgressionGraph":
            race_choices = list(scoringMapping)
            race_choices.insert(0, ("", "--Scoring Systems--"))

        else:
            race_choices = raceMapping.get(season_value, [])
            race_choices.insert(0, ("", "--Race--"))

        form.races.choices = race_choices
        print(f"Form valid: {form.validate_on_submit()}")
    
    if form.errors:
        print(f"Form errors: {form.errors}")

    if request.method == "POST" and form.validate_on_submit():
        graphType = form.graphType.data
        season = form.season.data
        race = form.races.data
        sprintScoring = form.sprintScoring.data if graphType == "championshipProgressionGraph" else None

        img_url = url_for("plot_png", graphType=graphType, season=season, race=race, sprintScoring=sprintScoring)
    
    return render_template("index.html", form=form, img_url=img_url)

@app.route('/get_races', methods = ['POST'])
def getRaces():
    seasonValue = request.json.get("season")
    raceChoices = raceMapping.get(seasonValue, [])
    raceChoices.insert(0, ("", "--Race--"))
    return jsonify(raceChoices)

@app.route('/get_pairings', methods = ['POST'])
def getPairings():
    seasonValue = request.json.get("season")
    pairingChoices= teammateMapping.get(seasonValue, [])
    pairingChoices.insert(0, ("", "--Driver Pairings--"))
    return jsonify(pairingChoices)

@app.route('/get_systems', methods = ['POST'])
def getSystems():
    systemChoices = list(scoringMapping)
    systemChoices.insert(0, ("", "--Scoring Systems--"))
    return jsonify(systemChoices)

@app.route("/plot.png")
def plot_png():
    graphType = request.args.get("graphType")
    season = request.args.get("season")
    race = request.args.get("race")
    sprintScoring = request.args.get("sprintScoring")
    
    if graphType == "positionChanges":
        try:
            fig = positionChanges(season, race)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype="image/png")
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close('all')
            return f"Error generating plot: {e}", 500
        
    elif graphType == "qualifyingTimings":
        try:
            fig = qualifyingTimings(season, race)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype="image/png")
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close('all')
            return f"Error generating plot: {e}", 500
    
    elif graphType == "qualifyingAverages":
        try:
            fig = qualifyingAverages(season, race)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype="image/png")
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close('all')
            return f"Error generating plot: {e}", 500
        
    elif graphType == "singleQualifyingAverages":
        try:
            fig = singleQualifyingAverages(season, race)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype="image/png")
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close('all')
            return f"Error generating plot: {e}", 500
        
    elif graphType == "championshipProgressionGraph":
        try:
            fig = championshipProgression(season, race, sprintScoring)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype="image/png")
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            plt.close('all')
            return f"Error generating plot: {e}", 500
    

    
    return "Invalid graph type", 400

if __name__ == '__main__':
    app.run(debug=True, port = 5050)
