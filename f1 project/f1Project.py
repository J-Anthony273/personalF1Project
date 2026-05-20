#General Imports
import io
import numpy as np
import pandas as pd
from datetime import timedelta

#Website Imports
import fastf1
import fastf1.plotting
import plotly.graph_objects as go                                              
from flask import Flask, send_file, jsonify, render_template, request, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField                                     
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap

#Personal Data imports
from data.raceMapping import raceMapping
from data.teammateMapping import teammateMapping
from data.seasonMapping import seasonMapping
from data.scoringSystems import scoringSystems, sprintScoringSystems
from data.scoringMapping import scoringMapping, sprintScoringMapping

#Races where non full points should be used.
HALF_POINTS_RACES = {
    (2021, 12),
}

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

def format_time(seconds):                                   
    if seconds is None:                             
        return "N/A"                                
    minutes = int(seconds // 60)                    
    secs = seconds % 60                             
    return f"{minutes}:{secs:06.3f}"                


def positionChanges(season, race):
    try:
        if race.endswith("S"):
            race = race[:-1]
            session = fastf1.get_session(int(season), int(race), "S")
        else:
            session = fastf1.get_session(int(season), int(race), "R")
        session.load(telemetry=False, weather=False)

        fig = go.Figure()                                                     

        for drv in session.drivers:
            drv_laps = session.laps.pick_drivers(drv)
            if not drv_laps.empty:
                abb = drv_laps['Driver'].iloc[0]
                try:                                                          
                    style = fastf1.plotting.get_driver_style(
                        identifier=abb, style=['color', 'linestyle'], session=session
                    )
                    color = style.get('color', '#888888')                   
                except Exception:                                             
                    color = '#888888'                                         

                laps      = drv_laps['LapNumber'].tolist()                    
                positions = drv_laps['Position'].tolist()                     

                fig.add_trace(go.Scatter(                                     
                    x=laps,                                                   
                    y=positions,                                              
                    mode='lines+markers',                                   
                    name=abb,                                                 
                    line=dict(color=color, width=2),                          
                    marker=dict(color=color, size=7, symbol='circle',         
                                line=dict(color='white', width=1)),         
                    hovertemplate=(                                           
                        f'<b>{abb}</b><br>'                                   
                        'Lap %{x}<br>'                                        
                        'Position: %{y}'                                      
                        '<extra></extra>'                                    
                    )                                                         
                ))                                                            

        fig.update_layout(                                                   
            template='plotly_dark',                                           
            hovermode='x unified',                                           
            xaxis_title='Lap',                                                
            yaxis=dict(                                                       
                title='Position',                                             
                autorange='reversed',                                         
                tickvals=[1, 5, 10, 15, 20],                                  
            ),                                                                
            legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0)',                
                        font=dict(size=11)),                                  
            hoverlabel=dict(namelength=-1),                                    
            margin=dict(r=120),                                               
            height=800,                                                        
            autosize=True,                                                     
        )                                                                     

        return fig.to_json()                                                 

    except Exception as e:
        print(f"Error in positionChanges: {e}")
        return _error_fig(str(e))                                           


def qualifyingTimings(season, race):
    try:
        if race.endswith("S"):
            newRace = race[:-1]
            session = fastf1.get_session(int(season), int(newRace), "SQ")
        else:
            session = fastf1.get_session(int(season), int(race), "Q")
        session.load(telemetry=False, weather=False)
        results = session.results

        fig = go.Figure()                                                     
        all_seconds = []                                                      

        for _, row in results.iterrows():
            abb = row['Abbreviation']
            q1, q2, q3 = row['Q1'], row['Q2'], row['Q3']
            try:                                                              
                style = fastf1.plotting.get_driver_style(
                    identifier=abb, style=['color', 'linestyle'], session=session
                )
                color = style.get('color', '#888888')                        
            except Exception:                                                
                color = '#888888'                                            

            x_labels, y_vals, hover_texts = [], [], []                        
            for label, val in [("Q1", q1), ("Q2", q2), ("Q3", q3)]:         
                secs = to_seconds(val) if not pd.isna(val) else None          
                if secs is not None:                                          
                    x_labels.append(label)                                    
                    y_vals.append(secs)                                       
                    hover_texts.append(format_time(secs))                    
                    all_seconds.append(secs)                                  

            if not y_vals:                                                    
                continue                                                      

            fig.add_trace(go.Scatter(                                    
                x=x_labels,                                                   
                y=y_vals,                                                     
                mode='lines+markers',                                 
                name=abb,                                                     
                line=dict(color=color, width=2),                              
                marker=dict(color=color, size=9, symbol='circle',          
                            line=dict(color='white', width=1)),               
                customdata=hover_texts,                                     
                hovertemplate=(                                               
                    f'<b>{abb}</b><br>'                                       
                    '%{x}: %{customdata}'                                   
                    '<extra></extra>'                                         
                )                                                             
            ))                                                                

        # Build readable y-axis tick labels (M:SS.mmm)                       
        if all_seconds:                                                       
            lo, hi = min(all_seconds) - 0.5, max(all_seconds) + 0.5          
            tick_step = 0.5                                                   
            tick_vals = list(np.arange(                                       
                np.floor(lo / tick_step) * tick_step,                         
                np.ceil(hi  / tick_step) * tick_step + tick_step,            
                tick_step                                                     
            ))                                                                
            tick_texts = [format_time(t) for t in tick_vals]                 
        else:                                                                 
            tick_vals, tick_texts = [], []                                    

        fig.update_layout(                                            
            template='plotly_dark',                                           
            hovermode='x unified',                                             
            xaxis=dict(title='Session', categoryorder='array',                
                       categoryarray=['Q1', 'Q2', 'Q3']),                    
            yaxis=dict(                                                       
                title='Lap Time',                                             
                tickvals=tick_vals,                                         
                ticktext=tick_texts,                                          
                range=[min(all_seconds, default=0) - 0.5,                    
                       max(all_seconds, default=1) + 0.5] if all_seconds else None, 
            ),                                                                
            legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0)',                
                        font=dict(size=11)),                                  
            hoverlabel=dict(namelength=-1),                                   
            margin=dict(r=120),                                               
            height=750,                                                       
            autosize=True,                                                    
        )                                                                     

        return fig.to_json()                                                  

    except Exception as e:
        print(f"Error in qualifyingTimings: {e}")
        return _error_fig(str(e))                                             


def qualifyingAverages(season, pairing):
    try:
        noOfRaces = {2018: 21, 2019: 21, 2020: 17, 2021: 22,
                     2022: 19, 2023: 22, 2024: 24, 2025: 24}
        if " - " not in pairing:
            raise ValueError(f"Invalid pairing format: '{pairing}'")
        driverA, driverB = pairing.split(" - ")
        differences, counter = [], 0
        xAxis, yAxis = [], []

        if season == "2025" and driverB == "Andrea Kimi Antonelli":
            driverB = "Kimi Antonelli"

        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            if event["EventFormat"] == "sprint":
                sprintSession = fastf1.get_session(int(season), int(raceNo), "SQ")
                sprintSession.load(telemetry=False, weather=False)
                results = sprintSession.results
                qualiNames = results['FullName'].tolist()                  
                if driverA in qualiNames and driverB in qualiNames:
                    if _same_team(results, driverA, driverB):                
                        counter, differences = qualifyingAverageCalculator(
                            results, counter, differences, driverA, driverB)
                        xAxis.append(counter)
                        yAxis.append(sum(differences) / len(differences))

            session = fastf1.get_session(int(season), int(raceNo), "Q")
            session.load(telemetry=False, weather=False)
            results = session.results
            if season == "2025":
                results['FullName'] = results['FullName'].str.replace(
                'Andrea Kimi Antonelli', 'Kimi Antonelli', regex=False)
            qualiNames = results['FullName'].tolist()
            if driverA in qualiNames and driverB in qualiNames:
                if _same_team(results, driverA, driverB):                   
                    counter, differences = qualifyingAverageCalculator(
                        results, counter, differences, driverA, driverB)
                    xAxis.append(counter)
                    yAxis.append(sum(differences) / len(differences))

        bar_colors = ['#e63946' if v > 0 else '#457b9d' for v in yAxis]      

        fig = go.Figure()                                                  
        fig.add_trace(go.Bar(                                                  
            x=xAxis,                                                          
            y=yAxis,                                                          
            marker_color=bar_colors,    
            text=[f"{('+' if v > 0 else '')}{v:.3f}s" for v in yAxis], 
            textposition='outside',                                      
            hovertemplate=(                                                 
                'Race #%{x}<br>'                                              
                'Avg gap: %{y:.3f}s'                                          
                '<extra></extra>'                                             
            )                                                                 
        ))                                                                    

        fig.update_layout(                                                     
            template='plotly_dark',                                           
            title=f"Cumulative Qualifying Average Gap — {pairing} — {season}", 
            xaxis=dict(title='Number of races as teammates', dtick=1),        
            yaxis=dict(title='Average Qualifying Gap (s)', range=[-2, 2],     
                       dtick=0.25),                                           
            annotations=[dict(                                             
                x=0.5, y=-0.18, xref='paper', yref='paper',                  
                text='Negative = first driver faster; Positive = second driver faster', 
                showarrow=False, font=dict(size=11)                           
            )],                                                               
            height=500,                                                       
            margin=dict(b=80),                                                
        )                                                                     

        return fig.to_json()                                                  

    except Exception as e:
        print(f"Error in qualifyingAverages: {e}")
        return _error_fig(str(e))                                             


def singleQualifyingAverages(season, pairing):
    try:
        noOfRaces = {2018: 21, 2019: 21, 2020: 17, 2021: 22,
                     2022: 19, 2023: 22, 2024: 24, 2025: 24}
        if " - " not in pairing:
            raise ValueError(f"Invalid pairing format: '{pairing}'")
        driverA, driverB = pairing.split(" - ")
        xAxis, yAxis = [], []

        if season == "2025" and driverB == "Andrea Kimi Antonelli":
            driverB = "Kimi Antonelli"

        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            if event["EventFormat"] == "sprint":
                sprintSession = fastf1.get_session(int(season), int(raceNo), "SQ")
                sprintSession.load(telemetry=False, weather=False)
                results = sprintSession.results
                qualiNames = results['FullName'].tolist()
                if driverA in qualiNames and driverB in qualiNames:
                    if _same_team(results, driverA, driverB):                 
                        avg = singleQualifyingAverageCalculator(results, [], driverA, driverB)
                        xAxis.append(f"Sprint {raceNo}")
                        yAxis.append(avg)

            session = fastf1.get_session(int(season), int(raceNo), "Q")
            session.load(telemetry=False, weather=False)
            results = session.results
            if season == "2025":
                results['FullName'] = results['FullName'].str.replace(
                'Andrea Kimi Antonelli', 'Kimi Antonelli', regex=False)
            qualiNames = results['FullName'].tolist()
            if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli":
                driverB = "Kimi Antonelli"
            if driverA in qualiNames and driverB in qualiNames:
                if _same_team(results, driverA, driverB):                    
                    avg = singleQualifyingAverageCalculator(results, [], driverA, driverB)
                    xAxis.append(f"Race {raceNo}")
                    yAxis.append(avg)

        clipped     = [max(min(v, 2), -2) for v in yAxis]
        bar_colors  = ['#e63946' if v > 0 else '#457b9d' for v in yAxis]     
        hover_texts = [f"{('+' if v > 0 else '')}{v:.3f}s" for v in yAxis]   

        fig = go.Figure()                                                     
        fig.add_trace(go.Bar(                                              
            x=xAxis,                                                          
            y=clipped,                                                        
            marker_color=bar_colors,  
            text=hover_texts,
            textposition='outside',                                        
            customdata=hover_texts,                                            
            hovertemplate=(                                                   
                '%{x}<br>'                                                    
                'Gap: %{customdata}'                                         
                '<extra></extra>'                                             
            )                                                                 
        ))                                                                    

        fig.update_layout(                                                     
            template='plotly_dark',                                           
            title=f"Per-Session Qualifying Gap — {pairing} — {season}",      
            xaxis=dict(title='Race / Sprint', tickangle=-90),                 
            yaxis=dict(title='Qualifying Gap (s)', range=[-2, 2], dtick=0.25), 
            annotations=[dict(                                            
                x=0.5, y=-0.25, xref='paper', yref='paper',                  
                text='Negative = first driver faster; Positive = second driver faster', 
                showarrow=False, font=dict(size=11)                           
            )],                                                               
            height=520,                                                       
            margin=dict(b=120),                                               
        )                                                                     

        return fig.to_json()                                                  

    except Exception as e:
        print(f"Error in singleQualifyingAverages: {e}")
        return _error_fig(str(e))                                             


def championshipProgression(season, system, sprintsystem=None):
    try:
        noOfRaces = {2018: 21, 2019: 21, 2020: 17, 2021: 22,
                     2022: 19, 2023: 22, 2024: 24, 2025: 24}

        drivers            = {}
        driversResults     = {}
        driver_points_prog = {}
        driversRacesRaced  = {}
        driversPositions   = {}  
        reference_session  = None

        scoring = dict(scoringSystems)[int(system)]

        sprint_scoring = None
        if sprintsystem:
            sid = int(sprintsystem)
            if sid == 1:
                sprint_scoring = {"1": 3, "2": 2, "3": 1}
            elif sid == 2:
                sprint_scoring = {"1": 8, "2": 7, "3": 6, "4": 5,
                                  "5": 4, "6": 3, "7": 2, "8": 1}

        for raceNo in range(1, noOfRaces[int(season)] + 1):
            event = fastf1.get_event(int(season), int(raceNo))
            if event["EventFormat"] == "sprint" and sprint_scoring:
                s_type = 'SQ' if int(season) == 2021 else 'S'
                sp_sess = fastf1.get_session(int(season), raceNo, s_type)
                sp_sess.load()
                sp_results = sp_sess.results.loc[
                    :, ['Abbreviation', 'FirstName', 'LastName', 'ClassifiedPosition']
                ].copy()
                sp_results['DriverName'] = sp_results['FirstName'] + ' ' + sp_results['LastName']
                if int(season) == 2025:
                    sp_results['DriverName'] = sp_results['DriverName'].str.replace(
                        'Andrea Kimi Antonelli', 'Kimi Antonelli', regex=False)
                for _, row in sp_results.iterrows():
                    abb  = row['Abbreviation']
                    name = row['DriverName']
                    pos  = row['ClassifiedPosition']
                    if name not in drivers:
                        drivers[name] = 0
                        driversResults[name] = []
                        driver_points_prog[abb] = []
                        driversPositions[abb] = []  
                    pts = sprint_scoring.get(str(pos), 0)
                    driversResults[name].append(pts)
                    drivers[name] += pts
                    driversPositions[abb].append(str(pos))  

            session = fastf1.get_session(int(season), raceNo, 'R')
            session.load()
            reference_session = session

            results = session.results.loc[
                :, ['Abbreviation', 'FirstName', 'LastName', 'ClassifiedPosition']
            ].copy()
            results['DriverName'] = results['FirstName'] + ' ' + results['LastName']
            if int(season) == 2025:
                results['DriverName'] = results['DriverName'].str.replace(
                    'Andrea Kimi Antonelli', 'Kimi Antonelli', regex=False)

            winner_abv = session.results.loc[session.results['Position'] == 1, 'Abbreviation'].iloc[0]
            winner_laps = session.laps[session.laps['Driver'] == winner_abv]['LapNumber'].max()

            if winner_laps > 0:
                classification_cutoff = 0.9 * winner_laps
            else:
                classification_cutoff = 0

            lap_counts = session.laps.groupby('Driver').size()
            classified_abbs = lap_counts[lap_counts >= classification_cutoff].index.tolist()

            valid_laps = session.laps[                                         
                session.laps['Driver'].isin(classified_abbs) &                  
                session.laps['LapTime'].notna() &                               
                ~session.laps['Deleted']                                    
            ]                                                                  
            if not valid_laps.empty:                                           
                fastest_driver = valid_laps.loc[                               
                    valid_laps['LapTime'].idxmin(), 'Driver'                   
                ]                                                              
            else:                                                              
                fastest_driver = None   

            results['GotFastestLap'] = results['Abbreviation'] == fastest_driver  


            for _, row in results.iterrows():
                abb    = row['Abbreviation']
                name   = row['DriverName']
                pos    = row['ClassifiedPosition']
                got_fl = row['GotFastestLap']

                if name not in drivers:
                    drivers[name] = 0
                    driversResults[name] = []
                    driver_points_prog[abb] = []
                    driversPositions[abb] = []  

                driversPositions[abb].append(str(pos))  
                multiplier = 0.5 if (int(season), raceNo) in HALF_POINTS_RACES else 1.0   
                pts     = scoring.get(pos, 0) * multiplier                   
                fl_rule = scoring.get("FL", False)
                if fl_rule and (int(season), raceNo) not in HALF_POINTS_RACES:
                    if isinstance(fl_rule, bool) and fl_rule and got_fl:
                        pts += 1 * multiplier                                 
                    elif isinstance(fl_rule, int) and got_fl:
                        try:                                                   
                            if int(pos) <= int(fl_rule):                       
                                pts += 1 * multiplier                        
                        except (ValueError, TypeError):                     
                            pass                                               

                driversResults[name].append(pts)
                counted = scoring.get("Counted")
                if isinstance(counted, int):
                    driversResults[name].sort(reverse=True)
                    drivers[name] = sum(driversResults[name][:counted])
                else:
                    drivers[name] += pts

                driver_points_prog[abb].append(drivers[name])
                driversRacesRaced.setdefault(abb, []).append(raceNo)          

        final_pts = {abb: totals[-1] for abb, totals in driver_points_prog.items()}

        def countback_key(item):                                               
            abb, pts = item                                                   
            positions = driversPositions.get(abb, [])                         
            pos_counts = tuple(                                               
                -positions.count(str(p)) for p in range(1, 21)               
            )                                                                 
            return (-pts, *pos_counts)                                     

        sorted_drivers = sorted(final_pts.items(), key=countback_key)      

        fallback_colors = [                                                   
            '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4',      
            '#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff',      
            '#9A6324','#fffac8','#800000','#aaffc3','#808000','#ffd8b1',      
            '#000075','#a9a9a9'                                               
        ]                                                                     
        color_index = 0

        fig = go.Figure()                                                

        for abb, _ in sorted_drivers:
            totals = driver_points_prog[abb]
            races  = driversRacesRaced[abb]
            label  = f"{abb} ({totals[-1]} pts)"                     

            try:
                style = fastf1.plotting.get_driver_style(
                    identifier=abb, style=['color', 'linestyle'],
                    session=reference_session
                )
                color = style.get('color', fallback_colors[color_index % len(fallback_colors)]) 
            except Exception:
                color = fallback_colors[color_index % len(fallback_colors)]
                color_index += 1

            fig.add_trace(go.Scatter(                                      
                x=races,                                                      
                y=totals,                                                     
                mode='lines+markers',                                       
                name=label,                                                   
                line=dict(color=color, width=2),                              
                marker=dict(color=color, size=7, symbol='circle',             
                            line=dict(color='white', width=1)),               
                hovertemplate=(                                               
                    f'<b>{abb}</b><br>'                                       
                    'After race %{x}<br>'                                     
                    'Points: %{y}'                                            
                    '<extra></extra>'                                         
                )                                                             
            ))                                                                

        systems_dict = dict(scoringMapping)
        sprint_text  = ""
        if sprintsystem and sprint_scoring:
            sprint_dict = dict(sprintScoringMapping)
            sprint_text = f" + {sprint_dict[int(sprintsystem)]}"

        fig.update_layout(                                                   
            template='plotly_dark',                                           
            hovermode='x unified',                                            
            title=f"Championship Progression — {season} — {systems_dict[int(system)]}{sprint_text}", 
            xaxis=dict(title='Race Number', dtick=1),                         
            yaxis=dict(title='Championship Points'),                          
            legend=dict(                                                      
                x=0, y=-0.25,                                                  
                orientation='h',                                              
                bgcolor='rgba(0,0,0,0)',                                      
                font=dict(size=10),                                           
            ),                                                                
            hoverlabel=dict(namelength=-1),                                  
            margin=dict(b=200, t=60, l=60, r=40),                       
            height=1000,                                                 
            autosize=True,                                                    
        )                                                                     

        return fig.to_json()                                                  

    except Exception as e:
        print(f"Error in championshipProgression: {e}")
        return _error_fig(str(e))                                             



def _error_fig(msg):                                                           
    fig = go.Figure()                                                         
    fig.add_annotation(text=f"Error: {msg}", x=0.5, y=0.5,                   
                       xref='paper', yref='paper', showarrow=False,           
                       font=dict(size=14, color='red'))                       
    fig.update_layout(template='plotly_dark', height=400)                     
    return fig.to_json()                                                      

def _same_team(results, driverA, driverB):                                     
    teamA = results.loc[results['FullName'] == driverA, 'TeamName'].values[0] 
    teamB = results.loc[results['FullName'] == driverB, 'TeamName'].values[0] 
    return teamA == teamB                                                      

def _fix_antonelli(season, raceNo, driverB):                                  
    if season == "2025" and raceNo > 2 and driverB == "Andrea Kimi Antonelli": 
        return "Kimi Antonelli"                                                
    return driverB                                                             

def qualifyingAverageCalculator(results, counter, differences, driverA, driverB):
    counter += 1
    for col in ['Q1', 'Q2', 'Q3']:                                            
        aVal = results.loc[results['FullName'] == driverA, col].values[0]     
        bVal = results.loc[results['FullName'] == driverB, col].values[0]     
        if not pd.isna(aVal) and not pd.isna(bVal):                           
            diff = (aVal - bVal) / np.timedelta64(1, 's')                     
            differences.append(diff)                                           
    return counter, differences

def singleQualifyingAverageCalculator(results, differences, driverA, driverB):
    for col in ['Q1', 'Q2', 'Q3']:                                            
        aVal = results.loc[results['FullName'] == driverA, col].values[0]     
        bVal = results.loc[results['FullName'] == driverB, col].values[0]     
        if not pd.isna(aVal) and not pd.isna(bVal):                           
            diff = (aVal - bVal) / np.timedelta64(1, 's')                     
            differences.append(diff)                                           
    if not differences:                                                      
        return 0                                                              
    return sum(differences) / len(differences)

@app.route('/', methods=['GET', 'POST'])
def homePage():
    form = optionForm()
    plot_data_url = None                                             

    season_value = request.form.get('season') if request.method == "POST" else None
    graph_type   = request.form.get('graphType') if request.method == "POST" else None
    race_value   = request.form.get('races') if request.method == "POST" else None  

    if season_value:
        if graph_type in ('qualifyingAverages', 'singleQualifyingAverages'):
            race_choices = list(teammateMapping.get(season_value, []))
            race_choices.insert(0, ("", "--Driver Pairings--"))
        elif graph_type == "championshipProgressionGraph":
            race_choices = list(scoringMapping)
            race_choices.insert(0, ("", "--Scoring Systems--"))
        else:
            race_choices = list(raceMapping.get(season_value, []))
            race_choices.insert(0, ("", "--Race--"))
        form.races.choices = race_choices

    if request.method == "POST" and graph_type and season_value and race_value: 
        sprint_value  = request.form.get('sprintScoring') if graph_type == "championshipProgressionGraph" else None 
        plot_data_url = url_for("plot_data",  
                                graphType = graph_type,                                 
                                season=season_value,                          
                                race=race_value,                              
                                sprintScoring=sprint_value)                   

    return render_template("index.html", form=form, plot_data_url=plot_data_url)  


@app.route('/get_races', methods=['POST'])
def getRaces():
    seasonValue = request.json.get("season")
    raceChoices = list(raceMapping.get(seasonValue, []))
    raceChoices.insert(0, ("", "--Race--"))
    return jsonify(raceChoices)

@app.route('/get_pairings', methods=['POST'])
def getPairings():
    seasonValue    = request.json.get("season")
    pairingChoices = list(teammateMapping.get(seasonValue, []))
    pairingChoices.insert(0, ("", "--Driver Pairings--"))
    return jsonify(pairingChoices)

@app.route('/get_systems', methods=['POST'])
def getSystems():
    systemChoices = list(scoringMapping)
    systemChoices.insert(0, ("", "--Scoring Systems--"))
    return jsonify(systemChoices)


@app.route("/plot_data")                                                      
def plot_data():                                                              
    """Returns Plotly JSON so the browser can render an interactive chart.""" 
    graphType     = request.args.get("graphType")                             
    season        = request.args.get("season")                                
    race          = request.args.get("race")                                  
    sprintScoring = request.args.get("sprintScoring")                         
          
    if not graphType or not season or not race:                               
        missing = [k for k, v in                                              
                   {"graphType": graphType, "season": season, "race": race}.items() 
                   if not v]                                                  
        return jsonify({"error": f"Missing required parameters: {missing}"}), 400 

    dispatch = {                                                              
        "positionChanges":              lambda: positionChanges(season, race), 
        "qualifyingTimings":            lambda: qualifyingTimings(season, race), 
        "qualifyingAverages":           lambda: qualifyingAverages(season, race), 
        "singleQualifyingAverages":     lambda: singleQualifyingAverages(season, race), 
        "championshipProgressionGraph": lambda: championshipProgression(season, race, sprintScoring), 
    }                                                                 

    fn = dispatch.get(graphType)                                           
    if fn is None:                                                          
        return jsonify({"error": f"Unknown graph type: {graphType}"}), 400  

    plot_json = fn()                                                       
    if plot_json is None:                                                  
        return jsonify({"error": "Failed to generate chart"}), 500          

    return app.response_class(                                               
        response=plot_json,                                                   
        status=200,                                                           
        mimetype='application/json'                                           
    )                                                                         


if __name__ == '__main__':
    app.run(debug=True, port=5050)