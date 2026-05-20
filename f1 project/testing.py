import fastf1
 # Speeds up repeated runs
print(fastf1.__version__)
fastf1.Cache.clear_cache()

# Get the race session; for example, the 2023 Abu Dhabi Grand Prix (round 22)
session = fastf1.get_session(1951, 3, 'R')
session.load()
results = session.results.loc[:, ['FirstName', 'LastName', 'TeamName', 'ClassifiedPosition', 'Points']]

# Combine first and last name into a single column

# Select and reorder columns
results = results[['ClassifiedPosition', 'DriverName', 'TeamName', 'Points']]
