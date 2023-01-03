import pandas as pd
import pyvo as vo
import time
# Initial Selection of Variables
variables = ['hostname', 'pl_letter', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_orbsmax', 'pl_orbper',
             'pl_orbeccen', 'st_spectype', 'st_rad', 'st_teff', 'st_mass']
service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP/sync')

print("\x1b[33mQuerying Exoplanet Archive...\x1b[0m")
start = time.perf_counter()
results = service.search(f"SELECT {', '.join(variables)} FROM ps")
end = time.perf_counter()
print(f"\x1b[32mQuery Complete in {end-start:.3f} seconds\x1b[0m")
# Data Conversion
frame = pd.DataFrame(results)
# Removed RV_AMP because of transit measurements (e.g. TRAPPIST-1)
frame = frame[variables]
frame.rename(
    columns={'hostname': 'Hostname', 'pl_letter': 'Letter', 'pl_rade': 'Earth Radii', 'pl_bmasse': 'Earth Masses',
             'pl_dens': 'Density (g/m^3)', 'pl_orbsmax': 'Semi-Major Axis', 'pl_orbper': 'Orbital Period',
             'pl_orbeccen': 'Eccentricity', 'st_spectype': 'Spectral Type', 'st_rad': 'Star Radius',
             'st_teff': 'Star Temperature', 'st_mass': 'Star Mass'}, inplace=True)

types = []
for label, content in frame.iterrows():
    rad = float(content['Earth Radii'])
    mass = float(content['Earth Masses'])
    pl_type = '?'
    if rad < 2.2 or mass < 5:
        pl_type = 'rocky'
    elif 2.2 <= rad < 6 or 5 < mass < 20:
        pl_type = 'icy'
    elif 6 <= rad or mass > 30:
        pl_type = 'gassy'
    types.append(pl_type)

frame['Planet Type'] = types

frame.to_csv('exoplanet_data.csv')
print('\x1b[32mData saved to "exoplanet_data.csv"\x1b[0m')