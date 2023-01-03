import math
import os
import re
import sys
from math import pi

import numpy as np
import pandas as pd
import pickle as pkl
from PIL import Image
from astropy.constants import M_earth, GM_sun, au, R_earth

IMG_SIZE = int(sys.argv[1])
name_splitter = re.compile(r'[ _]')
planets = {}

print('\x1b[33mLoading data...\x1b[0m')
for root, _, images in os.walk('images'):
    for image in images:
        path = f'{root}/{image}'
        img = Image.open(path).convert(mode='RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        planet = image.replace('_', ' ')[:image.index('.')]
        if planet not in planets:
            planets[planet] = []
        planets[planet].append(img)

############################################################################################################
# Remaking the DataFrame
############################################################################################################

frame = pd.read_csv('exoplanet_data.csv')

# Some filenames are incorrect due to Kaggle or the Exoplanet Archive
to_rename = {'PSR B125712': 'PSR B1257+12', 'TIC 172900988': 'TIC 172900988 Aa', '51 PEG': '51 Peg'}
renamer = re.compile(r".*(PSR B125712|TIC 172900988|PEG).*")

print('\x1b[33mMaking New Array\x1b[0m')
# Combine the planets into an array
planet_arrays = np.array([[]], dtype=object)
for planet in planets:
    hostname, letter = planet.rsplit(' ', maxsplit=1)
    match = renamer.match(hostname)
    if match:
        hostname = hostname.replace(match.group(0), to_rename[match.group(0)])
    pl_arr = np.array([[]], dtype=object)
    pl_type = '?'
    for label, content in frame.loc[(frame['Hostname'] == hostname) & (frame['Letter'] == letter)].iterrows():
        pl = np.array(
            [[content['Earth Radii'], content['Earth Masses'], content['Semi-Major Axis'], content['Orbital Period'],
              content['Eccentricity'], content['Star Radius'], content['Star Temperature'], content['Star Mass']]],
            dtype=float)
        if pl_type == '?':
            pl_type = content['Planet Type']
        pl_arr = np.concatenate((pl_arr, pl), axis=0) if pl_arr.size else pl
    if not pl_arr.size:
        print(f'\x1b[31m{planet}\x1b[0m')
        continue
    print(f'\x1b[32m{planet}\x1b[0m', end='\33[2K\r')
    #     print([[f for f in pl_arr[:, i] if f != float('nan')] for i in range(pl_arr.shape[1])])
    pl_mean = np.array([[np.mean([pl_arr[:, i][~np.isnan(pl_arr[:, i])]]) for i in range(pl_arr.shape[1])]])
    pl_final = np.concatenate((np.array([[planet]], dtype=object), pl_mean, np.array([[pl_type]], dtype=object)),
                              axis=1)
    planet_arrays = np.concatenate((planet_arrays, pl_final), axis=0) if planet_arrays.size else pl_final


############################################################################################################
# Modifying the DataFrame
############################################################################################################

print('\x1b[33mMaking New DataFrame\x1b[0m')

df = pd.DataFrame(planet_arrays)
df.index = df.index.map(str)
df.index = df[0]
del df[0]
df.columns =['Earth Radii', 'Earth Masses', 'Semi-Major Axis', 'Orbital Period', 'Eccentricity', 'Star Radius', 'Star Temperature', 'Star Mass', 'Planet Type']
df.index.name = 'Planet'
# df['Image Path'] = [planet_paths[planet] for planet in planets]
df = df.sort_index()
df['Eccentricity'] = df['Eccentricity'].fillna(0)
spd = 86400  # Seconds Per Day
dens = {'rocky': 5.1*(100**3)/1000, 'gassy': 0.9*(100**3)/1000, 'icy': 1.4*(100**3)/1000}
for label, content in df.iterrows():
    if np.isnan(content['Orbital Period']):
        df.at[label, 'Orbital Period'] = 2 * pi * math.sqrt((content['Semi-Major Axis'] * au.value)**3/(content['Star Mass'] * GM_sun.value)) / spd
    if np.isnan(content['Semi-Major Axis']):
        df.at[label, 'Semi-Major Axis'] = ((GM_sun.value * content['Star Mass'] * (content['Orbital Period'] * spd) ** 2)/(4 * pi ** 2)) ** (1/3) / au.value
    if np.isnan(content['Earth Radii']):
        df.at[label, 'Earth Radii'] = (0.75 * content['Earth Masses'] * M_earth.value /(dens[content['Planet Type']] * pi)) ** (1/3) / R_earth.value
    if np.isnan(content['Earth Masses']):
        df.at[label, 'Earth Masses'] = ((4/3) * pi * (content['Earth Radii'] * R_earth.value) ** 3 * (dens[content['Planet Type']])) / M_earth.value

df.at['TYC 8998-760-1 b', 'Star Radius'] = 1.01
df.at['kap And b', 'Star Radius'] = 2.29
df.at['PSR B125712 b', 'Star Temperature'] = 28_856
df.at['PSR B125712 b', 'Star Radius'] = 0.000015
for pl_type in df['Planet Type'].unique():
    df[f'Is {pl_type.capitalize()}'] = df['Planet Type'] == pl_type
df = df.drop('Star Radius', axis=1)
df = df.drop('Planet Type', axis=1)

df.to_csv('training_data.csv')

for planet, images in planets.items():
    for image in images:
        image.info = np.expand_dims(np.asarray(df.loc[planet], dtype=np.float32), axis=0)

pkl.dump(planets, open('planets.pkl', 'wb'))

image_labels = np.array([[label, image, image.info] for label, images in planets.items() for image in images], dtype=object)
labels = image_labels[:, 0]
images = image_labels[:, 1]
images_arrs = np.array(list(map(np.asarray, image_labels[:, 1])))
image_info = np.concatenate(image_labels[:, 2], axis=0)

np.save('training_images.npy', images_arrs)
np.save('training_labels.npy', labels)
np.save('training_info.npy', image_info)

print('\x1b[32mDone\x1b[0m')
