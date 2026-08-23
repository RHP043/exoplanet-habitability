import pandas as pd
import pyvo as vo
import os

service = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP/sync')

results = service.search("SELECT pl_name FROM ps")

frame = pd.DataFrame(results)

valid = set(frame.iloc[:,0])

print(valid)

planets = os.listdir('.')
for planet in planets:
    if '(' not in planet:
        continue
    print(planet, end=' ')
    planet = planet[:planet.index('(')].replace('_',' ')
    print(f'\x1b[3{1+(planet in valid)}m->\x1b[0m', planet)