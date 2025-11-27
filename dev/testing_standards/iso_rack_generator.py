import json
import numpy as np


bottom_height = 30
top_height = 160
step = 4
start_number = 1


rack_heights = np.arange(bottom_height, top_height + step, step)

iso_rack = {}

for rack_number, rack_height in enumerate(rack_heights, start=start_number):

    iso_rack[rack_number] = int(rack_height)

with open('iso_rack.json', 'w') as f:
    json.dump(iso_rack, f, indent=4)
