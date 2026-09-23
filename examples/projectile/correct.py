"""Projectile trajectory with SI gravity and a declared ground constraint."""
import json
from math import sqrt

projectile = True
g = 9.81
speed = 20.0
angle_sin = 0.6
vx = speed * sqrt(1 - angle_sin**2)
vy = speed * angle_sin
flight = 2 * vy / g
range_m = vx * flight
expected = 2 * speed**2 * angle_sin * sqrt(1 - angle_sin**2) / 9.81
print('EPSILONMU_JSON:' + json.dumps({'states': [0, range_m], 'constraints': [
    {'name': 'SI projectile range', 'value': range_m / expected, 'minimum': 0.99, 'maximum': 1.01}]}))
