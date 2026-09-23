"""Injected defect: inverse-square magnitude implemented as inverse-cube."""
import json
from math import hypot

orbit = True
dt = 0.01
x, y, vx, vy = 1.0, 0.0, 0.0, 0.8
kinetic, potential, angular_momentum, states = [], [], [], []
for _ in range(1000):
    radius = hypot(x, y)
    kinetic.append(0.5 * (vx * vx + vy * vy))
    potential.append(-1 / radius)
    angular_momentum.append(x * vy - y * vx)
    states.extend((x, y, vx, vy))
    ax, ay = -x / radius**4, -y / radius**4
    nx, ny = x + vx * dt + 0.5 * ax * dt**2, y + vy * dt + 0.5 * ay * dt**2
    nr = hypot(nx, ny)
    nax, nay = -nx / nr**4, -ny / nr**4
    vx += 0.5 * (ax + nax) * dt
    vy += 0.5 * (ay + nay) * dt
    x, y = nx, ny
print('EPSILONMU_JSON:' + json.dumps({'kinetic': kinetic, 'potential': potential,
                                      'angular_momentum': angular_momentum, 'states': states}))
