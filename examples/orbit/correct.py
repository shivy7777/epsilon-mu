"""Unit orbit integrated with velocity Verlet."""
import json
from math import hypot

orbit = True
dt = 0.01
x, y, vx, vy = 1.0, 0.0, 0.0, 1.0
kinetic, potential, angular_momentum, states = [], [], [], []
for _ in range(1000):
    radius = hypot(x, y)
    kinetic.append(0.5 * (vx * vx + vy * vy))
    potential.append(-1 / radius)
    angular_momentum.append(x * vy - y * vx)
    states.extend((x, y, vx, vy))
    ax, ay = -x / radius**3, -y / radius**3
    nx, ny = x + vx * dt + 0.5 * ax * dt**2, y + vy * dt + 0.5 * ay * dt**2
    nr = hypot(nx, ny)
    nax, nay = -nx / nr**3, -ny / nr**3
    vx += 0.5 * (ax + nax) * dt
    vy += 0.5 * (ay + nay) * dt
    x, y = nx, ny
print('EPSILONMU_JSON:' + json.dumps({'kinetic': kinetic, 'potential': potential,
                                      'angular_momentum': angular_momentum, 'states': states}))
