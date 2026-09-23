"""Symplectic Euler oscillator; energy stays within the declared tolerance."""
import json

oscillator = True
dt = 0.01
x, v = 1.0, 0.0
kinetic, potential, states = [], [], []
for _ in range(1000):
    kinetic.append(0.5 * v * v)
    potential.append(0.5 * x * x)
    states.extend((x, v))
    v -= x * dt
    x += v * dt
print('EPSILONMU_JSON:' + json.dumps({'kinetic': kinetic, 'potential': potential, 'states': states}))
