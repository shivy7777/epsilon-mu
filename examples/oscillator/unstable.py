"""Injected defect: explicit Euler gains energy on this oscillator."""
import json

oscillator = True
dt = 0.1
x, v = 1.0, 0.0
kinetic, potential, states = [], [], []
for _ in range(1000):
    kinetic.append(0.5 * v * v)
    potential.append(0.5 * x * x)
    states.extend((x, v))
    x, v = x + v * dt, v - x * dt
print('EPSILONMU_JSON:' + json.dumps({'kinetic': kinetic, 'potential': potential, 'states': states}))
