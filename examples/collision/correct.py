"""Equal-mass elastic collision swaps velocities."""
import json

collision = True
m1 = m2 = 1.0
before = m1 * 1 + m2 * 0
after = m1 * 0 + m2 * 1
print('EPSILONMU_JSON:' + json.dumps({'states': [1, 0, 0, 1], 'linear_momentum': [before, after]}))
