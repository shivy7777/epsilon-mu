from epsilonmu.relationships import extract_relationships, build_graph
from epsilonmu.relationships.symbolic import parse_expression, symbol_power
from epsilonmu.context import extract_context
from epsilonmu.dimensions import MASS, LENGTH, TIME, CURRENT, TEMPERATURE, FORCE, ENERGY


def test_equivalent_radius_powers_and_reordered_factors():
    forms = ['G*M*m/r**2', 'm*G*M/(r*r)', 'M*m*G/pow(r, 2)']
    assert len({parse_expression(form) for form in forms}) == 1
    assert symbol_power(parse_expression(forms[0]), 'r') == -2


def test_sqrt_numpy_and_math_are_normalized():
    forms = ['np.sqrt(x**2+y**2)', 'math.sqrt(y*y+x*x)', 'sqrt(pow(y,2)+pow(x,2))']
    assert len({parse_expression(form) for form in forms}) == 1


def test_extraction_graph_and_updates_ignore_variable_meaning():
    source = 'import numpy as np\na = np.sqrt(b*b+c*c)\nd = 2/(a*a)\ne += d*h\n'
    relations = extract_relationships(source)
    assert [(r.target, r.kind) for r in relations] == [('a', 'assignment'), ('d', 'assignment'), ('e', 'update')]
    assert relations[1].dependencies == ('a',)
    graph = build_graph(relations)
    assert graph.path('b', 'e') == ('b', 'a', 'd', 'e')
    assert graph.path('c', 'e') == ('c', 'a', 'd', 'e')


def test_parallel_component_assignments_extend_graph():
    source = 'r = hypot(x,y)\nax, ay = -x/r**3, -y/r**3\nvx += ax*dt\nx, y = x+vx*dt, y+vy*dt\n'
    relations = extract_relationships(source)
    assert {'r', 'ax', 'ay', 'vx', 'x', 'y'} <= {item.target for item in relations}
    graph = build_graph(relations)
    assert graph.path('r', 'vx') == ('r', 'ax', 'vx')


def test_relationship_location_and_no_expression_execution():
    source = 'x = danger()\ny = x + 2\n'
    relations = extract_relationships(source)
    assert len(relations) == 1
    assert relations[0].target == 'y'
    assert relations[0].line == 2
    assert relations[0].provenance == 'ast'


def test_context_docstring_comment_and_no_false_earth():
    item = extract_context('"""Newtonian gravity in SI units near Earth\'s surface."""\n# two-body orbit\nx = 1\n')
    assert 'newtonian_gravity' in item.labels
    assert 'earth_surface' in item.labels
    assert 'si_units' in item.labels
    assert extract_context('g = 1.6\n').labels == ()


def test_dimension_algebra():
    assert MASS * LENGTH / (TIME ** 2) == FORCE
    assert FORCE * LENGTH == ENERGY
    assert CURRENT != TEMPERATURE
