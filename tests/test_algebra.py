from epsilonmu.relationships.symbolic import equivalent, parse_expression


def test_distributed_vector_component():
    assert equivalent(parse_expression('a*(u[1]+b)'), parse_expression('a*u[1]+a*b'))


def test_square_root_keeps_absolute_value():
    assert not equivalent(parse_expression('np.sqrt(np.square(q))'), parse_expression('q'))
