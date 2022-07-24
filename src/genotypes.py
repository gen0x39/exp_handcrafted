import genotypes
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

ADVRUSH = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

CELL0 = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
CELL1 = Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
CELL2 = Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
CELL3 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))