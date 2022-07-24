import random
from genotypes import PRIMITIVES, Genotype
import os

CELL_KINDS = 4

conv = [op for op in PRIMITIVES if 'conv' in op]
pool = [op for op in PRIMITIVES if 'pool' in op]
not_conv = [op for op in PRIMITIVES if 'conv' not in op]

for i in range(CELL_KINDS):
    normal_ops = random.choices(conv, k=6) + random.choices(pool, k=2)
    random.shuffle(normal_ops)
    normal_con = random.choices([0, 1], k=8)

    normal = list(zip(normal_ops, normal_con))
    normal_concat=range(2, 6)

    reduce_ops = random.choices(conv, k=4) + random.choices(not_conv, k=4)
    random.shuffle(reduce_ops)
    reduce_con = [0, 1, 0, 2] * 2
    reduce = list(zip(reduce_ops, reduce_con))
    reduce_concat=range(2, 6)

    cell = Genotype(
    normal = normal,
    normal_concat = normal_concat,
    reduce = reduce,
    reduce_concat = reduce_concat,
    )

    file_name = 'genotypes.py'
    path = os.path.join('.', file_name)

    with open(path, mode='a') as f:
        f.write(f"\nCELL{i} = {str(cell)}")