import os
import sys
import genotypes
from graphviz import Digraph


def plot(genotype_name, genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("cell_{k-2}", fillcolor='darkseagreen2')
  g.node("cell_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  # hidden node
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    # i = 0 -> k = 0, 1
    # i = 1 -> k = 2, 3
    # ...
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "cell_{k-2}"
      elif j == 1:
        u = "cell_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("cell_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "cell_{k}", fillcolor="gray")
  
  root = os.path.join("outputs", "visualize_cell")
  output_path = os.path.join(root, genotype_name)
  g.render(os.path.join(output_path, filename), view=True)
  os.remove(os.path.join(output_path, filename))

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype_name, genotype.normal, f"{genotype_name}_normal")
  plot(genotype_name, genotype.reduce, f"{genotype_name}_reduction")