from numpy import dot
from numpy.linalg import norm
import numpy as np


def cosineSimilarity(A, B):
       return dot(A, B)/(norm(A)*norm(B))


doc1 = [0.01, 0.00, 0.01, 0.02, 0.05]
doc2 = [0.01, 0.05, 0.00, 0.01, 0.00]
doc3 = [0.02, 0.00, 0.02, 0.04, 0.01]
doc4 = [0.01, 0.00, 0.01, 0.02, 0.05]

npdoc1 = np.array(doc1)
npdoc2 = np.array(doc2)
npdoc3 = np.array(doc3)
npdoc4 = np.array(doc4)

sim1 =0.0
sim2 = 0.0
sim1 = cosineSimilarity(npdoc1,npdoc2)
sim2 = cosineSimilarity(npdoc1,npdoc4)
print(sim1)
print(sim2)
