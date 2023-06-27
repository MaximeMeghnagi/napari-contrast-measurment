import numpy as np

size = 5

sim_x = np.empty((size,), dtype=object)  # Initialisation de sim_x comme un tableau d'objets

for i in range(3):
    sim_x[i] = np.arange(1,5,0.2)

print(sim_x[1])
sim_x = np.array(sim_x)
print(sim_x)

print(sim_x[1][1])