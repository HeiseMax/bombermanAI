import numpy as np
import matplotlib.pyplot as plt

#f = open("agent_code/n_step_agent/SquaredLoss0.010.5.txt", "r")
#x = f.read()
T =[]
with open("agent_code/n_step_agent/SquaredLossDefault0.010.5.txt", "r") as file1:
    f_list = [float(i) for line in file1 for i in line.split('\t') if i.strip()]
    T = f_list
T = np.array(T)
#T = T.reshape((T.shape[0]//6, 6))
#T = np.sum(T, axis = 1)

plt.plot(T)
#plt.yscale('log')
plt.show()
