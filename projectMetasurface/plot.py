import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

data = np.genfromtxt("sim_error_5k_MSE.txt")
#data = np.genfromtxt("sim_error.txt")


fig= plt.figure()

plt.plot(data, label = 'Accuracy')
#plt.plot(data, label = 'Loss')


#fig.suptitle('Loss vs Iterations', fontsize = 20)
#fig.suptitle('Accuracy vs Iterations', fontsize = 20)

plt.xlabel('Number of Iterations')

#plt.ylabel('Accuracy')
# #plt.ylabel('Loss')

plt.legend()
fig.savefig('plot.jpg')
plt.show()
