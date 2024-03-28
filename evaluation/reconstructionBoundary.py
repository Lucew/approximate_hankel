import numpy as np
import matplotlib.pyplot as plt


def rank_e(hankel_size: int):
    epsilon = np.linspace(0.00001, 1, 10000)
    return epsilon, 2*np.ceil((2*np.log((8*hankel_size//2)/np.pi)*np.log(16/epsilon))/(np.pi**2))+2

def value_decay(hankel_size: int):
    k = np.arange(1, (hankel_size - 1) // 2 + 1, step=2)
    return k, 16*(np.exp(np.pi**2/(4*np.log(8*(hankel_size//2)/np.pi))))**(-k+2)

comp = rank_e(100)
plt.plot(comp[1], comp[0])
plt.show()
for size in [10, 100, 1000]:
    plt.plot(*value_decay(int(size)))
plt.gca().set_yscale("log")
plt.gca().set_xlim([0, 20])
plt.gca().set_ylim([10e-16, 10e0])
plt.show()
