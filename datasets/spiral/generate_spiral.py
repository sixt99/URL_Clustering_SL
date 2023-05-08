import math
import numpy as np


def generate_spiral(N):
    pi = math.pi
    # Modifiy the multiplying the following expression to twist the spirals more or twist them less.
    theta = 2 * np.sqrt(np.random.rand(N)) * (2 * pi) # np.linspace(0,2*pi,100)

    r_a = 2 * theta + pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = 1 * (data_a + 0.7 * np.random.randn(N, 2))

    r_b = -2 * theta - pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = 1 * (data_b + 0.7 * np.random.randn(N, 2))

    res_a = np.append(x_a, np.zeros((N, 1), dtype=int), axis=1)
    res_b = np.append(x_b, np.ones((N, 1), dtype=int), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    np.savetxt("spiral.csv", res, delimiter=",", header="x,y,label", comments="", fmt='%.5f')


if __name__ == '__main__':
    generate_spiral(4000)
