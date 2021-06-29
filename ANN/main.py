import numpy as np
from model import Model


def main():
    topology = []
    topology.append(2)
    topology.append(3)
    topology.append(2)

    model = Model(topology)
    model.set_eta(0.09)

    while True:
        err = 0
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])
        for i in range(len(inputs)):
            err = err + model.train(inputs[i], outputs[i])
        print ("error: ", err)
        if err < 0.05:
            break

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        print (model.test([float(a), float(b)]))


if __name__ == '__main__':
    main()