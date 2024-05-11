import numpy as np
from perceptron.v1.basic_perceptron import Perceptron
from layer.v1.basic_layer import Layer
import random

random.seed(14)

def sample_floats(low, high, k=1) -> list[float]:
    """ Return a k-length list of random floats
        in the range of low <= x <= high
    """
    result = []
    for i in range(k):
        x = random.uniform(low, high)
        result.append(x)
    return result

def get_random_neurons(num_neurons: int, inputs: list[float]) -> list[Perceptron]:

    neurons_list: list[Perceptron] = []
    for _ in range(num_neurons):
        weights = sample_floats(-1, 1, k=4)
        bias = random.random()
        node = Perceptron(inputs, weights, bias)
        node.forward()
        neurons_list.append(node)
    return neurons_list

def main():
    inputs = sample_floats(-5, 5, k=4)
    neurons_list = get_random_neurons(num_neurons=3, inputs=inputs)
    layer = Layer(neurons_list)
    print(layer.outputs)

if __name__=='__main__':
    main()
