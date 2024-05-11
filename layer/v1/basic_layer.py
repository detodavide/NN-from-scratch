from perceptron.v1.basic_perceptron import Perceptron

class Layer():
    def __init__(self, perceptrons: list[Perceptron]) -> None:
        self.perceptrons: list[Perceptron] = perceptrons
        self.outputs: list[float] = self._get_layer_output()

    def _get_layer_output(self):
        return [perceptron.output for perceptron in self.perceptrons]