class Perceptron():
    def __init__(self, inputs, weights, bias) -> None:
        self.output: float = 0.0
        self.inputs: list[float] = inputs
        self.weights: list[float] = weights
        self.bias: float = bias

    def forward(self) -> None:
        # sum inputs to their weights
        for input, weight in zip(self.inputs, self.weights):
            out = input*weight
            self.output += out
        self.output += self.bias