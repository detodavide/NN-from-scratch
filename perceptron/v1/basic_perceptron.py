class Perceptron():
    def __init__(self, inputs, weights, bias) -> None:
        self.output: float = 0.0
        self.inputs: list[float] = inputs
        self.weight: list[float] = weights
        self.bias: float = bias

    def forward(self) -> None:
        # sum inputs to their weights
        for idx, input in enumerate(self.inputs):
            out = input*self.weight[idx]
            self.output += out
        self.output += self.bias