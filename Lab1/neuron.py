class Neuron:

    def __init__(self, value: float):
        self.value = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if not isinstance(value, float):
            raise TypeError("the 'value' must be a float")
        self.__value = value

    def __mul__(self, other):
        return self.value * other

    def __str__(self):
        return str(self.value)
