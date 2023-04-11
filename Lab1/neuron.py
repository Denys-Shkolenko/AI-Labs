class Neuron:

    def __init__(self, value: int | float):
        self.value = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("the 'value' must be int or float")
        self.__value = value
