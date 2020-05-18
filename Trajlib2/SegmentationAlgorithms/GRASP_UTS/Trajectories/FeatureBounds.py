
class FeatureBounds:
    def __init__(self, values):
        values.sort()
        self.min = values[0]
        self.max = values[len(values)-1]