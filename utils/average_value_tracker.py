class AverageValueTracker:
    def __init__(self, keys):
        self.keys = keys
        self.reset()

    def reset(self):
        self.values = {key: 0.0 for key in self.keys}
        self.counts = {key: 0 for key in self.keys}

    def update(self, values, counts=None):
        for key in self.keys:
            self.values[key] += values[key]
            if counts is not None:
                self.counts[key] += counts[key]
            else:
                self.counts[key] += 1

    def get_averages(self):
        averages = {key: self.values[key] / self.counts[key] if self.counts[key] > 0 else 0.0 for key in self.keys}
        return averages