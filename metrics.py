from collections import defaultdict

class Metrics:
    def __init__(self):
        self.names = defaultdict(list)

    def extract_name(self, response):
        # naive: first word as name
        return response.split()[0]

    def update(self, obj, responses):
        for r in responses:
            name = self.extract_name(r)
            self.names[obj].append(name)

    def consensus(self, obj):
        names = self.names[obj]
        return len(set(names)) / len(names) if names else 1.0
