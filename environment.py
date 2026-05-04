import random

class Environment:
    def __init__(self, agents, objects):
        self.agents = agents
        self.objects = objects

    def step(self, obj):
        a1, a2 = random.sample(self.agents, 2)

        prompt = f"""
A new object is discovered: {obj}.
Give it a name and explain its use.
        """

        r1 = a1.interact(prompt)
        r2 = a2.interact(prompt)

        return {
            "object": obj,
            "agent1": r1,
            "agent2": r2
        }
