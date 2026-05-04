class Agent:
    def __init__(self, agent_id, llm):
        self.id = agent_id
        self.memory = []
        self.llm = llm

    def interact(self, prompt):
        response = self.llm.generate(prompt, self.memory)
        self.memory.append((prompt, response))
        return response
