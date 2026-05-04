class LLMInterface:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def generate(self, prompt, memory):
        context = "\n".join([f"{p} -> {r}" for p, r in memory[-5:]])

        full_prompt = f"""
{context}

{prompt}
        """

    
        return "SIMULATED_RESPONSE"
