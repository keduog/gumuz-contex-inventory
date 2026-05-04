from src.agent import Agent
from src.environment import Environment
from src.llm_interface import LLMInterface
from src.metrics import Metrics
import yaml

# Load config
config = yaml.safe_load(open("config.yaml"))

# Setup
llm = LLMInterface()
agents = [Agent(i, llm) for i in range(config["num_agents"])]

env = Environment(agents, config["new_objects"])
metrics = Metrics()

# Run
for round in range(config["num_rounds"]):
    for obj in config["new_objects"]:
        result = env.step(obj)

        responses = [result["agent1"], result["agent2"]]
        metrics.update(obj, responses)

# Output
for obj in config["new_objects"]:
    print(obj, "consensus:", metrics.consensus(obj))
