from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.bms import BMSContext, LearningBrainAgent

def main():
    brain = LearningBrainAgent(epsilon=0.0)
    context = BMSContext(timestamp=0.0, metrics={"loads": [0.9]}, active_requests=({"id": 1, "leave_time": 10.0},))
    for step in range(5):
        result = brain.decide(context)
        assert result["action_type"] in brain.MACRO_ACTIONS
        assert "reasoning" in result
        print(step, result["action_type"], result["reasoning"])
    print("BMS smoke passed")

if __name__ == "__main__": main()
