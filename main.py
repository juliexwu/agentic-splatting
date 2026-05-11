from typing import TypedDict
from langgraph.graph import StateGraph, END
from transformers import pipeline as hf_pipeline
import torch

from dotenv import load_dotenv
from huggingface_hub import login
import os

# Change this to any huggingface model
load_dotenv()
login(token=os.getenv("HF_TOKEN"))
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

device = "cuda" if torch.cuda.is_available() else "cpu"

llm = hf_pipeline(
    "text-generation",
    model=MODEL,
)

def call_llm(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    out = llm(messages)
    return out[0]["generated_text"][-1]["content"].strip()

# This is where we choose what information comes in and out of the agents
class State(TypedDict):
    story: str
    filtered_story: str
    improved_story: str
    feedback: str
    approved: bool
    iteration: int

# Agents go here
def prefiltering_agent(state: State) -> State:
    print("Prefiltering Agent running...")
    result = call_llm(
        f"Fix typos and grammar in this story. Return only the corrected text:\n\n{state['story']}"
    )
    return {**state, "filtered_story": result}


def actor_agent(state: State) -> State:
    print(f"Actor Agent running (iteration {state['iteration']})...")
    feedback_section = f"\n\nAddress this feedback:\n{state['feedback']}" if state["feedback"] else ""
    result = call_llm(
        f"Rewrite this story to be more vivid and engaging.{feedback_section}\n\nStory:\n{state['filtered_story']}"
    )
    return {**state, "improved_story": result}


def critic_agent(state: State) -> State:
    print("Critic Agent running...")
    result = call_llm(
        f"""Review this story. Reply in exactly this format:
        APPROVED: yes or no
        FEEDBACK: one sentence on the most important improvement needed

        Story:
        {state['improved_story']}"""
    )
    approved = "approved: yes" in result.lower()
    feedback = result.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in result else result
    print(f"  approved={approved} | feedback={feedback}")
    return {**state, "feedback": feedback, "approved": approved, "iteration": state["iteration"] + 1}


# Put termination here
def route(state: State) -> str:
    if state["approved"] or state["iteration"] >= 3:
        return "end"
    return "actor_agent"


# This create the langgraph graph and runs the agentic loop
graph = StateGraph(State)
graph.add_node("prefiltering_agent", prefiltering_agent)
graph.add_node("actor_agent", actor_agent)
graph.add_node("critic_agent", critic_agent)

graph.set_entry_point("prefiltering_agent")
graph.add_edge("prefiltering_agent", "actor_agent")
graph.add_edge("actor_agent", "critic_agent")
graph.add_conditional_edges("critic_agent", route, {
    "actor_agent": "actor_agent",
    "end": END,
})

pipeline = graph.compile()

if __name__ == "__main__":
    story = """
    the old man sat by the window. it was raining. he thinked about his son 
    who gone away many years ago. he was sad. the cup of tea in his hand 
    was getting cold. outside a bird landed on the fence.
    """

    result = pipeline.invoke({
        "story": story,
        "filtered_story": "",
        "improved_story": "",
        "feedback": "",
        "approved": False,
        "iteration": 0,
    })

    print("\n--- Final Story ---")
    print(result["improved_story"])
    print(f"\nCompleted in {result['iteration']} iteration(s)")