from langgraph.graph import StateGraph , END
from langgraph.checkpoint.memory import  MemorySaver
# other imports 
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict , Annotated 
from dotenv import load_dotenv
import operator
load_dotenv(override=True)
import os
import time
# langsmith env

class AgentState(TypedDict):
    task : str
    loop_count : Annotated[int , operator.add]
    manager_plan : str
    creative_ideas: str
    creative_discussion : str
    analytical_feedback : str
    analytical_discussion: str
    final_output:str
    next_agent: str
    conversation_history: list[dict]
    agent_message: list

# adding tools

search_tool = DuckDuckGoSearchRun()

def calculator(expression : str) -> str:
    """calculate the mathamtical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error:{str(e)}"


tools = [
    Tool(
        name="web search",
        func=search_tool.run,
        description="search the web for the current information , data , facts , trends"

    ),
    Tool(
        name="calculator",
        function=calculator,
        description="Calculate the mathmatical expression. Output should be a valid python expressiion"
    )
]

# calling LLm
def get_llm(temperature=0.5, streaming = True):
    return ChatOpenAI(
        model="qwen/qwen3-235b-a22b:free",
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_ai_base = "https://openrouter.ai/api/v1",
        streaming = streaming

    ).bind_tools(tools)

# helper functions
def add_to_conversation(state :AgentState,agent_name:str,messsage:str , agent_type:str ="info"):
    """Add message to conversation history"""
    if "conversation_history"  not in state or "conversation_history" is None:
        state["conversation_history"] =[]

    state["conversation_history"].append({
        "agent" : agent_name,
        "message":messsage,
        "type": agent_type,
        "timestamp": time.time()
    })

    return state


