"""
Streamlit UI for Multi-Agent System with Real-time Conversation Display
Shows agent interactions as they happen with LangSmith monitoring
"""

import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv
import time

load_dotenv(override=True)

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-creative-system"


# ========== STATE DEFINITION ==========
class AgentState(TypedDict):
    task: str
    round: int
    manager_plan: str
    creative_ideas: list[str]
    creative_discussion: str
    analytical_feedback: list[str]
    analytical_discussion: str
    final_output: str
    next_step: str
    conversation_history: list[dict]

# ========== TOOLS ==========
search_tool = DuckDuckGoSearchRun()

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(
        name="web_search",
        func=search_tool.run,
        description="Search the web for current information."
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="Calculate math expressions."
    )
]

# ========== LLM SETUP ==========
def get_llm(temperature=0.7):
    return ChatOpenAI(
        model="qwen/qwen3-235b-a22b:free",
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

# ========== HELPER FUNCTION ==========
def add_to_conversation(state: AgentState, agent_name: str, message: str, agent_type: str = "info"):
    """Add message to conversation history"""
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []
    
    state["conversation_history"].append({
        "agent": agent_name,
        "message": message,
        "type": agent_type,
        "timestamp": time.time()
    })
    return state

# ========== AGENT NODES ==========

def manager_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.3)
    
    state = add_to_conversation(state, "🎯 Manager", "Starting project planning...", "manager")
    
    prompt = f"""You are a project manager. Create a brief plan for:
Task: {state['task']}

Provide a 2-3 sentence plan. Be concise."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["manager_plan"] = response.content
    state["round"] = 0
    state["next_step"] = "creative"
    
    state = add_to_conversation(state, "🎯 Manager", f"Plan created: {response.content}", "manager")
    
    return state


def ideator_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.9)
    
    round_num = state.get("round", 0) + 1
    state = add_to_conversation(state, "💡 Ideator", f"Round {round_num}: Generating creative ideas...", "creative")
    
    prompt = f"""You are a creative ideator. Generate 3 unique ideas for:
Task: {state['task']}
Plan: {state['manager_plan']}

Format: Just list 3 short ideas (1-2 sentences each)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if "creative_ideas" not in state or state["creative_ideas"] is None:
        state["creative_ideas"] = []
    
    state["creative_ideas"].append(response.content)
    state["next_step"] = "stylist"
    
    state = add_to_conversation(state, "💡 Ideator", response.content, "creative")
    
    return state


def stylist_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.7)
    
    ideas = state.get("creative_ideas", [])[-1] if state.get("creative_ideas") else "No ideas yet"
    
    state = add_to_conversation(state, "🎨 Stylist", "Reviewing and refining ideas...", "creative")
    
    prompt = f"""You are a creative stylist. Review these ideas and improve them:

Ideas:
{ideas}

Provide: 1 refined version combining the best elements. Keep it short (2-3 sentences)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["creative_discussion"] = response.content
    state["round"] = state.get("round", 0) + 1
    
    state = add_to_conversation(state, "🎨 Stylist", response.content, "creative")
    
    if state["round"] < 2:
        state["next_step"] = "ideator"
        state = add_to_conversation(state, "🎨 Stylist", "Needs more iteration. Going back to Ideator...", "creative")
    else:
        state["next_step"] = "analyst"
        state = add_to_conversation(state, "🎨 Stylist", "Creative phase complete. Moving to analysis...", "creative")
    
    return state


def analyst_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.3)
    
    creative_output = state.get("creative_discussion", "No creative output")
    
    state = add_to_conversation(state, "📊 Analyst", "Analyzing creative output...", "analytical")
    
    prompt = f"""You are an analytical reviewer. Evaluate this creative work:

Work:
{creative_output}

Provide: 2-3 bullet points of feedback. Be critical but constructive."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if "analytical_feedback" not in state or state["analytical_feedback"] is None:
        state["analytical_feedback"] = []
    
    state["analytical_feedback"].append(response.content)
    state["next_step"] = "editor"
    
    state = add_to_conversation(state, "📊 Analyst", response.content, "analytical")
    
    return state


def editor_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.5)
    
    feedback = state.get("analytical_feedback", [])[-1] if state.get("analytical_feedback") else "No feedback"
    creative = state.get("creative_discussion", "")
    
    state = add_to_conversation(state, "✏️ Editor", "Reviewing analysis and suggesting improvements...", "analytical")
    
    prompt = f"""You are an editor. Review the analysis and suggest final improvements:

Creative Work:
{creative}

Analyst Feedback:
{feedback}

Provide: 1 concrete recommendation (1-2 sentences)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["analytical_discussion"] = response.content
    
    state = add_to_conversation(state, "✏️ Editor", response.content, "analytical")
    
    round_count = state.get("round", 0)
    if round_count < 3:
        state["next_step"] = "analyst"
        state["round"] = round_count + 1
        state = add_to_conversation(state, "✏️ Editor", "Needs another analytical round...", "analytical")
    else:
        state["next_step"] = "finalizer"
        state = add_to_conversation(state, "✏️ Editor", "Analysis complete. Moving to finalization...", "analytical")
    
    return state


def finalizer_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.5)
    
    state = add_to_conversation(state, "🎁 Finalizer", "Creating final deliverable...", "final")
    
    prompt = f"""You are a finalizer. Combine all work into a final deliverable:

Original Task: {state['task']}
Manager Plan: {state['manager_plan']}
Creative Output: {state.get('creative_discussion', 'N/A')}
Analytical Review: {state.get('analytical_discussion', 'N/A')}

Create: A final polished output (3-5 sentences). Be concise and actionable."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    state["final_output"] = response.content
    state["next_step"] = "end"
    
    state = add_to_conversation(state, "🎁 Finalizer", f"✅ FINAL OUTPUT:\n\n{response.content}", "final")
    
    return state


# ========== ROUTING LOGIC ==========
def route_next(state: AgentState) -> Literal["ideator", "stylist", "analyst", "editor", "finalizer", "end"]:
    next_step = state.get("next_step", "end")
    if next_step == "creative":
        return "ideator"
    elif next_step == "end":
        return "end"
    return next_step


# ========== BUILD GRAPH ==========
def create_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("manager", manager_agent)
    workflow.add_node("ideator", ideator_agent)
    workflow.add_node("stylist", stylist_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("editor", editor_agent)
    workflow.add_node("finalizer", finalizer_agent)
    
    workflow.set_entry_point("manager")
    
    workflow.add_conditional_edges("manager", route_next, {"ideator": "ideator", "end": END})
    workflow.add_conditional_edges("ideator", route_next, {"stylist": "stylist", "end": END})
    workflow.add_conditional_edges("stylist", route_next, {"ideator": "ideator", "analyst": "analyst", "end": END})
    workflow.add_conditional_edges("analyst", route_next, {"editor": "editor", "end": END})
    workflow.add_conditional_edges("editor", route_next, {"analyst": "analyst", "finalizer": "finalizer", "end": END})
    workflow.add_conditional_edges("finalizer", route_next, {"end": END})
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========== STREAMLIT UI ==========

def main():
    st.set_page_config(page_title="Multi-Agent System", page_icon="🤖", layout="wide")
    
    st.title("🤖 Multi-Agent Creative System")
    st.markdown("Watch agents collaborate in real-time with hierarchical discussion workflow")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API keys
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        
        if openrouter_key:
            st.success("✅ OpenRouter API Key loaded")
        else:
            st.error("❌ OpenRouter API Key missing")
        
        if langsmith_key:
            st.success("✅ LangSmith API Key loaded")
            st.info("🔍 Traces available in LangSmith dashboard")
        else:
            st.warning("⚠️ LangSmith API Key missing (optional)")
        
        st.markdown("---")
        st.markdown("### Agent Workflow")
        st.markdown("""
        1. 🎯 **Manager** - Creates plan
        2. 💡 **Ideator** → 🎨 **Stylist** 
        3. 📊 **Analyst** → ✏️ **Editor** 
        4. 🎁 **Finalizer** - Final output
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Task")
        task = st.text_area(
            "Enter your task:",
            value="Create a marketing slogan for an eco-friendly water bottle",
            height=100
        )
        
        run_button = st.button("🚀 Run Multi-Agent System", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 System Status")
        status_placeholder = st.empty()
    
    # Conversation display
    st.markdown("---")
    st.header("💬 Agent Conversation")
    conversation_container = st.container()
    
    # Run the system
    if run_button:
        if not openrouter_key:
            st.error("Please set OPENROUTER_API_KEY in your .env file")
            return
        
        # Clear previous conversation
        conversation_placeholder = conversation_container.empty()
        
        # Initialize state
        initial_state = {
            "task": task,
            "round": 0,
            "manager_plan": "",
            "creative_ideas": [],
            "creative_discussion": "",
            "analytical_feedback": [],
            "analytical_discussion": "",
            "final_output": "",
            "next_step": "",
            "conversation_history": []
        }
        
        config = {"configurable": {"thread_id": "streamlit_session"}}
        app = create_workflow()
        
        # Stream execution
        with st.spinner("Agents are working..."):
            for event in app.stream(initial_state, config):
                # Get current state
                for node_name, node_state in event.items():
                    # Update status
                    status_placeholder.info(f"🔄 Current Agent: **{node_name}**")
                    
                    # Display conversation history
                    conversation_history = node_state.get("conversation_history", [])
                    
                    with conversation_placeholder.container():
                        for msg in conversation_history:
                            agent = msg["agent"]
                            message = msg["message"]
                            msg_type = msg.get("type", "info")
                            
                            # Style based on agent type
                            if msg_type == "manager":
                                st.info(f"**{agent}**\n\n{message}")
                            elif msg_type == "creative":
                                st.success(f"**{agent}**\n\n{message}")
                            elif msg_type == "analytical":
                                st.warning(f"**{agent}**\n\n{message}")
                            elif msg_type == "final":
                                st.balloons()
                                st.success(f"**{agent}**\n\n{message}")
                            else:
                                st.write(f"**{agent}**\n\n{message}")
                    
                    time.sleep(0.5)  # Small delay for visual effect
        
        # Final status
        status_placeholder.success("✅ All agents completed!")
        
        # Display final output prominently
        final_state = app.get_state(config)
        if final_state.values.get("final_output"):
            st.markdown("---")
            st.header("🎉 Final Output")
            st.success(final_state.values["final_output"])
            
            # Download button
            st.download_button(
                label="📥 Download Final Output",
                data=final_state.values["final_output"],
                file_name="agent_output.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()