import streamlit as st
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv
import time
import json
import operator



load_dotenv(override=True)

# LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-creative-system"


# ========== STATE DEFINITION ==========
class AgentState(TypedDict):
    task: str
    loop_count: Annotated[int, operator.add]  
    manager_plan: str
    creative_ideas: list[str]
    creative_discussion: str
    analytical_feedback: list[str]
    analytical_discussion: str
    final_output: str
    next_agent: str  
    conversation_history: list[dict]
    agent_messages: list  

# ========== TOOLS ==========
search_tool = DuckDuckGoSearchRun()

def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(
        name="web_search",
        func=search_tool.run,
        description="Search the web for current information, facts, trends, or data."
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="Calculate mathematical expressions. Input should be a valid Python expression."
    )
]

# ========== LLM SETUP ==========
def get_llm(temperature=0.7, streaming=True):
    return ChatOpenAI(
        model="qwen/qwen3-235b-a22b:free",
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=streaming
    ).bind_tools(tools)

# ========== HELPER FUNCTIONS ==========
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

def stream_response(llm, messages, placeholder):
    """Stream LLM response token by token"""
    full_response = ""
    tool_calls = []
    
    for chunk in llm.stream(messages):
        if hasattr(chunk, 'content') and chunk.content:
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        
        # Capture tool calls
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
    
    placeholder.markdown(full_response)
    return full_response, tool_calls

def execute_tools(tool_calls, state, agent_name):
    """Execute tool calls and return results"""
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        state = add_to_conversation(
            state, 
            agent_name, 
            f"🔧 Using tool: {tool_name} with args: {tool_args}", 
            "tool"
        )
        
        # Execute tool
        for tool in tools:
            if tool.name == tool_name:
                try:
                    if tool_name == "web_search":
                        result = tool.func(tool_args.get('query', ''))
                    elif tool_name == "calculator":
                        result = tool.func(tool_args.get('expression', ''))
                    else:
                        result = tool.func(**tool_args)
                    
                    results.append({
                        "tool": tool_name,
                        "result": result
                    })
                    
                    state = add_to_conversation(
                        state, 
                        agent_name, 
                        f"📊 Tool result: {result[:200]}...", 
                        "tool"
                    )
                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    results.append({
                        "tool": tool_name,
                        "result": error_msg
                    })
                    state = add_to_conversation(
                        state, 
                        agent_name, 
                        f"❌ {error_msg}", 
                        "tool"
                    )
    
    return results, state

# ========== AGENT NODES WITH STREAMING ==========

def manager_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.3)
    
    state = add_to_conversation(state, "🎯 Manager", "Starting project planning...", "manager")
    
    prompt = f"""You are a project manager. Analyze this task and decide the workflow:

Task: {state['task']}

1. Create a 2-3 sentence plan
2. Decide which agent should handle this FIRST (choose ONE):
   - "ideator" for creative/brainstorming tasks
   - "analyst" for analytical/data tasks
   - "finalizer" if task is simple and direct

Use tools if needed (web_search for trends, calculator for numbers).

Respond in JSON format:
{{"plan": "your plan", "next_agent": "agent_name", "reasoning": "why this agent"}}"""
    
    # Create placeholder for streaming
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    # Handle tool calls if any
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "🎯 Manager")
        
        # Re-prompt with tool results
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up_prompt = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide your final answer."
        
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up_prompt)], placeholder)
    
    # Parse JSON response
    try:
        response_data = json.loads(response.strip().replace("```json", "").replace("```", ""))
        state["manager_plan"] = response_data["plan"]
        state["next_agent"] = response_data["next_agent"]
        
        state = add_to_conversation(
            state, 
            "🎯 Manager", 
            f"Plan: {response_data['plan']}\n\nNext Agent: {response_data['next_agent']}\nReasoning: {response_data['reasoning']}", 
            "manager"
        )
    except:
        state["manager_plan"] = response
        state["next_agent"] = "ideator"
        state = add_to_conversation(state, "🎯 Manager", f"Plan: {response}", "manager")
    
    return state


def ideator_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.9)
    
    state = add_to_conversation(state, "💡 Ideator", "Generating creative ideas...", "creative")
    
    prompt = f"""You are a creative ideator. Generate 3 innovative ideas:

Task: {state['task']}
Plan: {state['manager_plan']}
Previous ideas: {state.get('creative_ideas', [])}

Use web_search to find current trends if helpful. List 3 ideas (2-3 sentences each).

Then decide next agent:
- "stylist" to refine ideas
- "analyst" if ideas need validation
- "finalizer" if ready

Format: 
Ideas:
1. ...
2. ...
3. ...

Next Agent: [agent_name]
Reasoning: [why]"""
    
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    # Handle tools
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "💡 Ideator")
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide your ideas."
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up)], placeholder)
    
    if "creative_ideas" not in state:
        state["creative_ideas"] = []
    state["creative_ideas"].append(response)
    
    # Parse next agent
    if "Next Agent:" in response:
        next_agent = response.split("Next Agent:")[1].split("\n")[0].strip().lower()
        state["next_agent"] = next_agent if next_agent in ["stylist", "analyst", "finalizer"] else "stylist"
    else:
        state["next_agent"] = "stylist"
    
    state = add_to_conversation(state, "💡 Ideator", response, "creative")
    
    return state


def stylist_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.7)
    
    state = add_to_conversation(state, "🎨 Stylist", "Refining creative ideas...", "creative")
    
    ideas = state.get("creative_ideas", [])[-1] if state.get("creative_ideas") else "No ideas"
    
    prompt = f"""You are a creative stylist. Refine and polish these ideas:

Ideas:
{ideas}

Create 1 polished version (3-4 sentences). Use web_search for style trends if needed.

Then decide next agent:
- "analyst" for critical review
- "ideator" if needs more creativity (only if loop_count < 2)
- "finalizer" if ready

Current loop_count: {state.get('loop_count', 0)}

Format:
Refined Output:
[your refined version]

Next Agent: [agent_name]
Reasoning: [why]"""
    
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "🎨 Stylist")
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide refined output."
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up)], placeholder)
    
    state["creative_discussion"] = response
    
    # Parse next agent with loop limit
    if "Next Agent:" in response:
        next_agent = response.split("Next Agent:")[1].split("\n")[0].strip().lower()
        if next_agent == "ideator" and state.get("loop_count", 0) >= 2:
            state["next_agent"] = "analyst"
        else:
            state["next_agent"] = next_agent if next_agent in ["analyst", "ideator", "finalizer"] else "analyst"
    else:
        state["next_agent"] = "analyst"
    
    state = add_to_conversation(state, "🎨 Stylist", response, "creative")
    
    return state


def analyst_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.3)
    
    state = add_to_conversation(state, "📊 Analyst", "Analyzing output...", "analytical")
    
    creative_output = state.get("creative_discussion", "No output")
    
    prompt = f"""You are an analytical reviewer. Evaluate this work:

Work:
{creative_output}

Use web_search for data validation or calculator for metrics if needed.
Provide 3 bullet points of critical feedback.

Then decide next agent:
- "editor" for refinement suggestions
- "finalizer" if analysis is complete

Format:
Analysis:
- ...
- ...
- ...

Next Agent: [agent_name]
Reasoning: [why]"""
    
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "📊 Analyst")
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide analysis."
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up)], placeholder)
    
    if "analytical_feedback" not in state:
        state["analytical_feedback"] = []
    state["analytical_feedback"].append(response)
    
    # Parse next agent
    if "Next Agent:" in response:
        next_agent = response.split("Next Agent:")[1].split("\n")[0].strip().lower()
        state["next_agent"] = next_agent if next_agent in ["editor", "finalizer"] else "editor"
    else:
        state["next_agent"] = "editor"
    
    state = add_to_conversation(state, "📊 Analyst", response, "analytical")
    
    return state


def editor_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.5)
    
    state = add_to_conversation(state, "✏️ Editor", "Editing and suggesting improvements...", "analytical")
    
    feedback = state.get("analytical_feedback", [])[-1] if state.get("analytical_feedback") else "No feedback"
    
    prompt = f"""You are an editor. Review analysis and suggest improvements:

Analyst Feedback:
{feedback}

Provide 1 concrete recommendation (2-3 sentences).

Then decide next agent:
- "analyst" if needs more analysis (only if loop_count < 2)
- "finalizer" to complete the work

Current loop_count: {state.get('loop_count', 0)}

Format:
Recommendation:
[your recommendation]

Next Agent: [agent_name]
Reasoning: [why]"""
    
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "✏️ Editor")
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide recommendation."
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up)], placeholder)
    
    state["analytical_discussion"] = response
    
    # Parse next agent with loop limit
    if "Next Agent:" in response:
        next_agent = response.split("Next Agent:")[1].split("\n")[0].strip().lower()
        if next_agent == "analyst" and state.get("loop_count", 0) >= 2:
            state["next_agent"] = "finalizer"
        else:
            state["next_agent"] = next_agent if next_agent in ["analyst", "finalizer"] else "finalizer"
    else:
        state["next_agent"] = "finalizer"
    
    state = add_to_conversation(state, "✏️ Editor", response, "analytical")
    
    return state


def finalizer_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.5)
    
    state = add_to_conversation(state, "🎁 Finalizer", "Creating final deliverable...", "final")
    
    prompt = f"""You are a finalizer. Create the final polished deliverable:

Task: {state['task']}
Manager Plan: {state['manager_plan']}
Creative Work: {state.get('creative_discussion', 'N/A')}
Analysis: {state.get('analytical_discussion', 'N/A')}

Create a final output (4-6 sentences). Be concise, actionable, and professional."""
    
    placeholder = st.empty()
    response, tool_calls = stream_response(llm, [HumanMessage(content=prompt)], placeholder)
    
    if tool_calls:
        tool_results, state = execute_tools(tool_calls, state, "🎁 Finalizer")
        tool_context = "\n".join([f"{r['tool']}: {r['result']}" for r in tool_results])
        follow_up = f"{prompt}\n\nTool Results:\n{tool_context}\n\nNow provide final output."
        placeholder = st.empty()
        response, _ = stream_response(llm, [HumanMessage(content=follow_up)], placeholder)
    
    state["final_output"] = response
    state["next_agent"] = "end"
    
    state = add_to_conversation(state, "🎁 Finalizer", f"✅ FINAL OUTPUT:\n\n{response}", "final")
    
    return state


# ========== DYNAMIC ROUTING ==========
def route_next(state: AgentState) -> Literal["ideator", "stylist", "analyst", "editor", "finalizer", "end"]:
    """Dynamic routing based on agent decisions"""
    next_agent = state.get("next_agent", "end")
    
    # Safety: enforce max loop count
    if state.get("loop_count", 0) >= 2 and next_agent in ["ideator", "analyst"]:
        return "finalizer"
    
    if next_agent == "end":
        return "end"
    
    return next_agent


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
    
    # Dynamic conditional edges
    workflow.add_conditional_edges(
        "manager", 
        route_next, 
        {"ideator": "ideator", "analyst": "analyst", "finalizer": "finalizer", "end": END}
    )
    workflow.add_conditional_edges(
        "ideator", 
        route_next, 
        {"stylist": "stylist", "analyst": "analyst", "finalizer": "finalizer", "end": END}
    )
    workflow.add_conditional_edges(
        "stylist", 
        route_next, 
        {"ideator": "ideator", "analyst": "analyst", "finalizer": "finalizer", "end": END}
    )
    workflow.add_conditional_edges(
        "analyst", 
        route_next, 
        {"editor": "editor", "finalizer": "finalizer", "end": END}
    )
    workflow.add_conditional_edges(
        "editor", 
        route_next, 
        {"analyst": "analyst", "finalizer": "finalizer", "end": END}
    )
    workflow.add_conditional_edges("finalizer", route_next, {"end": END})
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ========== STREAMLIT UI ==========

def main():
    st.set_page_config(page_title="Enhanced Multi-Agent System", page_icon="🤖", layout="wide")
    
    st.title("🤖 Enhanced Multi-Agent System")
    st.markdown("**Features:** Tool Usage | Dynamic Routing | Real-time Streaming | Max 2 Loops")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        langsmith_key = os.getenv("LANGCHAIN_API_KEY")
        
        if openrouter_key:
            st.success("✅ OpenRouter API Key loaded")
        else:
            st.error("❌ OpenRouter API Key missing")
        
        if langsmith_key:
            st.success("✅ LangSmith API Key loaded")
        else:
            st.warning("⚠️ LangSmith API Key missing")
        
        st.markdown("---")
        st.markdown("### 🛠️ Available Tools")
        st.markdown("""
        - 🔍 **Web Search** - Current info & trends
        - 🧮 **Calculator** - Math operations
        """)
        
        st.markdown("### 🔄 Dynamic Workflow")
        st.markdown("""
        Agents decide their own path:
        - 🎯 Manager → Any agent
        - 💡 Ideator → Stylist/Analyst/Finalizer
        - 🎨 Stylist → Ideator/Analyst/Finalizer
        - 📊 Analyst → Editor/Finalizer
        - ✏️ Editor → Analyst/Finalizer
        - Max 2 loop iterations enforced
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Task")
        task = st.text_area(
            "Enter your task:",
            value="Create a data-driven marketing campaign for sustainable fashion targeting Gen Z",
            height=100
        )
        
        run_button = st.button("🚀 Run Enhanced System", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 System Status")
        status_placeholder = st.empty()
        loop_counter = st.empty()
    
    # Conversation display
    st.markdown("---")
    st.header("💬 Agent Conversation (Real-time Streaming)")
    conversation_container = st.container()
    
    # Run system
    if run_button:
        if not openrouter_key:
            st.error("Please set OPENROUTER_API_KEY in your .env file")
            return
        
        # Initialize
        initial_state = {
            "task": task,
            "loop_count": 0,
            "manager_plan": "",
            "creative_ideas": [],
            "creative_discussion": "",
            "analytical_feedback": [],
            "analytical_discussion": "",
            "final_output": "",
            "next_agent": "",
            "conversation_history": [],
            "agent_messages": []
        }
        
        config = {"configurable": {"thread_id": f"streamlit_{time.time()}"}}
        app = create_workflow()
        
        # Stream execution
        with st.spinner("🔄 Agents collaborating..."):
            for event in app.stream(initial_state, config):
                for node_name, node_state in event.items():
                    # Update status
                    status_placeholder.info(f"🤖 Active Agent: **{node_name.upper()}**")
                    loop_counter.metric("Loop Count", node_state.get("loop_count", 0), 
                                      delta="Max: 2", delta_color="normal")
                    
                    # Display conversation
                    conversation_history = node_state.get("conversation_history", [])
                    
                    with conversation_container:
                        for msg in conversation_history[-1:]:  # Show only latest message
                            agent = msg["agent"]
                            message = msg["message"]
                            msg_type = msg.get("type", "info")
                            
                            if msg_type == "manager":
                                st.info(f"**{agent}**\n\n{message}")
                            elif msg_type == "creative":
                                st.success(f"**{agent}**\n\n{message}")
                            elif msg_type == "analytical":
                                st.warning(f"**{agent}**\n\n{message}")
                            elif msg_type == "tool":
                                st.code(f"{agent}\n{message}")
                            elif msg_type == "final":
                                st.balloons()
                                st.success(f"**{agent}**\n\n{message}")
                            else:
                                st.write(f"**{agent}**\n\n{message}")
        
        status_placeholder.success("✅ All agents completed!")
        
        # Final output
        final_state = app.get_state(config)
        if final_state.values.get("final_output"):
            st.markdown("---")
            st.header("🎉 Final Deliverable")
            st.success(final_state.values["final_output"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Output",
                    data=final_state.values["final_output"],
                    file_name="agent_output.txt",
                    mime="text/plain"
                )
            with col2:
                st.metric("Total Loops", final_state.values.get("loop_count", 0))


if __name__ == "__main__":
    main()