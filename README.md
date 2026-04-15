# 🤖 Enhanced Multi-Agent AI System with Real-Time Streaming & Notifications

An advanced **LLM-powered multi-agent system** built using **LangGraph, LangChain, and Streamlit**, where multiple intelligent agents collaborate dynamically to solve complex tasks.

This system goes beyond a chatbot — it simulates a **real-world AI workflow pipeline** with planning, creativity, analysis, refinement, and final delivery, including **phone notifications via Pushover**.

---

## 🚀 Features

- 🧠 **Multi-Agent Architecture**
  - Manager, Ideator, Stylist, Analyst, Editor, Finalizer, Pushover Agent
- 🔄 **Dynamic Routing**
  - Agents decide the next step autonomously
- 🛠️ **Tool Integration**
  - Web Search (DuckDuckGo)
  - Calculator
- ⚡ **Real-Time Streaming**
  - Token-by-token response rendering in Streamlit
- 🔁 **Controlled Iterations**
  - Max 2 loop cycles to avoid infinite reasoning
- 📱 **Pushover Integration**
  - Sends final output directly to your phone
- 📊 **Conversation Tracking**
  - Full agent interaction history with timestamps
- 🎯 **Production-Like Workflow**
  - Input → Planning → Execution → Analysis → Output → Notification

---

## 🏗️ System Architecture

User Input
↓
🎯 Manager Agent (Planning & Routing)
↓
💡 Ideator Agent (Idea Generation)
↓
🎨 Stylist Agent (Refinement)
↓
📊 Analyst Agent (Critical Evaluation)
↓
✏️ Editor Agent (Improvements)
↓
🎁 Finalizer Agent (Final Output)
↓
📱 Pushover Agent (Notification)
↓
End


---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM Orchestration:** LangGraph  
- **LLM Framework:** LangChain  
- **Model Provider:** OpenRouter (Qwen 3 235B)  
- **Tools:** DuckDuckGo Search, Custom Calculator  
- **Notifications:** Pushover API  
- **Tracing:** LangSmith  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/multi-agent-system.git
cd multi-agent-system

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

OPENROUTER_API_KEY=your_openrouter_api_key
LANGCHAIN_API_KEY=your_langsmith_key

# Optional (for notifications)
PUSHOVER_TOKEN=your_pushover_app_token
PUSHOVER_USER=your_pushover_user_key

streamlit run app22.py
