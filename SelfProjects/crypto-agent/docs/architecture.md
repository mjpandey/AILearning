# HLL System Architecture

User Query
   ↓
LangGraph Agent
   ↓
Planner (LLM)
   ↓
Tools
   ├── price_tool
   ├── news_tool
   ├── sentiment_tool
   └── portfolio_tool
   ↓
Agent reasoning
   ↓
Final recommendation