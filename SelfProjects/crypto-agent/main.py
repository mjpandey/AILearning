from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables (e.g., API keys, LangSmith config)
load_dotenv()

def main():
    print("="*50)
    print("🤖 Crypto Agent Central Hub")
    print("="*50)
    print("Select a workflow to run:")
    print("  1. SingleAgent Flow (graphs/trading_graph.py)")
    print("  2. MultiAgent Sequential Flow (graphs/trading_graph.py)")
    print("  3. Conditional Workflow Agent (agents/multi_agent.py)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice not in ["1", "2", "3"]:
        print("\nInvalid choice. Exiting.")
        return
        
    query = input("\nEnter your query (e.g., 'Should I long BTC today?'): ").strip()
    if not query:
        print("\nQuery cannot be empty. Exiting.")
        return
        
    if choice == "1":
        from graphs.trading_graph import app as single_agent_app
        print("\n" + "-"*40)
        print("🚀 Running SingleAgent Flow")
        print("-"*40)
        
        result = single_agent_app.invoke({"query": query})
        print("\n✅ Final Result:")
        print(result["result"])
        
    elif choice == "2":
        # Based on user input, option 2 invokes the app from graphs/trading_graph.py for now
        from graphs.trading_multiagent_graph import app as multi_agent_seq_app
        print("\n" + "-"*40)
        print("🚀 Running MultiAgent Sequential Flow")
        print("-"*40)
        
        result = multi_agent_seq_app.invoke({
            "messages":[HumanMessage(content=query)]})
        print("\n✅ Final Result:")
        print(result["final_decision"])
        
    elif choice == "3":
        from agents.multi_agent import app as conditional_workflow_app
        print("\n" + "-"*40)
        print("🚀 Running Conditional Workflow Agent")
        print("-"*40)
        
        initial_payload = {
            "messages": [HumanMessage(content=query)],
            "metrics": {}
        }
        
        final_metrics = {}
        
        # Stream the graph execution to show the step-by-step routing
        for s in conditional_workflow_app.stream(initial_payload):
            if "__end__" not in s:
                node_name = list(s.keys())[0]
                print(f"\n[Node Execution: {node_name}]")
                
                state_update = s[node_name]
                
                # Print the assistant's message if they replied
                if "messages" in state_update:
                    print(state_update["messages"][-1].content)
                
                # Print the routing decision if the supervisor acted
                if "next" in state_update:
                    print(f"Routing task to -> {state_update['next']}")
                    
                # Save the latest tracking metrics
                if "metrics" in state_update:
                    final_metrics = state_update["metrics"]
                    
        # Print final observability summary for this execution
        print("\n" + "-"*40)
        print("📊 Execution Metrics Summary:")
        if final_metrics:
            for key, val in final_metrics.items():
                if "time" in key:
                    print(f"   - {key.replace('_', ' ').title()}: {val:.2f} seconds")
                else:
                    print(f"   - {key.replace('_', ' ').title()}: {val} calls")
        else:
            print("   - No worker nodes engaged.")
        print("-" * 40 + "\n--- Done ---")

if __name__ == "__main__":
    main()