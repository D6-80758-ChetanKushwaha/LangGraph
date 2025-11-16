from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver  # ✅ correct
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="")
from langgraph.types import Command,interrupt
from typing import TypedDict, Optional, Annotated
class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: Optional[str]  # "rag", "travel", "finance", etc.

# ---- RAG Agent ----
rag_graph = StateGraph(AppState)
def rag_agent(state):
    
    print("RAG Agent Invoked")
    last = state["messages"][-1].content.lower()

    # If user asks travel-related stuff → handoff to travel agent
    if "flight" in last or "travel" in last:
        print("Handoff (rag → travel)")
        # Ask user if they want to switch
        confirm = interrupt("Would you like to switch to Travel agent? (yes/no)")
        # After resuming, confirm will be the user's answer
        if confirm.strip().lower() == "yes":
            print("User confirmed → Handoff (rag → travel)")
            return Command(
                goto="travel_node",
                update={"active_agent":"travel"}
            )
        else:
            print("User declined handoff → stay in RAG")
            # continue in rag agent: maybe respond here
            resp = llm.invoke(state["messages"])
            return Command(update={"messages": [resp],"active_agent":"rag"})


    
    resp = llm.invoke(state["messages"])
    # resp.content = "📚 "  # RAG agent flair
    return Command(
        update={"messages": [resp], "active_agent": "rag"}
    )
    

rag_graph.add_node("rag", rag_agent)
rag_graph.set_entry_point("rag")
rag_graph.set_finish_point("rag")
rag_graph = rag_graph.compile()



# ---- Travel Agent ----
travel_graph = StateGraph(AppState)
def travel_agent(state):
    
    print("Travel Agent Invoked")
    
    last_msg = state["messages"][-1].content.lower()

    # --- Handoff detection ---
    if "rag" in last_msg or "switch to rag" in last_msg:
        print("Handoff (travel → rag)")
        return Command(
            goto="rag_node",
            update={"active_agent": "rag"},
            graph=Command.PARENT
        )
    
    resp = llm.invoke(state["messages"])
    # resp.content = "🌍✈️ "  # Travel agent flair
    return Command(
        update={"messages": [resp], "active_agent": "travel"}
    )


travel_graph.add_node("travel", travel_agent)
travel_graph.set_entry_point("travel")
travel_graph.set_finish_point("travel")
travel_graph = travel_graph.compile()




class RouteAgent(BaseModel):
    agent: str = Field(..., description="One of ['rag', 'travel']")

router_llm = llm.with_structured_output(RouteAgent)



def router_node(state: AppState):
    if state.get("active_agent"):
        return {"active_agent": state["active_agent"]}

    print("Router Invoked")
    # extract last user message
    last_msg = state["messages"][-1].content

    route = router_llm.invoke(last_msg)

    return {"active_agent": route.agent}



def route_next(state: AppState):
    if state["active_agent"] == "rag":
        return "rag_node"
    elif state["active_agent"] == "travel":
        return "travel_node"
    return "__end__"



router_graph = StateGraph(AppState)
router_graph.add_node("router", router_node)

# Link subgraphs as callable nodes
router_graph.add_node("rag_node", rag_graph)
router_graph.add_node("travel_node", travel_graph)

# Conditional edge based on routing
router_graph.add_conditional_edges("router", route_next, {
    "rag_node": "rag_node",
    "travel_node": "travel_node",
    "__end__": "__end__"
})

router_graph.set_entry_point("router")
# router_graph.set_finish_point("__end__")
# memory = SqliteSaver.from_conn_string(":memory:") 

# compiled_graph = router_graph.compile(checkpointer=memory)



# Manual activation via frontend icon
state = {"messages": [("user", "switch to travel")], "active_agent": "rag"}

with SqliteSaver.from_conn_string("app_memory.db") as memory:
    compiled_graph = router_graph.compile(checkpointer=memory)
    
    # Now you can run the graph safely
    result = compiled_graph.invoke(
        state,
        config={"configurable": {"thread_id": "my-thread-1"}}
    )
    print(result["messages"][-1].content)
    
    
    ans = input()
    
    result = compiled_graph.invoke(Command(resume=ans), config={"configurable": {"thread_id": "my-thread-1"}})
    print(result["messages"][-1].content)
    