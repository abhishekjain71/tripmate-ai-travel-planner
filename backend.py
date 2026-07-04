import os
import certifi

import sqlite3
from dotenv import load_dotenv
load_dotenv()

os.environ["SSL_CERT_FILE"]= certifi.where()
os.environ["REQUEST_CA_BUNDLE"]= certifi.where()

from typing import TypedDict,Annotated
import operator
import uuid



from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain.chat_models import init_chat_model
from tools.hotel_tool import tavily_search
from tools.flight_tool import search_flights

## database--
def database_connection():
    db_paht= os.getenv("DATABASE_PATH", "tripmate.db")
    conn= sqlite3.connect(db_paht, check_same_thread=False)
    return conn

    
    
### llm--
llm= init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider='groq'
)

## state--
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    hotel_results: str
    flight_results: str
    itinerary: str
    llm_calls : int
    
    
### flight agent--
def flight_agent(state: TravelState)->TravelState:
    query= state['user_query']
    flight_data= search_flights(query=query)
    
    return{
        "flight_result": flight_data,
        "messages": [
            AIMessage(content="Flight Result Fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) +1
    }
    

### Hotel Agent--
def hotel_agent(state: TravelState)->TravelState:
    query= f"Best Hotel for {state['user_query']}"
    hotel_results= tavily_search(query)
    
    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


## Build Graph--
graph= StateGraph(TravelState)

## add nodes--
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

## add edges--
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

## memory
conn= database_connection()
checkpointer = SqliteSaver(conn)

## compile--
workflow= graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = workflow.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }