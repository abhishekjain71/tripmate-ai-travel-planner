from tools.hotel_tool import tavily_search
from tools.flight_tool import search_flights

from backend import run_travel_agent

# res= tavily_search("best hotel in lalitpur up")
# print(res)

# rs= search_flights("plan a 7 days trip from delhi to pune")
# print(rs)


user_input= input("Enter the travel request ")

response= run_travel_agent(
    user_input=user_input,
    thread_id="test_user"

)

print("\n Final response : \n ")
print(response['answer'])