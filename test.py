from tools.hotel_tool import tavily_search
from tools.flight_tool import search_flights


# res= tavily_search("best hotel in lalitpur up")
# print(res)

rs= search_flights("plan a 7 days trip from delhi to pune")
print(rs)

