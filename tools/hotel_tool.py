import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from tavily import TavilyClient
client= TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def tavily_search(query: str)->str:
    
    """Search the web for hotel information, prices, and recommendations 
    for a given city. Use this when you need to find hotels, accommodation 
    options, or general travel-related information.
    """
    try:
        response=client.search( query=query,max_results=5)
        if not response.get("results"):
            return "No results found for this search"
        
        results= []

        for i, r in enumerate(response['results'],1):
            title= r.get("title","Unknown")
            url= r.get("url", "")
            snippet= r.get("content","").strip()
            if len(snippet)> 300:
                snippet=snippet[:300].rsplit(" ",1)[0] + "..."
                
            results.append(f"{i}. **{title}**\n {url}\n {snippet}")
            
        return "\n\n".join(results)
    
    except Exception as e:
        return f"Search failed : {str(e)}"

    
        