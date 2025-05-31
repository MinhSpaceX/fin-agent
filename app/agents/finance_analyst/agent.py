import asyncio
from typing import Dict, List
from langgraph.prebuilt import create_react_agent

from agents.agent_library import agent_library
try:
    from agents.prompts import FUNDAMENTAL_ANALYST_PROMPT
except ImportError:
    FUNDAMENTAL_ANALYST_PROMPT: str = "You are a financial analyst helping users with financial queries."

from langgraph_swarm import create_handoff_tool, create_swarm

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

search = TavilySearchResults(max_results=2)

@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> Dict:
    """Fetches the exchange rate for a given base and target currency."""
    try:
        alpha_vantage = AlphaVantageAPIWrapper()
        exchange_rate = alpha_vantage._get_exchange_rate(base_currency, target_currency)
        return exchange_rate
    except Exception as e:
        return {"error": str(e)}

finance_client = MultiServerMCPClient(
    {
        "financial-datasets": {
            "url": "http://mcp-server:8005/sse",
            "transport": "sse",
        },
        "stock-analysis-mcp": {
            "url": "http://yahoo-finance-mcp:8003/sse",
            "transport": "sse",
        },
    }
)

async def fetch_mcp_tools_async(client: MultiServerMCPClient) -> List:
    try:
        tools_list = await client.get_tools()
        return tools_list
    except Exception as e:
        return []

async def setup_swarm_graph():
    finance_mcp_tools = await fetch_mcp_tools_async(finance_client)

    handoff_web_to_fin = create_handoff_tool(
        agent_name="Fin Agent",
        description="Transfer to Fin Agent for financial and crypto tasks.",
    )

    handoff_fin_to_web = create_handoff_tool(
        agent_name="Web Agent",
        description="Transfer to Web Agent for general knowledge and non-finance topics.",
    )

    web_agent = create_react_agent(
        model=model,
        tools=[search, handoff_web_to_fin],
        prompt=(
            "You are a web search agent. Use the search tool. "
            "For finance or crypto tasks, handoff to the Fin Agent."
        ),
        name="Web Agent",
    )

    fin_agent_tools = finance_mcp_tools[:]
    fin_agent_tools.append(get_exchange_rate)
    fin_agent_tools.append(handoff_fin_to_web)
    fin_agent_tools.extend(agent_library["Market_Analyst"]["tools"])

    finance_agent = create_react_agent(
        model=model,
        tools=fin_agent_tools,
        prompt=FUNDAMENTAL_ANALYST_PROMPT,
        name="Fin Agent",
    )

    workflow = create_swarm(
        [finance_agent, web_agent],
        default_active_agent="Fin Agent",
    )

    graph = workflow.compile()
    return graph

graph = asyncio.run(setup_swarm_graph())
