import yaml
# Inline YAML config
CONFIG_YAML = '''genie_agents:
- agent_name: GENIE_REXEL
  description: Agent for GENIE_REXEL
  space_id: 01f0972a3e1c1d2097052cfdc24adb11
  tool_description: The agent provides advanced business analytics across products,
    channels, regions, and customer segments. It identifies top revenue-generating
    categories by channel, analyzes margin variations, sales mix, and promotion effectiveness,
    and assesses competitor impact. It highlights underperforming high-potential categories,
    price sensitivity, ROI relative to CapEx investments, and resilient segments under
    competitive pressure. Additionally, it evaluates channel trends, investment alignment,
    and strategies to optimize revenue, margin, and sustainable growth across category-region-channel
    combinations.
  tool_name: genie_rexel_query_tool
llm_endpoint: databricks-claude-sonnet-4
uc_catalog_schemas:
- catalog: mc
  schema: rexel
'''

config_data = yaml.safe_load(CONFIG_YAML)

from typing import Any, Generator, Optional, Sequence, Union, Dict, List
from dataclasses import dataclass

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from databricks.sdk import WorkspaceClient
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from databricks_langchain.genie import GenieAgent
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

@dataclass
class GenieAgentConfig:
    """Configuration for a Genie agent"""
    space_id: str
    agent_name: str
    description: str
    tool_name: str
    tool_description: str

@dataclass
class AgentConfig:
    """Main configuration for the agent"""
    genie_agents: List[GenieAgentConfig]
    llm_endpoint: str = "databricks-claude-sonnet-4"
    uc_tool_names: List[str] = None
    
    def __post_init__(self):
        if self.uc_tool_names is None:
            self.uc_tool_names = []

def extract_genie_response(response) -> str:
    """
    Get the human-readable answer text out of the object returned by
    GenieAgent.invoke().
    """
    # Case 1 – Genie’s documented schema (dict with AIMessage objects)
    if isinstance(response, dict) and response.get("messages"):
        last_msg = response["messages"][-1]
        if hasattr(last_msg, "content"):
            return last_msg.content
    # Case 2 – already an AIMessage
    if hasattr(response, "content"):
        return response.content
    # Fallback – stringify whatever we got
    return str(response)

def create_genie_tool(genie_config: GenieAgentConfig) -> BaseTool:
    """Create a Genie tool from configuration"""
    genie_agent = GenieAgent(
        genie_space_id=genie_config.space_id,
        genie_agent_name=genie_config.agent_name,
        description=genie_config.description
    )
    
    @tool
    def genie_query(query: str) -> str:
        """Query the Genie agent"""
        response = genie_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return extract_genie_response(response)
    
    # Update the tool's metadata
    genie_query.name = genie_config.tool_name
    genie_query.description = genie_config.tool_description
    
    return genie_query

def create_genie_tools(genie_configs: List[GenieAgentConfig]) -> List[BaseTool]:
    """Create multiple Genie tools from configurations"""
    tools = []
    for config in genie_configs:
        tools.append(create_genie_tool(config))
    return tools

def parse_markdown_table(markdown_str: str) -> pd.DataFrame:
    """Convert Genie markdown table into pandas DataFrame."""
    try:
        # Extract table section
        table_match = re.findall(r"\|.*\|", markdown_str)
        if table_match:
            table_str = "\n".join(table_match)
            df = pd.read_csv(io.StringIO(table_str), sep="|").dropna(axis=1, how="all")
            df = df.applymap(lambda x: str(x).strip())
            # Drop index col if present
            if df.columns[0].strip() == "":
                df = df.drop(df.columns[0], axis=1)
            return df
    except Exception as e:
        print(f"Parsing failed: {e}")
    return None



def create_agent_with_config(config: AgentConfig) -> CompiledGraph:
    """Create the agent with external configuration"""
    
    # Create LLM
    llm = ChatDatabricks(endpoint=config.llm_endpoint)
    
    # Create system prompt
    system_prompt = """
You are an expert data analyst. If the intent uses genie agent, analyze the query results and extract all numerical data, then recommend the best chart type.

Please respond with ONLY a valid JSON object (no markdown formatting, no code blocks, no additional text) containing:

1. "hasData": boolean - whether numerical data was found
2. "extractedData": array of objects with "label" and "value" properties
3. "dataType": string - type of data (percentage, currency, quantity, ratio, etc.)
4. "chartType": string - recommended chart type (bar, line, pie, doughnut, radar, scatter)
5. "title": string - suggested chart title
6. "insights": string - FORMATTED AS MARKDOWN : detailed analysis of the data patterns +  As a business strategist what you would advice to do based on the data to achieve business efficency - please use MARKDOWN with three hasthags for titles and so on, format the content in a nice way for clarity and neat professional rendering
7. "xAxisLabel": string - label for x-axis (if applicable)
8. "yAxisLabel": string - label for y-axis (if applicable)

Rules:
- Extract ALL numerical values with their context
- For percentages, keep as numbers (e.g., 8.5 not 0.085)
- For currencies, extract just the number
- For time series, preserve the time labels
- Choose chart types based on data relationships:
  - Bar/Column: comparisons, categories
  - Line: trends over time, continuous data
  - Pie/Doughnut: parts of a whole, percentages that sum to 100%
  - Radar: multi-dimensional comparisons
  - Scatter: correlations, relationships

IMPORTANT: Respond with ONLY the JSON object, no markdown code blocks, no additional formatting.
"""
    
    # Create tools
    tools = []
    
    # Add Genie tools
    genie_tools = create_genie_tools(config.genie_agents)
    tools.extend(genie_tools)
    
    # Add Unity Catalog tools
    if config.uc_tool_names:
        uc_toolkit = UCFunctionToolkit(function_names=config.uc_tool_names)
        tools.extend(uc_toolkit.tools)
    
    return create_tool_calling_agent(llm, tools, system_prompt)


#####################
## Define agent logic
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )

DEFAULT_CONFIG = AgentConfig(
    genie_agents=[GenieAgentConfig(**agent) for agent in config_data['genie_agents']],
    llm_endpoint=config_data['llm_endpoint'],
    uc_tool_names=[]
    #uc_tool_names=config_data['uc_tool_names']
)

def create_agent(genie_space_ids: Optional[List[str]] = None, 
                genie_agent_names: Optional[List[str]] = None,
                genie_descriptions: Optional[List[str]] = None,
                tool_names: Optional[List[str]] = None,
                tool_descriptions: Optional[List[str]] = None,
                llm_endpoint: Optional[str] = None,
                uc_tool_names: Optional[List[str]] = None) -> LangGraphChatAgent:
    """
    Create the agent with external parameters.
    
    Args:
        genie_space_ids: List of Genie space IDs
        genie_agent_names: List of Genie agent names
        genie_descriptions: List of Genie agent descriptions
        tool_names: List of tool names
        tool_descriptions: List of tool descriptions
        llm_endpoint: LLM endpoint name
        uc_tool_names: List of Unity Catalog tool names
    
    Returns:
        LangGraphChatAgent instance
    """
    
    # Use provided parameters or defaults
    if genie_space_ids is not None:
        # Create custom configuration
        genie_agents = []
        for i, space_id in enumerate(genie_space_ids):
            agent_name = genie_agent_names[i] if genie_agent_names and i < len(genie_agent_names) else f"Genie_{i}"
            description = genie_descriptions[i] if genie_descriptions and i < len(genie_descriptions) else f"Genie agent {i}"
            tool_name = tool_names[i] if tool_names and i < len(tool_names) else f"genie_query_{i}"
            tool_description = tool_descriptions[i] if tool_descriptions and i < len(tool_descriptions) else f"Query genie agent {i}"
            
            genie_agents.append(GenieAgentConfig(
                space_id=space_id,
                agent_name=agent_name,
                description=description,
                tool_name=tool_name,
                tool_description=tool_description
            ))
        
        config = AgentConfig(
            genie_agents=genie_agents,
            llm_endpoint=llm_endpoint or DEFAULT_CONFIG.llm_endpoint,
            uc_tool_names=uc_tool_names or DEFAULT_CONFIG.uc_tool_names
        )
    else:
        # Use default configuration
        config = DEFAULT_CONFIG
    
    # Create the agent
    agent = create_agent_with_config(config)
    return LangGraphChatAgent(agent)

# Create the default agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
AGENT = create_agent()
mlflow.models.set_model(AGENT)


