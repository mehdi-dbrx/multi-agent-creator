"""
Tool-calling Agent with GenieAgent using OpenAI SDK for Databricks

Installation Requirements:
    %pip install -U typing_extensions>=4.6.0
    %pip install -U backoff databricks-openai uv databricks-agents mlflow-skinny[databricks] pyyaml
"""

import json
import yaml
from typing import Any, Callable, Generator, Optional, List, Dict
from dataclasses import dataclass
from uuid import uuid4
import warnings
import re
import io

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client
import pandas as pd

# Import our GenieAgent for OpenAI
from genie_agent_openai import GenieAgent

mlflow.openai.autolog()

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

############################################
# Configuration Classes
############################################

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
    uc_tool_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.uc_tool_names is None:
            self.uc_tool_names = []

############################################
# Utility Functions
############################################

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

############################################
# Tool Info Class
############################################

class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    """
    name: str
    spec: dict
    exec_fn: Callable

############################################
# Tool Creation Functions
############################################

def create_genie_tool_info(genie_config: GenieAgentConfig) -> ToolInfo:
    """Create a Genie tool from configuration"""
    # Don't pass a client - let GenieAgent create its own with proper auth
    genie_agent = GenieAgent(
        genie_space_id=genie_config.space_id,
        genie_agent_name=genie_config.agent_name,
        description=genie_config.description
        # No client parameter!
    )
    
    # Get the tool spec and update with our custom name and description
    tool_spec = genie_agent.tool.copy()
    tool_spec["function"]["name"] = genie_config.tool_name
    tool_spec["function"]["description"] = genie_config.tool_description
    
    # Remove strict if present
    tool_spec["function"].pop("strict", None)
    
    return ToolInfo(
        name=genie_config.tool_name,
        spec=tool_spec,
        exec_fn=genie_agent.execute
    )

def create_genie_tools(genie_configs: List[GenieAgentConfig]) -> List[ToolInfo]:
    """Create multiple Genie tools from configurations"""
    tools = []
    for config in genie_configs:
        tools.append(create_genie_tool_info(config))
    return tools

def create_uc_tool_info(tool_spec: dict, uc_function_client) -> ToolInfo:
    """Create a Unity Catalog tool"""
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")
    
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn)

############################################
# Agent Class
############################################

SYSTEM_PROMPT = """
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

class ToolCallingAgent(ResponsesAgent):
    """
    Tool-calling Agent with GenieAgent support
    """

    def __init__(self, llm_endpoint: str, tools: List[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prep_msgs_for_cc_llm(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from self.output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)

############################################
# Agent Creation Functions
############################################

def create_agent_with_config(config: AgentConfig) -> ToolCallingAgent:
    """Create the agent with external configuration"""
    
    tools = []
    
    # Add Genie tools (each creates its own client with proper auth)
    genie_tools = create_genie_tools(config.genie_agents)
    tools.extend(genie_tools)
    
    # Add Unity Catalog tools
    if config.uc_tool_names:
        uc_toolkit = UCFunctionToolkit(function_names=config.uc_tool_names)
        uc_function_client = get_uc_function_client()
        for tool_spec in uc_toolkit.tools:
            tools.append(create_uc_tool_info(tool_spec, uc_function_client))
    
    return ToolCallingAgent(llm_endpoint=config.llm_endpoint, tools=tools)

############################################
# Default Configuration
############################################

DEFAULT_CONFIG = AgentConfig(
    genie_agents=[GenieAgentConfig(**agent) for agent in config_data['genie_agents']],
    llm_endpoint=config_data['llm_endpoint'],
    uc_tool_names=[]
)

def create_agent(genie_space_ids: Optional[List[str]] = None, 
                genie_agent_names: Optional[List[str]] = None,
                genie_descriptions: Optional[List[str]] = None,
                tool_names: Optional[List[str]] = None,
                tool_descriptions: Optional[List[str]] = None,
                llm_endpoint: Optional[str] = None,
                uc_tool_names: Optional[List[str]] = None) -> ToolCallingAgent:
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
        ToolCallingAgent instance
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
    return create_agent_with_config(config)

# Create the default agent object
AGENT = create_agent()
mlflow.models.set_model(AGENT)


# Example usage (for testing in Databricks):
# To test this agent, use the following code:
#
# from genie_agent_with_config_openai import AGENT
#
# # Correct input format - must use "input" key, not "messages"
# response = AGENT.predict({
#     "input": [
#         {"role": "user", "content": "What are the top revenue-generating categories?"}
#     ]
# })
# print(response)

