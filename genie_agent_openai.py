"""
GenieAgent for OpenAI SDK (from https://github.com/mehdi-dbrx/databricks-openai-genie-agent)
"""
import json
import time
from typing import Optional
from pydantic import BaseModel, Field
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieGetMessageQueryResultResponse
import mlflow
from mlflow.entities import SpanType


class GenieAgentInput(BaseModel):
    """Input schema for GenieAgent"""
    query: str = Field(description="The question to ask the Genie agent")


class GenieAgentOutput(BaseModel):
    """Output schema for GenieAgent"""
    result: str = Field(description="The answer from Genie")
    sql_query: Optional[str] = Field(description="SQL query used (if available)", default=None)
    reasoning: Optional[str] = Field(description="Reasoning from Genie", default=None)


class GenieAgent:
    """
    GenieAgent for OpenAI SDK integration
    
    Usage:
        genie_agent = GenieAgent(
            genie_space_id="your-space-id",
            genie_agent_name="MyGenie",
            description="Genie for analytics"
        )
        
        # Get the tool spec for OpenAI
        tool_spec = genie_agent.tool
        
        # Execute a query
        result = genie_agent.execute(query="What is the revenue?")
    """
    
    def __init__(
        self,
        genie_space_id: str,
        genie_agent_name: str = "Genie",
        description: str = "A Genie agent for data queries",
        client: Optional[WorkspaceClient] = None,
    ):
        """
        Initialize GenieAgent
        
        Args:
            genie_space_id: The Genie space ID
            genie_agent_name: Name for the agent
            description: Description of what the agent does
            client: Optional WorkspaceClient (creates one if not provided)
        """
        self.genie_space_id = genie_space_id
        self.genie_agent_name = genie_agent_name
        self.description = description
        self.client = client or WorkspaceClient()
        self._conversation_id = None
    
    def _create_tool(self) -> dict:
        """Create OpenAI function calling tool specification"""
        return {
            "type": "function",
            "function": {
                "name": f"genie_{self.genie_agent_name.lower().replace(' ', '_')}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question to ask the Genie agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    @property
    def tool(self) -> dict:
        """Get the OpenAI tool specification"""
        return self._create_tool()
    
    @mlflow.trace(span_type=SpanType.TOOL, name="genie_query")
    def execute(self, query: str, conversation_id: Optional[str] = None) -> str:
        """
        Execute a query against the Genie space
        
        Args:
            query: The question to ask
            conversation_id: Optional conversation ID for context
            
        Returns:
            JSON string with result, sql_query, and reasoning
        """
        try:
            # Use provided conversation_id or the instance one
            conv_id = conversation_id or self._conversation_id
            
            # Determine message_id and conversation_id based on whether conversation exists
            if conv_id is None:
                # Start a new conversation
                start_response = self.client.genie.start_conversation(
                    space_id=self.genie_space_id,
                    content=query
                )
                conversation_id = start_response.conversation_id
                message_id = start_response.message_id
                self._conversation_id = conversation_id
            else:
                # Use existing conversation
                msg_wait = self.client.genie.create_message(
                    space_id=self.genie_space_id,
                    conversation_id=conv_id,
                    content=query
                )
                # Wait for the async response to complete
                msg_response = msg_wait.result()
                
                conversation_id = conv_id
                # Get message_id from the response
                message_id = msg_response.id if hasattr(msg_response, 'id') else msg_response.message_id
            
            # Wait for Genie to process the query
            time.sleep(2)
            
            # Get the query result
            result_response: GenieGetMessageQueryResultResponse = (
                self.client.genie.get_message_query_result(
                    space_id=self.genie_space_id,
                    message_id=message_id,
                    conversation_id=conversation_id
                )
            )
            
            # Extract information - need to fetch actual chunk data
            result_text = "No result available"
            if hasattr(result_response, 'statement_response') and result_response.statement_response:
                stmt_resp = result_response.statement_response
                
                # Check if we have a statement_id to fetch chunk data
                if hasattr(stmt_resp, 'statement_id') and stmt_resp.statement_id:
                    statement_id = stmt_resp.statement_id
                    
                    # Fetch the actual result chunk
                    try:
                        chunk_result = self.client.statement_execution.get_statement_result_chunk_n(
                            statement_id=statement_id,
                            chunk_index=0
                        )
                        
                        # Extract data_array from chunk
                        if hasattr(chunk_result, 'data_array') and chunk_result.data_array:
                            result_text = str(chunk_result.data_array)
                        else:
                            result_text = str(chunk_result)
                    except Exception as e:
                        result_text = f"Error fetching result data: {e}"
            
            # Get SQL query
            sql_query = None
            if hasattr(result_response, 'statement_response') and result_response.statement_response:
                if hasattr(result_response.statement_response, 'statement') and result_response.statement_response.statement:
                    sql_query = result_response.statement_response.statement.statement_text
            
            output = GenieAgentOutput(
                result=result_text,
                sql_query=sql_query,
                reasoning=getattr(result_response, 'description', None)
            )
            
            return output.model_dump_json()
            
        except Exception as e:
            error_output = GenieAgentOutput(
                result=f"Error querying Genie: {str(e)}",
                sql_query=None,
                reasoning=None
            )
            return error_output.model_dump_json()

