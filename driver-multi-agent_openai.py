# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC     %pip install -U typing_extensions>=4.6.0 backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from genie_agent_with_config_openai5 import AGENT


# COMMAND ----------



query = "list products families"

response = AGENT.predict({
    "input": [
        {"role": "user", "content": f"{query}"}  # 
    ]
})


# COMMAND ----------

from multi_agent_with_config_openai import AGENT

# COMMAND ----------

query = "information about Backup and restore for DynamoDB."

response = AGENT.predict({
    "input": [
        {"role": "user", "content": f"{query}"}  # 
    ]
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

import yaml
import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksTable,
)
from pkg_resources import get_distribution
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from databricks.sdk import WorkspaceClient

# Load YAML
with open("./genie-agents-config.yaml") as f:
    config = yaml.safe_load(f)

client = WorkspaceClient()

# Retrieve LLM endpoint from YAML
llm_endpoint = config["llm_endpoint"]

resources = [DatabricksServingEndpoint(endpoint_name=llm_endpoint)]

# Add Genie spaces dynamically
for agent in config.get("genie_agents", []):
    genie_space_id = agent["space_id"]
    resources.append(DatabricksGenieSpace(genie_space_id=genie_space_id))
    warehouse_id = client.genie.get_space(genie_space_id).warehouse_id
    resources.append(DatabricksSQLWarehouse(warehouse_id=warehouse_id))

# Add tables dynamically from catalog/schema
for cs in config.get("uc_catalog_schemas", []):
    catalog = cs["catalog"]
    schema = cs["schema"]
    tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").drop("isTemporary")
    for t in tables_df.collect():
        resources.append(DatabricksTable(table_name=f"{catalog}.{schema}.{t['tableName']}"))



# COMMAND ----------

resources

# COMMAND ----------

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is an LLM agent?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="genie_agent_with_config.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Où la marge reste-t-elle la plus forte malgré une pression concurrentielle élevée (catégories/régions résilientes)"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "mc"
schema = "rexel"
model_name = "rexel_genie_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})