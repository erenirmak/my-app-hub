from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    mlops = "https://github.com/graviraja/MLOps-Basics"

    applications = [
        {'name': 'İstanbul Sanayi Odası - Agent Test', 'url': 'http://10.0.0.1:8501'},
        {'name': 'Eksim Holding - Agent Test', 'url': 'http://10.0.0.1:8502'},
        {'name': 'Airflow', 'url': 'http://10.0.0.1:8080/home'},
        {'name': 'Jupyter Server', 'url': 'http://10.0.0.1:8888/tree'},
    ]
    
    collapsibles = {
    'AI Agents': [
        {
            'Agent Frameworks': [
                {'name': 'LangChain / LangGraph', 'url': 'https://github.com/langchain-ai/langchain'},
                {'name': 'Langgraph Supervisor (multi-agent)', 'url': 'https://github.com/langchain-ai/langgraph-supervisor-py'},
                {'name': 'Langflow', 'url': 'https://github.com/langflow-ai/langflow'},
                {'name': 'LlamaIndex', 'url': 'https://github.com/jerryjliu/llama_index'},
                {'name': 'smolagents', 'url': 'https://github.com/huggingface/smolagents'},
                {'name': 'AgnoAGI', 'url': 'https://github.com/agno-agi/agno/'},
                {'name': 'CrewAI', 'url': 'https://github.com/crewAIInc/crewAI'},
                {'name': 'PydanticAI', 'url': 'https://github.com/pydantic/pydantic-ai'},
                {'name': 'swarms', 'url': 'https://github.com/kyegomez/swarms'},
                {'name': 'Atomic Agents', 'url': 'https://github.com/BrainBlend-AI/atomic-agents'},
                {'name': 'QuantaLogic', 'url': 'https://github.com/quantalogic/quantalogic'},
                {'name': 'AutoGen (Microsoft)', 'url': 'https://github.com/microsoft/autogen'},
                {'name': 'AG2 (formerly AutoGen)', 'url': 'https://github.com/ag2ai/ag2'},
                {'name': 'Agents SDK (OpenAI)', 'url': 'https://github.com/openai/openai-agents-python'},
                {'name': 'SuperAGI', 'url': 'https://github.com/TransformerOptimus/SuperAGI'},
                {'name': 'Intellagent', 'url': 'https://github.com/plurai-ai/intellagent'},
                {'name': 'Qwen-Agent', 'url': 'https://github.com/QwenLM/Qwen-Agent'},
                {'name': 'AutoAgent', 'url': 'https://github.com/HKUDS/AutoAgent'},
                {'name': 'CAMEL', 'url': 'https://github.com/camel-ai/camel'},
                {'name': 'Mistral (commercial)', 'url': 'https://mistral.ai/'}
            ]
        },
        {
            'RAG Frameworks': [
                {'name': 'DSPy (Stanford)', 'url': 'https://github.com/stanfordnlp/dspy'},
                {'name': 'GraphRAG (Microsoft)', 'url': 'https://github.com/microsoft/GraphRAG'},
                {'name': 'LLM-Graph-Builder (Neo4j)', 'url': 'https://github.com/neo4j-labs/llm-graph-builder'},
                {'name': 'GroundX (EyeLevel)', 'url': 'https://github.com/eyelevelai/groundx-on-prem'},
                {'name': 'Byaldi (multimodal retrieval on top of LangChain)', 'url': 'https://github.com/AnswerDotAI/byaldi'},
            ]
        },
        {
            'Memory': [
                {'name': 'langmem', 'url': 'https://github.com/langchain-ai/langmem'},
                {'name': 'A-MEM', 'url': 'https://github.com/agiresearch/A-mem'},
            ]
        },
        {
            'Connector': [
                {'name': 'Model Context Protocol (MCP)', 'url': 'https://github.com/modelcontextprotocol'},
                {'name': 'Llama-MCP', 'url': 'https://github.com/llama-mcp'},
                {'name': 'GenAI Toolbox (Google - Server to integrate with databases)', 'url': 'https://github.com/googleapis/genai-toolbox'},
            ]
        },
        {
            'Structured Outputs': [
                {'name': 'Instructor', 'url': 'https://github.com/jxnl/instructor'},
                {'name': 'Outlines', 'url': 'https://github.com/outlines-dev/outlines'},
                {'name': 'lm-format-enforcer', 'url': 'https://github.com/noamgat/lm-format-enforcer'},
                {'name': 'xgrammar', 'url': 'https://github.com/mlc-ai/xgrammar'},
                ]
        },
        {
            'Tracing': [
                {'name': 'LangSmith (commercial)', 'url': 'https://github.com/langchain-ai/langsmith-sdk'},
                {'name': 'Langfuse', 'url': 'https://github.com/langfuse/langfuse'},
                {'name': 'Pydantic Logfire', 'url': 'https://github.com/pydantic/logfire'},
                {'name': 'AgentOps', 'url': 'https://github.com/AgentOps-AI/agentops'},
                {'name': 'Phoenix', 'url': 'https://github.com/Arize-ai/phoenix'},
                {'name': 'Opik', 'url': 'https://github.com/comet-ml/opik'},
                {'name': 'Datadog (commercial)', 'url': 'https://www.datadoghq.com/'},
                {'name': 'Braintrust (commercial)', 'url': 'https://www.braintrust.dev/'},
                {'name': 'Keywords (commercial)', 'url': 'https://www.keywordsai.co/'},
            ]
        },
        {
            'Evaluation': [
                {'name': 'Ragas', 'url': 'https://github.com/explodinggradients/ragas'},
                {'name': 'TruLens', 'url': 'https://github.com/truera/trulens'},
                {'name': 'Opik', 'url': 'https://github.com/comet-ml/opik'},
                {'name': 'OpenEvals', 'url': 'https://github.com/langchain-ai/openevals'},
                {'name': 'Scorecard (commercial)', 'url': 'https://www.scorecard.io/'} 
            ]
        },
        {
            'UI': [
                {'name': 'Agent Chat UI(LangChain)', 'url': 'https://github.com/langchain-ai/agent-chat-ui'},
                {'name': 'Text Generation Inference (TGI)', 'url': 'https://github.com/huggingface/text-generation-inference'},
                {'name': 'Streamlit', 'url': 'https://github.com/streamlit/streamlit'},
                {'name': 'Gradio', 'url': 'https://github.com/gradio-app/gradio'},
                {'name': 'Chainlit', 'url': 'https://github.com/Chainlit/chainlit'},
                {'name': 'Panel', 'url': 'https://github.com/holoviz/panel'},
                {'name': 'NiceGUI', 'url': 'https://github.com/zauberzeug/nicegui'},
            ]
        },
        {
            'Quantization': [
                {'name': 'LlamaCPP', 'url': 'https://github.com/ggml-org/llama.cpp'},
                {'name': 'AWQ', 'url': 'https://github.com/mit-han-lab/llm-awq'},
                {'name': 'AutoAWQ', 'url': 'https://github.com/casper-hansen/AutoAWQ'},
                {'name': 'bitsandbytes', 'url': 'https://github.com/bitsandbytes-foundation/bitsandbytes'},
                {'name': 'GPTQ', 'url': 'https://github.com/IST-DASLab/gptq'},
                {'name': 'AutoGPTQ', 'url': 'https://github.com/AutoGPTQ/AutoGPTQ'},
            ]
        },
        {
            'Deployment': [
                {'name': 'vLLM', 'url': 'https://github.com/vllm-project/vllm'},
                {'name': 'Ollama', 'url': 'https://github.com/ollama/ollama'},
                {'name': 'AIBrix (deployment on Kubernetes environment - vLLM extension)', 'url': 'https://github.com/vllm-project/aibrix'},
                {'name': 'LoRAX (Predibase - server)', 'url': 'https://github.com/predibase/lorax'},
                {'name': 'LlamaCPP - llama-server', 'url': 'https://github.com/ggml-org/llama.cpp'},
            ]
        },
        {
            'Environment': [
                {'name': 'E2B', 'url': ''},
                {'name': 'Docker', 'url': ''},
            ]
        },
        {
            'Discovery': [
                {'name': 'IBM BeeAI', 'url': 'https://github.com/i-am-bee/beeai'},
            ]
        },
        {
            'Development': [
                {'name': 'PyTorch (Meta)', 'url': 'https://github.com/pytorch/pytorch'},
                {'name': 'Tensorflow (Google)', 'url': 'https://github.com/tensorflow/tensorflow'},
                {'name': 'MLX (Apple)', 'url': 'https://github.com/ml-explore/mlx'},
                {'name': 'JAX (Google - TPUs)', 'url': 'https://github.com/jax-ml/jax'},
                {'name': 'Triton (OpenAI)', 'url': 'https://github.com/triton-lang/triton'},
                {'name': 'Ludwig (Predibase)', 'url': 'https://github.com/ludwig-ai/ludwig'},
            ]
        },
        {
            'Experimentation': [
                {'name': 'Weights & Biases', 'url': 'https://github.com/wandb/wandb'},
                {'name': 'Comet', 'url': 'https://www.comet.ml/'},
                {'name': 'Neptune (commercial)', 'url': 'https://neptune.ai/'},
                {'name': 'MLflow', 'url': 'https://github.com/mlflow/mlflow'},
                {'name': 'DVC', 'url': 'https://github.com/iterative/dvc.org'},
                {'name': 'Polyaxon', 'url': 'https://github.com/polyaxon/polyaxon'},
                {'name': 'Sacred', 'url': 'https://github.com/IDSIA/sacred'},
                {'name': 'Sacredboard', 'url': 'https://github.com/chovanecm/sacredboard'},
                {'name': 'Omniboard', 'url': 'https://github.com/vivekratnavel/omniboard'},
                {'name': 'TensorBoard', 'url': 'https://github.com/tensorflow/tensorboard'},
                {'name': 'Visdom', 'url': 'https://github.com/fossasia/visdom'},
            ]
        }
    ],
    'Commercial Agentic Products': [
        {
            'agentic_workflows': [
                {'name': 'UiPath', 'url': 'https://www.uipath.com/'},
                {'name': 'n8n', 'url': 'https://n8n.io/'},
                {'name': 'Automation Anywhere', 'url': 'https://www.automationanywhere.com/'},
                {'name': 'Blue Prism', 'url': 'https://www.blueprism.com/'},
                {'name': 'Orkes', 'url': 'https://orkes.io'},
                {'name': 'aiXplain', 'url': 'https://aixplain.com/'},
                {'name': 'Smyth|OS', 'url': 'https://smythos.com/'},
            ]
        },
        {
            'e-mail_&_chat_services': [
                {'name': 'Spike', 'url': 'https://www.spikenow.com/'},
                {'name': 'Canary', 'url': 'https://canarymail.io/'},
                {'name': 'conversica', 'url': 'https://www.conversica.com/'},
            ]
        }
    ],
    'Databases': [
        {
            'Relational': [
                {'name': 'PostgreSQL', 'url': 'https://www.postgresql.org/'},
                {'name': 'MySQL', 'url': 'https://www.mysql.com/'},
                {'name': 'Microsoft SQL Server', 'url': 'https://www.microsoft.com/en-us/sql-server/'},
                {'name': 'Oracle Database', 'url': 'https://www.oracle.com/database/'},
                {'name': 'IBM DB2', 'url': 'https://www.ibm.com/products/db2-database'},
                {'name': 'MariaDB', 'url': 'https://mariadb.org/'},
                {'name': 'CockroachDB', 'url': 'https://www.cockroachlabs.com/'},
            ]
        },
        {
            'NoSQL': [
                {'name': 'MongoDB', 'url': 'https://www.mongodb.com/'},
                {'name': 'Couchbase', 'url': 'https://www.couchbase.com/'},
                {'name': 'DataStax (Cassandra)', 'url': 'https://www.datastax.com/'},
                {'name': 'Cassandra', 'url': 'https://cassandra.apache.org/'},
                {'name': 'HBase', 'url': 'https://hbase.apache.org/'},
                {'name': 'CouchDB', 'url': 'https://couchdb.apache.org/'},
            ]
        },
        {
            'Key-Value Stores': [
                {'name': 'Redis', 'url': 'https://redis.io/'},
                {'name': 'Amazon DynamoDB', 'url': 'https://aws.amazon.com/dynamodb/'},
                {'name': 'Riak', 'url': 'https://basho.com/riak/'},
                {'name': 'Memcached', 'url': 'https://memcached.org/'},
            ]
        },
        {
            'Analytics': [
                {'name': 'MicroStrategy', 'url': 'https://www.microstrategy.com/'},
                {'name': 'SQL Server Analysis Services', 'url': 'https://learn.microsoft.com/en-us/sql/analysis-services/analysis-services-overview?view=as-ssdt-2019'},
                {'name': 'Teradata', 'url': 'https://www.teradata.com/'},
                {'name': 'ClickHouse', 'url': 'https://clickhouse.com/'},
            ]
        },
        {
            'Object Storage': [
                {'name': 'S3', 'url': 'https://aws.amazon.com/s3/'},
                {'name': 'MinIO', 'url': 'https://min.io/'},
            ]
        },
        {
            'VectorDB': [
                {'name': 'Weaviate', 'url': 'https://github.com/weaviate/weaviate'},
                {'name': 'Qdrant', 'url': 'https://github.com/qdrant/qdrant'},
                {'name': 'pgvector (PostgreSQL extension)', 'url': 'https://github.com/pgvector/pgvector'},
                {'name': 'Chroma', 'url': 'https://github.com/chroma-core/chroma'},
                {'name': 'FAISS', 'url': 'https://github.com/facebookresearch/faiss'},
                {'name': 'LanceDB', 'url': 'https://github.com/lancedb/lancedb'},
                {'name': 'Pinecone', 'url': 'https://www.pinecone.io/'},
                {'name': 'Milvus', 'url': 'https://github.com/milvus-io/milvus'},
                {'name': 'Vald', 'url': 'https://vald.vdaas.org/'},
            ]
        },
        {
            'Multi-Model': [
                {'name': 'ArangoDB', 'url': 'https://www.arangodb.com/'},
                {'name': 'SurrealDB', 'url': 'https://surrealdb.com/'},
            ]
        },
        {
            'GraphDB': [
                {'name': 'Neo4j', 'url': 'https://neo4j.com/'},
                {'name': 'Apache AGE (PostgreSQL extension)', 'url': 'https://github.com/apache/age'},
                {'name': 'AllegroGraph', 'url': 'https://franz.com/agraph/allegrograph/'},
                {'name': 'RedisGraph (Redis Module)', 'url': 'https://redis.io/docs/stack/graph/'},
            ]
        },
        {
            'Search': [
                {'name': 'ElasticSearch', 'url': 'https://www.elastic.co/elasticsearch/'},
                {'name': 'Opensearch', 'url': 'https://opensearch.org/'},
                {'name': 'Vespa', 'url': 'https://vespa.ai/'},
            ]
        },
        {
            'Time-Series': [
                {'name': 'TimescaleDB (PostgreSQL extension)', 'url': 'https://www.timescale.com/'},
                {'name': 'InfluxDB', 'url': 'https://www.influxdata.com/'},
            ]
        },
        {
            'Embedded': [
                {'name': 'SQLite', 'url': 'https://www.sqlite.org/'},
                {'name': 'DuckDB', 'url': 'https://duckdb.org/'},
                {'name': 'LevelDB', 'url': 'https://github.com/google/leveldb'},
                {'name': 'RocksDB', 'url': 'https://rocksdb.org/'},
            ]
        },
        {
            'Time-Travel': [
                {'name': 'Dolt', 'url': 'https://www.dolthub.com/'},
                {'name': 'Delta Lake (Databricks)', 'url': 'https://delta.io/'},
                {'name': 'Temporal Table (PostgreSQL extension)', 'url': ''},
            ]
        },
        {
            'Data Lineage': [
                {'name': 'DataHub', 'url': 'https://datahubproject.io/'},
                {'name': 'Amundsen', 'url': 'https://www.amundsen.io/'},
                {'name': 'Data Catalog (Google)', 'url': 'https://cloud.google.com/data-catalog'},
                {'name': 'Data Catalog (AWS)', 'url': 'https://aws.amazon.com/datacatalog/'},
            ]
        },
        {
            'Data Quality': [
                {'name': 'Great Expectations', 'url': 'https://greatexpectations.io/'},
                {'name': 'DataRobot', 'url': 'https://www.datarobot.com/'},
                {'name': 'Truera', 'url': 'https://www.truera.com/'},
            ]
        },
    ],
    'Big Data Tools': [
        {
            'Schedulers': [
                {'name': 'Airflow', 'url': 'https://airflow.apache.org/'},
                {'name': 'Kestra', 'url': 'https://kestra.io/'},
                {'name': 'Luigi', 'url': 'https://luigi.readthedocs.io/en/stable/'},
                {'name': 'Prefect', 'url': 'https://www.prefect.io/'},
                {'name': 'Digdag', 'url': 'https://www.digdag.io/'},
                {'name': 'Azkaban', 'url': 'https://azkaban.github.io/'},
                {'name': 'Oozie', 'url': 'https://oozie.apache.org/'},
                {'name': 'Kronos', 'url': ''},
                ],
            'big_data_tools': [
                {'name': 'Hadoop', 'url': 'https://hadoop.apache.org/'},
                {'name': 'Spark', 'url': 'https://spark.apache.org/'},
                {'name': 'Flink', 'url': 'https://flink.apache.org/'},
                {'name': 'Hive', 'url': 'https://hive.apache.org/'},
                {'name': 'Presto', 'url': 'https://prestodb.io/'},
                {'name': 'Druid', 'url': 'https://druid.apache.org/'},
                {'name': 'Kafka', 'url': 'https://kafka.apache.org/'},
                {'name': 'Cassandra', 'url': 'http://cassandra.apache.org/'},
                {'name': 'HBase', 'url': 'https://hbase.apache.org/'},
                {'name': 'Impala', 'url': 'https://impala.apache.org/'},
                {'name': 'Pulsar', 'url': 'https://pulsar.apache.org/'},
                {'name': 'Pinot', 'url': 'https://pinot.apache.org/'},
                {'name': 'Flink SQL', 'url': 'https://ci.apache.org/projects/flink/flink-docs-stable/dev/table/sql/queries/queries.html'},
                {'name': 'Hive SQL', 'url': 'https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select'},
                {'name': 'Presto SQL', 'url': 'https://prestodb.io/docs/current/sql/'},
                {'name': 'Druid SQL', 'url': 'https://druid.apache.org/docs/latest/querying/sql.html'},
                {'name': 'Kafka SQL', 'url': 'https://docs.confluent.io/platform/current/ksql/docs/index.html'},
                {'name': 'Cassandra SQL', 'url': 'https://cassandra.apache.org/doc/latest/cql/'},
                {'name': 'HBase SQL', 'url': 'https://hbase.apache.org/book.html#sql'},
                {'name': 'Impala SQL', 'url': 'https://impala.apache.org/docs/build/html/topics/impala_sql_reference.html'},
                {'name': 'Pulsar SQL', 'url': 'https://pulsar.apache.org/docs/en/sql-overview/'},
                {'name': 'Pinot SQL', 'url': 'https://pinot.apache.org/pinot-docs/query-guide/sql'},
            ]
        },
        {
            'Environment': [
                {'name': 'Docker', 'url': 'https://www.docker.com/'},
                {'name': 'Kubernetes', 'url': ''},
                {'name': 'Kubeflow', 'url': ''},
                {'name': 'OpenShift', 'url': ''},
                {'name': 'Nomad', 'url': ''},
                {'name': 'Rancher', 'url': ''},
                {'name': 'Vagrant', 'url': ''},
                {'name': 'VirtualBox', 'url': ''},
                {'name': 'VMware', 'url': ''},
                {'name': 'WSL', 'url': ''},
                {'name': 'WSL2', 'url': ''},
            ]
        }
    ]
}

    return render_template('index.html', applications=applications, collapsibles=collapsibles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True) # debug true for development