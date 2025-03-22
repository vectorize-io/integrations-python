# langchain-vectorize

This package contains the LangChain integrations for using Vectorize.

## Installation and Setup

Installation of this package:

```bash
pip install langchain-vectorize
```

## Integrations overview

### Retriever

See the [LangChain Retriever documentation](https://python.langchain.com/docs/concepts/retrievers/) for more information.
```python
from langchain_vectorize import VectorizeRetriever

retriever = VectorizeRetriever(
    api_token="...",
    organization="...",
    pipeline_id="...",
)
retriever.invoke("query")
```
See an example notebook [here](https://github.com/vectorize-io/integrations-python/tree/main/notebooks/langchain_retriever.ipynb).