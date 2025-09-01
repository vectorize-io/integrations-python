from typing import Literal

from langchain_vectorize.retrievers import VectorizeRetriever


def test_retrieve_init_args(
    environment: Literal["prod", "dev", "local", "staging"],
    api_token: str,
    org_id: str,
    pipeline_id: str,
) -> None:
    retriever = VectorizeRetriever(
        environment=environment,
        api_token=api_token,
        organization=org_id,
        pipeline_id=pipeline_id,
        num_results=2,
    )
    docs = retriever.invoke(input="What are you?")
    assert len(docs) == 2


def test_retrieve_invoke_args(
    environment: Literal["prod", "dev", "local", "staging"],
    api_token: str,
    org_id: str,
    pipeline_id: str,
) -> None:
    retriever = VectorizeRetriever(environment=environment, api_token=api_token)
    docs = retriever.invoke(
        input="What are you?",
        organization=org_id,
        pipeline_id=pipeline_id,
        num_results=2,
    )
    assert len(docs) == 2
