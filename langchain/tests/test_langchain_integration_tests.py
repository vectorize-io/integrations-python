from typing import Any, Literal

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from langchain_vectorize import VectorizeRetriever


class TestVectorizeRetrieverIntegration(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup(
        self,
        environment: Literal["prod", "dev", "local", "staging"],
        api_token: str,
        org_id: str,
        pipeline_id: str,
    ) -> None:
        self._environment = environment
        self._api_token = api_token
        self._org_id = org_id
        self._pipeline_id = pipeline_id

    @property
    def retriever_constructor(self) -> type[VectorizeRetriever]:
        return VectorizeRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {
            "environment": self._environment,
            "api_token": self._api_token,
            "organization": self._org_id,
            "pipeline_id": self._pipeline_id,
        }

    @property
    def retriever_query_example(self) -> str:
        return "What are you?"

    @pytest.mark.xfail(
        reason="VectorizeRetriever does not support k parameter in constructor"
    )
    def test_k_constructor_param(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(
        reason="VectorizeRetriever does not support k parameter in invoke"
    )
    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        raise NotImplementedError
