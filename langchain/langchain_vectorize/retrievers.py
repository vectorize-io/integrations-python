"""Vectorize LangChain retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override
from vectorize_client.api.pipelines_api import PipelinesApi
from vectorize_client.api_client import ApiClient
from vectorize_client.configuration import Configuration
from vectorize_client.models.retrieve_documents_request import RetrieveDocumentsRequest

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.runnables import RunnableConfig
    from vectorize_client.models.document import Document as VectorizeDocument

_METADATA_FIELDS = {
    "relevancy",
    "chunk_id",
    "total_chunks",
    "origin",
    "origin_id",
    "similarity",
    "source",
    "unique_source",
    "source_display_name",
    "pipeline_id",
    "org_id",
}
_NOT_SET = object()


class VectorizeRetriever(BaseRetriever):
    """Vectorize retriever.

    Setup:
        Install package ``langchain-vectorize``

        .. code-block:: bash

            pip install -U langchain-vectorize

    Init args:
        api_token: str
            The Vectorize API token.
        environment: Literal["prod", "dev", "local", "staging"]
            The Vectorize API environment. Defaults to "prod".
        organization: Optional[str]
            The Vectorize organization ID. Defaults to None.
        pipeline_id: Optional[str]
            The Vectorize pipeline ID. Defaults to None.
        num_results: int
            Number of documents to return. Defaults to 5.
        rerank: bool
            Whether to rerank the results. Defaults to False.
        metadata_filters: list[dict[str, Any]]
            The metadata filters to apply when retrieving the documents. Defaults to [].

    Instantiate:
        .. code-block:: python

            from langchain_vectorize import VectorizeRetriever

            retriever = VectorizeRetriever(
                api_token="xxxxx", "organization"="1234", "pipeline_id"="5678"
            )

    Usage:
        .. code-block:: python

            query = "what year was breath of the wild released?"
            retriever.invoke(query)

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-4o")

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("how many units did breath of the wild sell in 2020")
    """  # noqa: D301

    api_token: str
    """The Vectorize API token."""
    environment: Literal["prod", "dev", "local", "staging"] = "prod"
    """The Vectorize API environment."""
    organization: Optional[str] = None
    """The Vectorize organization ID."""
    pipeline_id: Optional[str] = None
    """The Vectorize pipeline ID."""
    num_results: int = 5
    """The number of documents to return."""
    rerank: bool = False
    """Whether to rerank the results."""
    metadata_filters: list[dict[str, Any]] = []
    """The metadata filters to apply when retrieving the documents."""

    _pipelines: PipelinesApi = _NOT_SET  # type: ignore[assignment]

    @override
    def model_post_init(self, /, context: Any) -> None:
        header_name = None
        header_value = None
        if self.environment == "prod":
            host = "https://api.vectorize.io/v1"
        elif self.environment == "dev":
            host = "https://api-dev.vectorize.io/v1"
        elif self.environment == "local":
            host = "http://localhost:3000/api"
            header_name = "x-lambda-api-key"
            header_value = self.api_token
        else:
            host = "https://api-staging.vectorize.io/v1"
        api = ApiClient(
            Configuration(host=host, access_token=self.api_token, debug=True),
            header_name,
            header_value,
        )
        self._pipelines = PipelinesApi(api)

    @staticmethod
    def _convert_document(document: VectorizeDocument) -> Document:
        metadata = {field: getattr(document, field) for field in _METADATA_FIELDS}
        return Document(id=document.id, page_content=document.text, metadata=metadata)

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        organization: str | None = None,
        pipeline_id: str | None = None,
        num_results: int | None = None,
        rerank: bool | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
    ) -> list[Document]:
        request = RetrieveDocumentsRequest(  # type: ignore[call-arg]
            question=query,
            num_results=num_results or self.num_results,
            rerank=rerank or self.rerank,
            metadata_filters=metadata_filters or self.metadata_filters,
        )
        organization_ = organization or self.organization
        if not organization_:
            msg = (
                "Organization must be set either at initialization "
                "or in the invoke method."
            )
            raise ValueError(msg)
        pipeline_id_ = pipeline_id or self.pipeline_id
        if not pipeline_id_:
            msg = (
                "Pipeline ID must be set either at initialization "
                "or in the invoke method."
            )
            raise ValueError(msg)

        response = self._pipelines.retrieve_documents(
            organization_, pipeline_id_, request
        )
        return [self._convert_document(doc) for doc in response.documents]

    @override
    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        organization: str = "",
        pipeline_id: str = "",
        num_results: int = _NOT_SET,  # type: ignore[assignment]
        rerank: bool = _NOT_SET,  # type: ignore[assignment]
        metadata_filters: list[dict[str, Any]] = _NOT_SET,  # type: ignore[assignment]
        **_kwargs: Any,
    ) -> list[Document]:
        """Invoke the retriever to get relevant documents.

        Main entry point for retriever invocations.

        Args:
            input: The query string.
            config: Configuration for the retriever. Defaults to None.
            organization: The organization to retrieve documents from.
                If set, overrides the organization set at the initialization of the
                retriever.
            pipeline_id: The pipeline ID to retrieve documents from.
                If set, overrides the pipeline ID set at the initialization of the
                retriever.
            num_results: The number of results to retrieve.
                If set, overrides the number of results set at the initialization of
                the retriever.
            rerank: Whether to rerank the retrieved documents.
                If set, overrides the reranking set at the initialization of the
                retriever.
            metadata_filters: The metadata filters to apply when retrieving documents.
                If set, overrides the metadata filters set at the initialization of the
                retriever.

        Returns:
            List of relevant documents.

        Examples:

            .. code-block:: python

                query = "what year was breath of the wild released?"
                docs = retriever.invoke(query, num_results=2)
        """
        if organization:
            _kwargs["organization"] = organization
        if pipeline_id:
            _kwargs["pipeline_id"] = pipeline_id
        if num_results is not _NOT_SET:
            _kwargs["num_results"] = num_results
        if rerank is not _NOT_SET:
            _kwargs["rerank"] = rerank
        if metadata_filters is not _NOT_SET:
            _kwargs["metadata_filters"] = metadata_filters

        return super().invoke(input, config, **_kwargs)
