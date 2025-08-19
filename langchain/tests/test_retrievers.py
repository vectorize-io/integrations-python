import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import urllib3
from vectorize_client.api.ai_platform_connectors_api import AIPlatformConnectorsApi
from vectorize_client.api.destination_connectors_api import DestinationConnectorsApi
from vectorize_client.api.pipelines_api import PipelinesApi
from vectorize_client.api.source_connectors_api import SourceConnectorsApi
from vectorize_client.api.uploads_api import UploadsApi
from vectorize_client.api_client import ApiClient
from vectorize_client.configuration import Configuration
from vectorize_client.models.ai_platform_config_schema import AIPlatformConfigSchema
from vectorize_client.models.ai_platform_type_for_pipeline import (
    AIPlatformTypeForPipeline,
)
from vectorize_client.models.create_source_connector_request import (
    CreateSourceConnectorRequest,
)
from vectorize_client.models.destination_connector_type_for_pipeline import (
    DestinationConnectorTypeForPipeline,
)
from vectorize_client.models.file_upload import FileUpload
from vectorize_client.models.pipeline_ai_platform_connector_schema import (
    PipelineAIPlatformConnectorSchema,
)
from vectorize_client.models.pipeline_configuration_schema import (
    PipelineConfigurationSchema,
)
from vectorize_client.models.pipeline_destination_connector_schema import (
    PipelineDestinationConnectorSchema,
)
from vectorize_client.models.pipeline_source_connector_schema import (
    PipelineSourceConnectorSchema,
)
from vectorize_client.models.schedule_schema import ScheduleSchema
from vectorize_client.models.schedule_schema_type import ScheduleSchemaType
from vectorize_client.models.source_connector_type import SourceConnectorType
from vectorize_client.models.start_file_upload_to_connector_request import (
    StartFileUploadToConnectorRequest,
)

from langchain_vectorize.retrievers import VectorizeRetriever

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def api_token() -> str:
    token = os.getenv("VECTORIZE_TOKEN")
    if not token:
        msg = "Please set the VECTORIZE_TOKEN environment variable"
        raise ValueError(msg)
    return token


@pytest.fixture(scope="session")
def org_id() -> str:
    org = os.getenv("VECTORIZE_ORG")
    if not org:
        msg = "Please set the VECTORIZE_ORG environment variable"
        raise ValueError(msg)
    return org


@pytest.fixture(scope="session")
def environment() -> Literal["prod", "dev", "local", "staging"]:
    env = os.getenv("VECTORIZE_ENV", "prod")
    if env not in {"prod", "dev", "local", "staging"}:
        msg = "Invalid VECTORIZE_ENV environment variable."
        raise ValueError(msg)
    return env  # type: ignore[return-value]


@pytest.fixture(scope="session")
def api_client(api_token: str, environment: str) -> Iterator[ApiClient]:
    header_name = None
    header_value = None
    if environment == "prod":
        host = "https://api.vectorize.io/v1"
    elif environment == "dev":
        host = "https://api-dev.vectorize.io/v1"
    elif environment == "local":
        host = "http://localhost:3000/api"
        header_name = "x-lambda-api-key"
        header_value = api_token
    else:
        host = "https://api-staging.vectorize.io/v1"

    with ApiClient(
        Configuration(host=host, access_token=api_token, debug=True),
        header_name,
        header_value,
    ) as api:
        yield api


@pytest.fixture(scope="session")
def pipeline_id(api_client: ApiClient, org_id: str) -> Iterator[str]:
    pipelines = PipelinesApi(api_client)

    connectors_api = SourceConnectorsApi(api_client)
    response = connectors_api.create_source_connector(
        org_id,
        CreateSourceConnectorRequest(FileUpload(name="from api", type="FILE_UPLOAD")),
    )
    source_connector_id = response.connector.id
    logger.info("Created source connector %s", source_connector_id)

    uploads_api = UploadsApi(api_client)
    upload_response = uploads_api.start_file_upload_to_connector(
        org_id,
        source_connector_id,
        StartFileUploadToConnectorRequest(  # type: ignore[call-arg]
            name="research.pdf",
            content_type="application/pdf",
            metadata=json.dumps({"created-from-api": True}),
        ),
    )

    http = urllib3.PoolManager()
    this_dir = Path(__file__).parent
    file_path = this_dir / "research.pdf"

    with file_path.open("rb") as f:
        http_response = http.request(
            "PUT",
            upload_response.upload_url,
            body=f,
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": str(file_path.stat().st_size),
            },
        )
    if http_response.status != 200:
        msg = "Upload failed:"
        raise ValueError(msg)
    else:
        logger.info("Upload successful")

    ai_platforms = AIPlatformConnectorsApi(api_client).get_ai_platform_connectors(
        org_id
    )
    builtin_ai_platform = next(
        c.id for c in ai_platforms.ai_platform_connectors if c.type == "VECTORIZE"
    )
    logger.info("Using AI platform %s", builtin_ai_platform)

    vector_databases = DestinationConnectorsApi(api_client).get_destination_connectors(
        org_id
    )
    builtin_vector_db = next(
        c.id for c in vector_databases.destination_connectors if c.type == "VECTORIZE"
    )
    logger.info("Using destination connector %s", builtin_vector_db)

    pipeline_response = pipelines.create_pipeline(
        org_id,
        PipelineConfigurationSchema(  # type: ignore[call-arg]
            source_connectors=[
                PipelineSourceConnectorSchema(
                    id=source_connector_id,
                    type=SourceConnectorType.FILE_UPLOAD,
                    config={},
                )
            ],
            destination_connector=PipelineDestinationConnectorSchema(
                id=builtin_vector_db,
                type=DestinationConnectorTypeForPipeline.VECTORIZE,
                config={},
            ),
            ai_platform_connector=PipelineAIPlatformConnectorSchema(
                id=builtin_ai_platform,
                type=AIPlatformTypeForPipeline.VECTORIZE,
                config=AIPlatformConfigSchema(),
            ),
            pipeline_name="Test pipeline",
            schedule=ScheduleSchema(type=ScheduleSchemaType.MANUAL),
        ),
    )
    pipeline_id = pipeline_response.data.id
    logger.info("Created pipeline %s", pipeline_id)

    yield pipeline_id

    try:
        pipelines.delete_pipeline(org_id, pipeline_id)
    except Exception:
        logger.exception("Failed to delete pipeline %s", pipeline_id)


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
    start = time.time()
    while True:
        try:
            docs = retriever.invoke(input="What are you?")
            if len(docs) == 2:
                break
        except Exception as e:
            if "503" in str(e):
                continue
            raise RuntimeError(e) from e

        if time.time() - start > 180:
            msg = "Docs not retrieved in time"
            raise RuntimeError(msg)
        time.sleep(1)


def test_retrieve_invoke_args(
    environment: Literal["prod", "dev", "local", "staging"],
    api_token: str,
    org_id: str,
    pipeline_id: str,
) -> None:
    retriever = VectorizeRetriever(environment=environment, api_token=api_token)
    start = time.time()
    while True:
        try:
            docs = retriever.invoke(
                input="What are you?",
                organization=org_id,
                pipeline_id=pipeline_id,
                num_results=2,
            )
            if len(docs) == 2:
                break

        except Exception as e:
            if "503" in str(e):
                continue
            raise RuntimeError(e) from e
        if time.time() - start > 180:
            msg = "Docs not retrieved in time"
            raise RuntimeError(msg)

        time.sleep(1)
