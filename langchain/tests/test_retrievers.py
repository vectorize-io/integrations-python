import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import urllib3
import vectorize_client as v
from vectorize_client import ApiClient

from langchain_vectorize.retrievers import VectorizeRetriever


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
    if env not in ["prod", "dev", "local", "staging"]:
        msg = "Invalid VECTORIZE_ENV environment variable."
        raise ValueError(msg)
    return env


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

    with v.ApiClient(
        v.Configuration(host=host, access_token=api_token, debug=True),
        header_name,
        header_value,
    ) as api:
        yield api


@pytest.fixture(scope="session")
def pipeline_id(api_client: v.ApiClient, org_id: str) -> Iterator[str]:
    pipelines = v.PipelinesApi(api_client)

    connectors_api = v.ConnectorsApi(api_client)
    response = connectors_api.create_source_connector(
        org_id,
        [
            v.CreateSourceConnector(
                name="from api", type=v.SourceConnectorType.FILE_UPLOAD
            )
        ],
    )
    source_connector_id = response.connectors[0].id
    logging.info("Created source connector %s", source_connector_id)

    uploads_api = v.UploadsApi(api_client)
    upload_response = uploads_api.start_file_upload_to_connector(
        org_id,
        source_connector_id,
        v.StartFileUploadToConnectorRequest(
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
        logging.info("Upload successful")

    ai_platforms = connectors_api.get_ai_platform_connectors(org_id)
    builtin_ai_platform = next(
        c.id for c in ai_platforms.ai_platform_connectors if c.type == "VECTORIZE"
    )
    logging.info("Using AI platform %s", builtin_ai_platform)

    vector_databases = connectors_api.get_destination_connectors(org_id)
    builtin_vector_db = next(
        c.id for c in vector_databases.destination_connectors if c.type == "VECTORIZE"
    )
    logging.info("Using destination connector %s", builtin_vector_db)

    pipeline_response = pipelines.create_pipeline(
        org_id,
        v.PipelineConfigurationSchema(
            source_connectors=[
                v.SourceConnectorSchema(
                    id=source_connector_id,
                    type=v.SourceConnectorType.FILE_UPLOAD,
                    config={},
                )
            ],
            destination_connector=v.DestinationConnectorSchema(
                id=builtin_vector_db,
                type=v.DestinationConnectorType.VECTORIZE,
                config={},
            ),
            ai_platform=v.AIPlatformSchema(
                id=builtin_ai_platform,
                type=v.AIPlatformType.VECTORIZE,
                config=v.AIPlatformConfigSchema(),
            ),
            pipeline_name="Test pipeline",
            schedule=v.ScheduleSchema(type=v.ScheduleSchemaType.MANUAL),
        ),
    )
    pipeline_id = pipeline_response.data.id
    logging.info("Created pipeline %s", pipeline_id)

    yield pipeline_id

    try:
        pipelines.delete_pipeline(org_id, pipeline_id)
    except Exception:
        logging.exception("Failed to delete pipeline %s", pipeline_id)


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
        docs = retriever.invoke(input="What are you?")
        if len(docs) == 2:
            break
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
        docs = retriever.invoke(
            input="What are you?",
            organization=org_id,
            pipeline_id=pipeline_id,
            num_results=2,
        )
        if len(docs) == 2:
            break
        if time.time() - start > 180:
            msg = "Docs not retrieved in time"
            raise RuntimeError(msg)
        time.sleep(1)
