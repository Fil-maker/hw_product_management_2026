"""HTTP wrapper over Triton: clients send an image and never see tensors or normalization.

Configuration (environment variables):
    TRITON_URL          host:port of the Triton HTTP API, e.g. triton:8000
    TRITON_MODEL_NAME   model name in Triton
    TRITON_MODEL_VERSION  model version in Triton
    STATS_PATH          stats.json of the data version the model was trained on
    MODEL_INFO_PATH     model_info.json written by cifar-export (class names)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from tritonclient.http import InferInput, InferRequestedOutput
from tritonclient.http.aio import InferenceServerClient
from tritonclient.utils import InferenceServerException

from hw_product_management_2026.data.common import read_json, setup_logging
from hw_product_management_2026.serving.preprocess import (
    InvalidImageError,
    Normalization,
    preprocess,
    softmax,
)

log = logging.getLogger(__name__)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TRITON_TIMEOUT_SECONDS = 30


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class Service:
    client: InferenceServerClient
    model_name: str
    model_version: str
    normalization: Normalization
    classes: list[str]


class Prediction(BaseModel):
    class_id: int
    class_name: str
    probabilities: list[float]
    model_name: str
    model_version: str


class Health(BaseModel):
    status: str
    model_name: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    # A single client for the whole process: it keeps the connection pool to Triton.
    client = InferenceServerClient(_required_env("TRITON_URL"))
    app.state.service = Service(
        client=client,
        model_name=_required_env("TRITON_MODEL_NAME"),
        model_version=_required_env("TRITON_MODEL_VERSION"),
        normalization=Normalization.from_stats_file(Path(_required_env("STATS_PATH"))),
        classes=list(read_json(Path(_required_env("MODEL_INFO_PATH")))["classes"]),
    )
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="CIFAR-10 classifier", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> Health:
    service: Service = request.app.state.service
    try:
        ready = await service.client.is_model_ready(service.model_name, service.model_version)
    except Exception as e:  # any failure to reach Triton means "not ready", whatever its type
        log.warning("Triton is not reachable: %s", e)
        raise HTTPException(503, "Triton is not reachable") from e
    if not ready:
        raise HTTPException(503, f"model {service.model_name} is not ready in Triton")
    return Health(status="ok", model_name=service.model_name, model_version=service.model_version)


@app.post("/predict")
async def predict(request: Request, file: Annotated[UploadFile, File()]) -> Prediction:
    service: Service = request.app.state.service
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "the uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"the file is larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        # Decoding a big photo takes long enough to hold up other requests on the event loop.
        batch = await run_in_threadpool(preprocess, data, service.normalization)
    except InvalidImageError as e:
        raise HTTPException(400, str(e)) from e

    model_input = InferInput(INPUT_NAME, list(batch.shape), "FP32")
    model_input.set_data_from_numpy(batch)
    try:
        response = await service.client.infer(
            service.model_name,
            [model_input],
            model_version=service.model_version,
            outputs=[InferRequestedOutput(OUTPUT_NAME)],
            timeout=TRITON_TIMEOUT_SECONDS,
        )
    except InferenceServerException as e:
        log.error("Triton returned an error: %s", e)
        raise HTTPException(502, "the model server returned an error") from e
    except Exception as e:  # connection errors of the HTTP client
        log.error("Triton is not reachable: %s", e)
        raise HTTPException(503, "the model server is not reachable") from e

    probabilities = softmax(response.as_numpy(OUTPUT_NAME)[0])
    class_id = int(probabilities.argmax())
    return Prediction(
        class_id=class_id,
        class_name=service.classes[class_id],
        probabilities=probabilities.tolist(),
        model_name=service.model_name,
        model_version=service.model_version,
    )
