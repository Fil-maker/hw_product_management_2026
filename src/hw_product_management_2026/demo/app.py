"""Gradio demo: drop an image into the browser, see the top-5 classes of the deployed model.

The demo never loads the model and never talks to Triton: every prediction is a POST /predict
to the HTTP wrapper from hw4.

Configuration (environment variables):
    API_URL              address of the wrapper, e.g. http://api:8080 inside docker compose
    API_TIMEOUT_SECONDS  how long to wait for the wrapper before reporting it (default 10)
    MODEL_INFO_PATH      model_info.json written by cifar-export: class names and the run
    EXAMPLES_DIR         directory with the clickable example images (default demo/examples)
    GRADIO_SERVER_NAME / GRADIO_SERVER_PORT  where Gradio listens (read by Gradio itself)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import requests
from PIL import Image

from hw_product_management_2026.data.common import read_json, setup_logging
from hw_product_management_2026.serving.preprocess import (
    IMAGE_SIZE,
    InvalidImageError,
    decode_image,
)

log = logging.getLogger(__name__)

TOP_K = 5
PREVIEW_SIZE = 192
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_EXAMPLES_DIR = Path("demo/examples")
# A 32x32 picture from test would otherwise be a thumbnail in the corner of the upload box.
CSS = (
    "#input-image img "
    "{ width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }"
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class ModelInfo:
    classes: list[str]
    experiment_name: str
    run_name: str
    data_version: str

    @classmethod
    def load(cls, path: Path) -> ModelInfo:
        raw: dict[str, Any] = read_json(path)
        return cls(
            classes=list(raw["classes"]),
            experiment_name=raw["experiment_name"],
            run_name=raw["run_name"],
            data_version=raw["data_version"],
        )


@dataclass(frozen=True)
class Settings:
    api_url: str
    timeout: float
    info: ModelInfo
    examples_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_url=_required_env("API_URL").rstrip("/"),
            timeout=float(os.environ.get("API_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
            info=ModelInfo.load(Path(_required_env("MODEL_INFO_PATH"))),
            examples_dir=Path(os.environ.get("EXAMPLES_DIR", DEFAULT_EXAMPLES_DIR)),
        )


class WrapperError(Exception):
    """Something the user has to be told about in plain words."""


def _error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200] or response.reason
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def request_prediction(settings: Settings, data: bytes, file_name: str) -> dict[str, Any]:
    """POST /predict; any failure becomes a WrapperError with a message for the user."""
    url = f"{settings.api_url}/predict"
    try:
        response = requests.post(url, files={"file": (file_name, data)}, timeout=settings.timeout)
    except requests.Timeout as e:
        raise WrapperError(
            f"Сервис модели не ответил за {settings.timeout:g} с. Попробуйте ещё раз позже."
        ) from e
    except requests.RequestException as e:
        log.warning("wrapper is not reachable: %s", e)
        raise WrapperError(
            f"Сервис модели недоступен ({settings.api_url}). "
            "Проверьте, что контейнер api запущен: make demo-up."
        ) from e

    if response.status_code in (400, 413, 422):
        raise WrapperError(f"Сервис не принял файл: {_error_detail(response)}")
    if response.status_code in (502, 503):
        raise WrapperError(f"Сервис модели сейчас не работает: {_error_detail(response)}")
    if not response.ok:
        raise WrapperError(
            f"Сервис модели вернул ошибку {response.status_code}: {_error_detail(response)}"
        )
    result: dict[str, Any] = response.json()
    return result


def top_classes(probabilities: list[float], classes: list[str]) -> dict[str, float]:
    if len(probabilities) != len(classes):
        raise WrapperError(
            f"Модель вернула {len(probabilities)} вероятностей, а классов {len(classes)}: "
            "model_info.json не от развёрнутой модели. Перезапустите make export."
        )
    order = np.argsort(probabilities)[::-1][:TOP_K]
    return {classes[i]: float(probabilities[i]) for i in order}


def model_input_preview(data: bytes) -> Image.Image | None:
    """The 32x32 picture the model actually gets, scaled up so that it can be seen."""
    try:
        pixels = decode_image(data)
    except InvalidImageError:
        return None
    return Image.fromarray(pixels).resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.NEAREST)


def description(info: ModelInfo) -> str:
    classes = ", ".join(f"`{name}`" for name in info.classes)
    return (
        "# Классификатор CIFAR-10\n\n"
        "Перетащите картинку (PNG или JPEG) в левое окно, выберите файл или нажмите "
        "на один из примеров — предсказание появится справа.\n\n"
        f"**Модель:** свёрточная сеть из эксперимента `{info.experiment_name}` "
        f"(запуск `{info.run_name}`), обучена на `{info.data_version}` — картинках "
        f"{IMAGE_SIZE}×{IMAGE_SIZE} пикселя. **Она знает только {len(info.classes)} классов:** "
        f"{classes}.\n\n"
        "Любую другую картинку — фото человека, еды, пейзажа — модель всё равно отнесёт "
        "к одному из этих классов: варианта «ничего из перечисленного» у неё нет. "
        f"Кроме того, перед предсказанием картинка уменьшается до {IMAGE_SIZE}×{IMAGE_SIZE}: "
        "что именно видит модель, показано под результатом."
    )


def build_demo(settings: Settings) -> gr.Blocks:
    info = settings.info

    def predict(path: str | None) -> tuple[dict[str, float] | None, Image.Image | None, str]:
        if path is None:  # the image was cleared
            return None, None, ""
        data = Path(path).read_bytes()
        try:
            result = request_prediction(settings, data, Path(path).name)
            top = top_classes(result["probabilities"], info.classes)
        except WrapperError as e:
            # Shown in place of the result, so that the previous answer does not stay on
            # screen as if it belonged to this picture, and also as a pop-up.
            gr.Warning(str(e), duration=None)
            return None, None, f"### ⚠️ Предсказания нет\n\n{e}"
        best, confidence = next(iter(top.items()))
        summary = (
            f"**Ответ модели:** `{best}`, {confidence:.1%}\n\n"
            f"Модель: `{result['model_name']}`, версия `{result['model_version']}`"
        )
        return top, model_input_preview(data), summary

    demo = gr.Blocks(title="CIFAR-10 demo")
    with demo:
        gr.Markdown(description(info))
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    label="Картинка",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=360,
                    elem_id="input-image",
                )
                examples = sorted(settings.examples_dir.glob("*.png"))
                if examples:
                    gr.Examples(
                        examples=[[str(path)] for path in examples],
                        inputs=image,
                        label="Примеры из test — по одному клику",
                    )
            with gr.Column():
                label = gr.Label(label=f"Топ-{TOP_K} классов", num_top_classes=TOP_K)
                summary = gr.Markdown()
                preview = gr.Image(
                    label=f"Что видит модель: {IMAGE_SIZE}×{IMAGE_SIZE}",
                    interactive=False,
                    height=PREVIEW_SIZE + 40,
                )
        # `change` fires on drag-n-drop, file dialog, paste, example click and clear alike.
        image.change(predict, inputs=image, outputs=[label, preview, summary])
    return demo


def main() -> None:
    setup_logging()
    settings = Settings.from_env()
    log.info("wrapper: %s, timeout %gs", settings.api_url, settings.timeout)
    build_demo(settings).launch(
        allowed_paths=[str(settings.examples_dir.resolve())],
        css=CSS,
    )


if __name__ == "__main__":
    main()
