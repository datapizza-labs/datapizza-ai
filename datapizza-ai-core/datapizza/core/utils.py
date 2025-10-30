import base64
import io
import logging
import os
import sys
from typing import Any

from typing_extensions import override

from datapizza.core.clients.models import TokenUsage

# logger: logging.Logger = logging.getLogger(__name__)


SENSITIVE_HEADERS = {"api-key", "authorization"}


def is_dict(obj: object) -> bool:
    return isinstance(obj, dict)


def _basic_config(logger: logging.Logger) -> None:
    # Color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31;1m",  # Bright Red
    }
    RESET = "\033[0m"

    # Create a formatter with level at the start and color
    formatter = logging.Formatter(
        fmt="%(levelname_colored)s [%(asctime)s - %(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add color to the log level
    old_format = formatter.format

    def format(record):
        record.levelname_colored = (
            f"{COLORS.get(record.levelname, '')}{record.levelname:<8}{RESET}"
        )
        return old_format(record)

    formatter.format = format

    # Create a handler (console handler) using stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(EnvLogLevelFilter())
    console_handler.addFilter(SensitiveHeadersFilter())

    logger.addFilter(EnvLogLevelFilter())
    logger.addFilter(SensitiveHeadersFilter())
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.propagate = False


def sum_token_usage(usages: list[TokenUsage]) -> TokenUsage:
    return sum(usages, TokenUsage())


class EnvLogLevelFilter(logging.Filter):
    """Filter that checks the environment variable for log level before each log record."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        # Get current log level from environment
        env_level = os.getenv("DATAPIZZA_LOG_LEVEL", "INFO")
        # Convert string level to numeric level
        numeric_level = getattr(logging, env_level.upper(), logging.INFO)
        # Check if this record should pass based on environment level
        return record.levelno >= numeric_level


class SensitiveHeadersFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.args
            and isinstance(record.args, dict)
            and "headers" in record.args
            and isinstance(record.args["headers"], dict)
        ):
            headers = record.args["headers"] = {**record.args["headers"]}
            for header in headers:
                if str(header).lower() in SENSITIVE_HEADERS:
                    headers[header] = "<redacted>"
        return True


def extract_media(coordinates, file_path, page_number):
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is not installed. Please install it using `pip install PyMuPDF`."
        ) from e

    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError(
            "PIL is not installed. Please install it using `pip install Pillow`."
        ) from e

    doc = fitz.open(file_path)

    current_page = doc[page_number - 1]

    polygon_coords_points = [coord * 72 for coord in coordinates]

    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = current_page.get_pixmap(matrix=mat)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  # type: ignore

    x_scale = pix.width / current_page.rect.width
    y_scale = pix.height / current_page.rect.height

    pixel_coords = [
        (polygon_coords_points[i] * x_scale, polygon_coords_points[i + 1] * y_scale)
        for i in range(0, len(polygon_coords_points), 2)
    ]

    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(pixel_coords, fill=255)

    img_masked = Image.new("RGBA", img.size, (255, 255, 255, 0))
    img_masked.paste(img, (0, 0), mask)

    bbox = mask.getbbox()
    img_cropped = img_masked.crop(bbox)

    # Convert image to base64
    img_buffer = io.BytesIO()
    img_cropped.save(img_buffer, format="PNG")  # Save as PNG
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode(
        "utf-8"
    )  # Convert to base64 string

    return img_base64


# Helper function to replace environment variables
def replace_env_vars(value, constants: dict[str, str] | None = None) -> Any:
    if not constants:
        constants = {}

    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]

        if var_name in constants:
            return constants[var_name]

        env_value = os.environ.get(var_name)
        if not env_value:
            raise ValueError(f"Environment variable {var_name} not found or empty")
        return env_value
    elif isinstance(value, dict):
        return {k: replace_env_vars(v, constants) for k, v in value.items()}
    elif isinstance(value, list):
        return [replace_env_vars(item, constants) for item in value]
    else:
        return value
