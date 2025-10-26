def extract_media_from_docling_bbox(
    bbox: dict,
    file_path: str,
    page_number: int,
    *,
    zoom: float = 2.0,
) -> str:
    """Extract a base64 PNG crop from a PDF using a Docling-style bbox.

    The bbox is expected to be a dict with keys:
    - l, t, r, b: floats in PDF coordinate space
    - coord_origin: "BOTTOMLEFT" (Docling default) or "TOPLEFT" (string or enum)
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is not installed. Please install it using `pip install PyMuPDF`."
        ) from e

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "PIL is not installed. Please install it using `pip install Pillow`."
        ) from e

    if not isinstance(bbox, dict):
        raise ValueError("bbox must be a dict with keys l,t,r,b,coord_origin")

    l = float(bbox.get("l", 0.0))
    r = float(bbox.get("r", 0.0))
    t = float(bbox.get("t", 0.0))
    b = float(bbox.get("b", 0.0))

    # Normalize coord origin: accept enum (with .name), strings like
    # "CoordOrigin.TOPLEFT", "TOP_LEFT", etc.
    origin_val = bbox.get("coord_origin", "BOTTOMLEFT")
    if hasattr(origin_val, "name"):
        origin_str = str(getattr(origin_val, "name", "")).upper()
    else:
        origin_str = str(origin_val).upper()
    if "." in origin_str:
        origin_str = origin_str.split(".")[-1]
    origin_str = origin_str.replace("_", "").replace("-", "")
    if origin_str not in {"BOTTOMLEFT", "TOPLEFT"}:
        raise ValueError(f"Unsupported coord_origin: {origin_val!r}")

    with fitz.open(file_path) as doc:
        page = doc[page_number - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)  # type: ignore

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)  # type: ignore
        x_scale = pix.width / page.rect.width
        y_scale = pix.height / page.rect.height

        if origin_str == "BOTTOMLEFT":
            left = l * x_scale
            right = r * x_scale
            top = pix.height - (t * y_scale)
            bottom = pix.height - (b * y_scale)
        else:
            left = l * x_scale
            right = r * x_scale
            top = t * y_scale
            bottom = b * y_scale

        left_i = max(0, min(round(left), img.width))
        right_i = max(0, min(round(right), img.width))
        top_i = max(0, min(round(top), img.height))
        bottom_i = max(0, min(round(bottom), img.height))

        # Ensure proper ordering after rounding/clamping
        left_i, right_i = (left_i, right_i) if left_i <= right_i else (right_i, left_i)
        top_i, bottom_i = (top_i, bottom_i) if top_i <= bottom_i else (bottom_i, top_i)

        if right_i - left_i <= 1 or bottom_i - top_i <= 1:
            raise ValueError("Invalid crop region computed from bbox")

        crop = img.crop((left_i, top_i, right_i, bottom_i))

        import base64
        import io

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
