import json
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def normalize_box(geom):
    """
    Normalize docTR geometry into (xmin, ymin, xmax, ymax) in relative coords [0, 1].

    geom can be:
    - [xmin, ymin, xmax, ymax]
    - ((xmin, ymin), (xmax, ymax))
    - [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]  # polygon
    """
    # Case 1: flat [xmin, ymin, xmax, ymax]
    if len(geom) == 4 and not isinstance(geom[0], (list, tuple)):
        xmin, ymin, xmax, ymax = geom
        return float(xmin), float(ymin), float(xmax), float(ymax)

    # Case 2: 2 points: ((xmin, ymin), (xmax, ymax))
    if len(geom) == 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in geom):
        (xmin, ymin), (xmax, ymax) = geom
        return float(xmin), float(ymin), float(xmax), float(ymax)

    # Case 3: polygon: list of (x, y)
    xs = [p[0] for p in geom]
    ys = [p[1] for p in geom]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def box_to_quad(geom, img_w, img_h):
    """
    Convert docTR geometry to 8-point boundingBox [x1,y1, x2,y2, x3,y3, x4,y4] in pixels.
    Order: top-left, top-right, bottom-right, bottom-left.
    """
    xmin, ymin, xmax, ymax = normalize_box(geom)
    x1, y1 = xmin * img_w, ymin * img_h
    x2, y2 = xmax * img_w, ymin * img_h
    x3, y3 = xmax * img_w, ymax * img_h
    x4, y4 = xmin * img_w, ymax * img_h
    return [
        int(round(x1)), int(round(y1)),
        int(round(x2)), int(round(y2)),
        int(round(x3)), int(round(y3)),
        int(round(x4)), int(round(y4)),
    ]


def doctr_ocr_to_json(image_path: str):
    # 1. Load image as a DocumentFile
    doc = DocumentFile.from_images(image_path)

    # 2. Load end-to-end OCR model (detection + recognition)
    predictor = ocr_predictor(
        det_arch="db_resnet50",     # detector backbone
        reco_arch="crnn_vgg16_bn",  # recognizer backbone
        pretrained=True,
    )

    # 3. Run prediction
    result = predictor(doc)

    # docTR gives relative coords, need absolute pixel values
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")
    img_h, img_w = img.shape[:2]

    pages_output = []

    # 4. Iterate through pages/blocks/lines/words
    for page_idx, page in enumerate(result.pages):
        page_lines = []

        for block in page.blocks:
            for line in block.lines:
                # compute line bounding box from all word boxes
                xs, ys = [], []
                for word in line.words:
                    xmin, ymin, xmax, ymax = normalize_box(word.geometry)
                    xs.extend([xmin * img_w, xmax * img_w])
                    ys.extend([ymin * img_h, ymax * img_h])

                if not xs or not ys:
                    continue

                line_box = [
                    int(round(min(xs))), int(round(min(ys))),
                    int(round(max(xs))), int(round(min(ys))),
                    int(round(max(xs))), int(round(max(ys))),
                    int(round(min(xs))), int(round(max(ys))),
                ]

                line_text = " ".join(w.value for w in line.words)

                words_output = []
                for word in line.words:
                    w_box = box_to_quad(word.geometry, img_w, img_h)
                    word_dict = {
                        "boundingBox": w_box,
                        "text": word.value,
                    }
                    # If your docTR version has confidence, include it
                    if hasattr(word, "confidence"):
                        word_dict["confidence"] = float(word.confidence)
                    words_output.append(word_dict)

                page_lines.append({
                    "boundingBox": line_box,
                    "text": line_text,
                    "words": words_output,
                })

        pages_output.append({
            "page": page_idx,
            "lines": page_lines,
        })

    return pages_output


if __name__ == "__main__":
    IMAGE_PATH = r"data\\spdocvqa_images\\zzml0226_1.png"  # change to your image

    data = doctr_ocr_to_json(IMAGE_PATH)
    with open("ocr/test_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
