import os
import json
import cv2
import re
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv 

# -----------------------------
# 설정
# -----------------------------
load_dotenv()

OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 읽기
OUT_LABEL_TEMPLATE = "cropped_label{}.png"  # 파일 이름 템플릿
client = OpenAI(api_key=API_KEY)  # API 키 설정

# -----------------------------
# 유틸
# -----------------------------
def clamp_bbox(x1, y1, x2, y2, w, h, pad=0):
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def safe_json_extract(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from model output.")

def crop_labels(image, data_url, index):
    instruction = f"""
    You are given a photo that contains a wine bottle.
    Return ONLY valid JSON.

    Goal:
    Detect the main front paper label on the bottle.

    Return schema:
    {{
      "label_bboxes": [
        {{ "x1": int, "y1": int, "x2": int, "y2": int }}
      ],
      "notes": "short"
    }}

    Coordinates are in pixels relative to:
    width={image.shape[1]}, height={image.shape[0]}
    """

    # OpenAI API 호출
    resp = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )

    # 응답 텍스트 추출
    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in resp.output:
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
        text = "\n".join(parts).strip()

    data = safe_json_extract(text)
    label_bboxes = data.get("label_bboxes", [])

    # 크롭 및 저장 (OpenCV만 사용)
    for idx, lb in enumerate(label_bboxes):

        lx1, ly1, lx2, ly2 = clamp_bbox(
            lb["x1"], lb["y1"], lb["x2"], lb["y2"],
            image.shape[1], image.shape[0],
            pad=20
        )

        cropped_label = image[ly1:ly2, lx1:lx2].copy()

        output_filename = OUT_LABEL_TEMPLATE.format(index + idx + 1)

        # OpenCV로 저장 (색상 문제 없음)
        success = cv2.imwrite(output_filename, cropped_label)

        if success:
            print(f"Saved label crop {index + idx + 1}: {output_filename}")
        else:
            print(f"Failed to save label crop {index + idx + 1}")
