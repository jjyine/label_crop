# fetch_data.py
import os
import re
import json
import base64
import requests
import pymysql
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# -----------------------------
# ENV / 설정
# -----------------------------
load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SAM
USE_SAM = os.getenv("USE_SAM", "1") == "1"
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")  # vit_b / vit_l / vit_h
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "./sam_vit_b_01ec64.pth")

# 출력 폴더
OUT_DIR = os.path.join(os.getcwd(), "out")
DEBUG_DIR = os.path.join(os.getcwd(), "out_debug")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


# -----------------------------
# OpenAI client (optional)
# -----------------------------
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    _openai_client = None


# -----------------------------
# SAM imports (optional)
# -----------------------------
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    _sam_available = True
except Exception:
    _sam_available = False

_DEVICE = "cuda" if (_sam_available and torch.cuda.is_available()) else "cpu"
_sam_predictor = None


# -----------------------------
# DB 이미지 선택 + fetch
# -----------------------------
def select_largest_vivino_image(image_data):
    selected_image = None
    max_area = 0
    selected_key = None

    for image_url, s3_key in image_data.items():
        if "vivino.com" not in image_url:
            continue

        area = 0
        m = re.search(r"(\d+)x(\d+)", image_url)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            area = w * h
        else:
            m = re.search(r"x(\d+)", image_url)
            if m:
                size = int(m.group(1))
                area = size * size

        if area > max_area:
            max_area = area
            selected_image = image_url
            selected_key = s3_key

    return selected_image, selected_key


def fetch_data(limit=5, total_results=5):
    connection = None
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=3306,
        )

        offset = 0
        fetched_results = 0

        while fetched_results < total_results:
            with connection.cursor() as cursor:
                sql = f"SELECT s3_images FROM wine LIMIT {limit} OFFSET {offset};"
                cursor.execute(sql)
                results = cursor.fetchall()

                if not results:
                    print("No more images to fetch.")
                    break

                for row in results:
                    image_data = json.loads(row[0])
                    vivino_url, selected_key = select_largest_vivino_image(image_data)

                    if vivino_url and selected_key:
                        final_image_url = f"https://vin-social.s3.amazonaws.com/{selected_key}"
                        print("Fetching image from URL:", final_image_url)

                        response = requests.get(final_image_url)
                        response.raise_for_status()

                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                        fetched_results += 1
                        yield img_cv, final_image_url

                offset += limit

    except pymysql.MySQLError as e:
        print("DB connect failed:", e)
    finally:
        if connection:
            connection.close()


# -----------------------------
# 유틸
# -----------------------------
def safe_filename(url: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", url)
    return name[-80:]


def clamp_xyxy(x1, y1, x2, y2, w, h, pad=0):
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w - 1, int(x2) + pad)
    y2 = min(h - 1, int(y2) + pad)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def encode_bgr_to_data_url_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def safe_json_extract(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from model output.")


# -----------------------------
# 1) EDGES -> candidate box
# -----------------------------
def create_edge_assist_image(bgr: np.ndarray):
    """
    - gray + CLAHE + Canny + dilate
    - assist: gray(3ch)에 edge를 흰색으로 표시
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    assist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assist[edges > 0] = (255, 255, 255)

    return assist, edges, enhanced


def find_label_candidate_box_from_edges(edges: np.ndarray):
    """
    edges(0/255)에서 라벨 후보 박스(대략)를 찾음.
    핵심: edge를 좀 뭉쳐서(contour) 사각형 후보를 고르고,
    중앙~하단 / 적당한 비율 / 충분한 면적을 점수화해서 베스트 선택.
    """
    h, w = edges.shape[:2]

    # edges를 덩어리로 만들기
    work = edges.copy()
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    work = cv2.dilate(work, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (w * h) * 0.01:
            continue

        aspect = cw / max(1, ch)
        # 라벨은 "대충 직사각형 영역"이 많아서 너무 세로/가로 극단은 제외
        if not (0.55 <= aspect <= 3.8):
            continue

        # 위치 점수: 중앙 + 하단 근처 선호
        cx = x + cw / 2
        cy = y + ch / 2
        center_bonus = 1.0 - (abs(cx - w / 2) / (w / 2)) * 0.7
        vertical_bonus = 1.0 - (abs(cy - h * 0.62) / (h * 0.62)) * 0.7

        # 너무 위(목쪽)면 강하게 패널티
        if cy < h * 0.22:
            vertical_bonus *= 0.2

        score = area * max(0.05, center_bonus) * max(0.05, vertical_bonus)

        if score > best_score:
            best_score = score
            best = (x, y, x + cw, y + ch)

    if best is None:
        return None, work

    x1, y1, x2, y2 = best
    # 약간 확장 (OpenAI/SAM이 경계 잡기 좋게)
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.12)
    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w, h, pad=0)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return np.array([x1, y1, x2, y2], dtype=np.int32), work


# -----------------------------
# 2) OpenAI -> bbox refine (candidate box 기반)
# -----------------------------
def openai_refine_bbox(color_bgr: np.ndarray, assist_bgr: np.ndarray, candidate_box_xyxy: np.ndarray | None):
    if _openai_client is None:
        return None, None

    h, w = color_bgr.shape[:2]
    color_data = encode_bgr_to_data_url_png(color_bgr)
    assist_data = encode_bgr_to_data_url_png(assist_bgr)

    cand_txt = "none"
    if candidate_box_xyxy is not None:
        x1, y1, x2, y2 = map(int, candidate_box_xyxy.tolist())
        cand_txt = f"[{x1}, {y1}, {x2}, {y2}]"

    instruction = f"""
You are given TWO images:
(1) Original COLOR image of a wine bottle.
(2) Edge-enhanced GRAYSCALE assist image (white lines are strong edges).

We also provide an initial candidate box (xyxy) computed from edges:
candidate_box = {cand_txt}

Task:
Return a TIGHT bounding box around ONLY the front paper label on the bottle body.

Rules:
- Use candidate_box as a strong hint. The label is very likely inside/near it.
- Use the assist image to align with real label boundaries (top/bottom especially).
- Do NOT include bottle neck/capsule.
- Do NOT include bottle glass outside the paper label.
- Do NOT include background.

Return ONLY valid JSON:
{{
  "label_bboxes": [{{"x1": int, "y1": int, "x2": int, "y2": int}}],
  "notes": "short"
}}

Image size: width={w}, height={h}
"""

    resp = _openai_client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_image", "image_url": color_data},
                {"type": "input_image", "image_url": assist_data},
            ],
        }],
    )

    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in getattr(resp, "output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        parts.append(c.get("text", ""))
        text = "\n".join(parts).strip()

    try:
        data = safe_json_extract(text)
    except Exception:
        return None, text

    bboxes = data.get("label_bboxes", [])
    if not isinstance(bboxes, list) or len(bboxes) == 0:
        return None, text

    # 여러개면 가장 큰 면적 선택
    cand = []
    for b in bboxes:
        if not all(k in b for k in ("x1", "y1", "x2", "y2")):
            continue
        x1, y1, x2, y2 = clamp_xyxy(b["x1"], b["y1"], b["x2"], b["y2"], w, h, pad=int(min(w, h) * 0.01))
        cand.append((x1, y1, x2, y2))

    if not cand:
        return None, text

    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in cand]
    best = cand[int(np.argmax(areas))]
    return best, text


# -----------------------------
# 3) SAM refine (OpenAI bbox 기반)
# -----------------------------
def get_sam_predictor():
    global _sam_predictor

    if _sam_predictor is not None:
        return _sam_predictor

    if not USE_SAM:
        return None
    if not _sam_available:
        return None
    if not os.path.exists(SAM_CHECKPOINT):
        return None

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=_DEVICE)
    _sam_predictor = SamPredictor(sam)
    print(f"[SAM] loaded: type={SAM_MODEL_TYPE}, device={_DEVICE}, ckpt={SAM_CHECKPOINT}")
    return _sam_predictor


def refine_bbox_with_sam(bgr: np.ndarray, init_bbox_xyxy: tuple[int, int, int, int]):
    predictor = get_sam_predictor()
    if predictor is None:
        return None, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    x1, y1, x2, y2 = init_bbox_xyxy
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, mask

    rx1, rx2 = int(xs.min()), int(xs.max())
    ry1, ry2 = int(ys.min()), int(ys.max())

    h, w = bgr.shape[:2]
    rx1, ry1, rx2, ry2 = clamp_xyxy(rx1, ry1, rx2, ry2, w, h, pad=int(min(w, h) * 0.005))
    return (rx1, ry1, rx2, ry2), mask


def crop_by_xyxy(bgr: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return bgr[y1:y2, x1:x2].copy()


# -----------------------------
# 디버깅 저장
# -----------------------------
def save_debug(idx: int,
               url: str,
               bgr: np.ndarray,
               assist: np.ndarray,
               edges: np.ndarray,
               edges_work: np.ndarray,
               candidate_box: np.ndarray | None,
               openai_bbox: tuple[int, int, int, int] | None,
               sam_bbox: tuple[int, int, int, int] | None,
               sam_mask: np.ndarray | None,
               openai_text: str | None):
    base = safe_filename(url)
    prefix = f"{idx:03d}_{base}"

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_orig.png"), bgr)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_assist.png"), assist)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_edges.png"), edges)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_edges_work.png"), edges_work)

    overlay = bgr.copy()

    if candidate_box is not None:
        x1, y1, x2, y2 = map(int, candidate_box.tolist())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)  # 보라: edges candidate
        cv2.putText(overlay, "edges_cand", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    if openai_bbox is not None:
        x1, y1, x2, y2 = openai_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 노랑: openai
        cv2.putText(overlay, "openai", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if sam_bbox is not None:
        x1, y1, x2, y2 = sam_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 초록: sam
        cv2.putText(overlay, "sam", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_overlay.png"), overlay)

    if sam_mask is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_mask.png"), (sam_mask * 255).astype(np.uint8))

        m = sam_mask.astype(bool)
        mask_overlay = bgr.copy()
        white = np.full_like(mask_overlay, 255)
        mask_overlay[m] = cv2.addWeighted(mask_overlay[m], 0.4, white[m], 0.6, 0)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_overlay.png"), mask_overlay)

    if openai_text:
        with open(os.path.join(DEBUG_DIR, f"{prefix}_openai_response.txt"), "w", encoding="utf-8") as f:
            f.write(openai_text)


# -----------------------------
# edges -> OpenAI -> SAM 파이프라인
# -----------------------------
def detect_label_edges_openai_sam(bgr: np.ndarray, url: str):
    """
    반환:
      final_bbox_xyxy or None,
      debug dict
    """
    assist, edges, enhanced = create_edge_assist_image(bgr)
    candidate_box, edges_work = find_label_candidate_box_from_edges(edges)

    # 기본은 edges candidate 를 시작점으로 (OpenAI/SAM 실패 시 fallback)
    edges_bbox = None
    if candidate_box is not None:
        x1, y1, x2, y2 = map(int, candidate_box.tolist())
        edges_bbox = (x1, y1, x2, y2)

    # 2) OpenAI refine (edges candidate hint)
    openai_bbox, openai_text = openai_refine_bbox(bgr, assist, candidate_box)

    # OpenAI가 실패하면 edges_bbox를 쓴다
    base_bbox = openai_bbox if openai_bbox is not None else edges_bbox

    # 3) SAM refine (base_bbox 기반)
    sam_bbox = None
    sam_mask = None
    if base_bbox is not None and USE_SAM:
        rb, mask = refine_bbox_with_sam(bgr, base_bbox)
        if rb is not None:
            sam_bbox = rb
        sam_mask = mask

    final_bbox = sam_bbox if sam_bbox is not None else base_bbox

    debug = {
        "assist": assist,
        "edges": edges,
        "edges_work": edges_work,
        "candidate_box": candidate_box,
        "openai_bbox": openai_bbox,
        "sam_bbox": sam_bbox,
        "sam_mask": sam_mask,
        "openai_text": openai_text,
    }
    return final_bbox, debug


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    print("[DEBUG] OUT_DIR:", OUT_DIR)
    print("[DEBUG] DEBUG_DIR:", DEBUG_DIR)
    print("[DEBUG] OpenAI:", "ON" if _openai_client else "OFF")
    print("[DEBUG] SAM:", "ON" if (USE_SAM and _sam_available and os.path.exists(SAM_CHECKPOINT)) else "OFF")

    for i, (img_cv, url) in enumerate(fetch_data(limit=5, total_results=5), start=1):
        final_bbox, dbg = detect_label_edges_openai_sam(img_cv, url)

        # 디버그 항상 저장
        save_debug(
            idx=i,
            url=url,
            bgr=img_cv,
            assist=dbg["assist"],
            edges=dbg["edges"],
            edges_work=dbg["edges_work"],
            candidate_box=dbg["candidate_box"],
            openai_bbox=dbg["openai_bbox"],
            sam_bbox=dbg["sam_bbox"],
            sam_mask=dbg["sam_mask"],
            openai_text=dbg["openai_text"],
        )

        if final_bbox is None:
            print(f"[{i:03d}] ❌ label not found:", url)
            continue

        crop = crop_by_xyxy(img_cv, final_bbox)

        base = safe_filename(url)
        out_img = os.path.join(OUT_DIR, f"{i:03d}_{base}_label.png")
        cv2.imwrite(out_img, crop)

        method = "sam" if dbg["sam_bbox"] is not None else ("openai" if dbg["openai_bbox"] is not None else "edges")
        print(f"[{i:03d}] ✅ saved: {out_img} ({method})")