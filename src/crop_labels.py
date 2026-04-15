import os
import re
import json
import pymysql
import base64
import numpy as np
import cv2
import boto3
import logging
import time
from urllib.parse import urlparse
from botocore.exceptions import BotoCoreError, ClientError
import threading

_thread_local = threading.local()

# -----------------------------
# 환경 설정
# -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Gemini AI 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

# aws 설정
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_LABEL_PREFIX = os.getenv("S3_LABEL_PREFIX")

# SAM 사용 여부 및 체크포인트
USE_SAM = os.getenv("USE_SAM", "1") == "1"
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")  # vit_b / vit_l / vit_h
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")  # 상대/절대 경로 모두 가능

# 테스트용 토글
# 1이면 실제 실행, 0이면 스킵
UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "1") == "1"
SAVE_TO_DB = os.getenv("SAVE_TO_DB", "1") == "1"

# -----------------------------
# Gemini client
# -----------------------------
try:
    from google.genai import Client  # Google Generative AI 클라이언트 임포트
    _gemini_client = Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    _gemini_client = None

from google.genai import types

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

# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "out")
DEBUG_DIR = os.path.join(BASE_DIR, "out_debug")
OUT_LABEL_TEMPLATE = "cropped_label{:03d}.png"

# -----------------------------
# 유틸
# -----------------------------
def get_logger():
    return getattr(_thread_local, "logger", None)

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s[-80:] if len(s) > 80 else s

def clamp_bbox(x1, y1, x2, y2, w, h, pad=0):
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w - 1, int(x2) + pad)
    y2 = min(h - 1, int(y2) + pad)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
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

def encode_bgr_to_data_url_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# -----------------------------
# 하이라이트(반사) 제거
# -----------------------------
def reduce_specular_glare(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    glare = cv2.inRange(hsv, (0, 0, 230), (180, 40, 255))
    k = np.ones((5, 5), np.uint8)
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, k, iterations=1)
    glare = cv2.dilate(glare, k, iterations=1)
    out = cv2.inpaint(bgr, glare, 3, cv2.INPAINT_TELEA)
    return out

# -----------------------------
# 이미지 전처리
# -----------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    return image.copy()

# -----------------------------
# SAM 멀티마스크 중 "라벨다운" 마스크 선택용 스코어링
# -----------------------------
def score_label_mask(mask: np.ndarray, w: int, h: int) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return -1e9

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)

    area = float(mask.sum())
    bbox_area = float(bw * bh) + 1e-6
    fill = area / bbox_area

    ar = bw / float(bh + 1e-6)
    ar_pen = 1.0 if (0.35 <= ar <= 3.2) else 0.5

    section_height = h / 40
    vertical_scores = []

    for i in range(40):
        section_y1 = int(i * section_height)
        section_y2 = int((i + 1) * section_height)

        section_mask = mask[section_y1:section_y2, :]
        section_area = float(section_mask.sum())

        if section_area > 0:
            score = section_area / (section_height * w)
            if i >= 38:
                score = -1
        else:
            score = 0

        vertical_scores.append(score)

    valid_scores = [score for score in vertical_scores if score > 0]
    position_score_vertical = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    final_position_score = position_score_vertical

    return (2.0 * fill + 0.6 * final_position_score) * ar_pen

def find_label_roi_from_edges(edges: np.ndarray, image_shape, debug: bool = False):
    h, w = image_shape[:2]
    e = edges.copy()

    if e.dtype != np.uint8:
        e = e.astype(np.uint8)
    if e.max() == 1:
        e = (e * 255).astype(np.uint8)

    e2 = e

    contours, _ = cv2.findContours(e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"reason": "no_contours"}

    candidates = []
    img_area = float(w * h)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area < img_area * 0.02:
            continue
        if area > img_area * 0.65:
            continue

        ar = cw / float(ch + 1e-6)
        if ar < 0.35 or ar > 3.2:
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0
        dist = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
        center_score = 1.0 - min(1.0, dist / (0.75 * (w ** 2 + h ** 2) ** 0.5))

        cnt_area = float(cv2.contourArea(cnt))
        rect_fill = cnt_area / (area + 1e-6)

        area_score = min(1.0, area / (img_area * 0.25))
        score = 0.45 * area_score + 0.35 * center_score + 0.20 * rect_fill

        candidates.append(((x, y, x + cw, y + ch), score, {"ar": ar, "rect_fill": rect_fill, "area": area}))

    if not candidates:
        return None, {"reason": "no_candidates_after_filter"}

    candidates.sort(key=lambda t: t[1], reverse=True)
    best_bbox, best_score, meta = candidates[0]

    pad = int(min(w, h) * 0.06)
    x1, y1, x2, y2 = clamp_bbox(*best_bbox, w, h, pad=pad)

    info = {
        "reason": "ok",
        "best_score": best_score,
        "meta": meta,
        "raw_bbox": best_bbox,
        "padded_bbox": (x1, y1, x2, y2),
        "num_candidates": len(candidates),
    }
    return (x1, y1, x2, y2), info

def offset_bbox(bbox, dx, dy):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

# -----------------------------
# Gemini AI: 컬러+보조 이미지로 bbox 받기
# -----------------------------
def detect_bbox_with_gemini(color_bgr: np.ndarray, assist_bgr: np.ndarray, original_url: str):
    logger = get_logger()

    if _gemini_client is None:
        if logger:
            logger.warning("gemini_client_not_available")
        return None, {"reason": "gemini_client_not_available"}

    h, w = color_bgr.shape[:2]

    def bgr_to_bytes(img):
        _, buffer = cv2.imencode(".png", img)
        return buffer.tobytes()

    instruction = f"""
    당신은 이미지 내의 라벨(Label)을 찾는 전문가입니다.
    제공된 컬러 이미지와 엣지(Edge) 보조 이미지를 참고하여, 물체 정면에 붙은 '라벨'의 좌표를 찾으세요.
    하이라이트(반사광)를 무시하고 라벨만 인식하세요
    워터마크(vivino)를 무시하고 라벨을 인식하십시오
    반드시 다음 JSON 형식으로만 답변하세요:
    {{
      "label_bboxes": [{{"x1": int, "y1": int, "x2": int, "y2": int}}],
      "notes": "short comment"
    }}
    이미지 크기: 너비={w}, 높이={h}
    """

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=instruction),
                        types.Part.from_bytes(
                            data=bgr_to_bytes(color_bgr),
                            mime_type="image/png"
                        ),
                        types.Part.from_bytes(
                            data=bgr_to_bytes(assist_bgr),
                            mime_type="image/png"
                        ),
                    ]
                )
            ]
        )
        resp_text = response.text

        data = safe_json_extract(resp_text)
        bboxes = data.get("label_bboxes", [])

        if not bboxes:
            if logger:
                logger.warning("gemini_no_bbox")
            return None, {"reason": "gemini_no_bbox", "response": resp_text}

        cand = []
        for b in bboxes:
            x1, y1, x2, y2 = clamp_bbox(b["x1"], b["y1"], b["x2"], b["y2"], w, h)
            cand.append((x1, y1, x2, y2))

        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in cand]
        best = cand[int(np.argmax(areas))]

        return best, {"response": resp_text, "parsed": data}

    except Exception as e:
        if logger:
            logger.exception(f"Gemini API 호출 중 오류 발생: {e}")
        else:
            print(f"Gemini API 호출 중 오류 발생: {e}")
        return None, {"reason": "gemini_error", "error": str(e)}

# -----------------------------
# SAM refine
# -----------------------------
def get_sam_predictor():
    logger = get_logger()

    predictor = getattr(_thread_local, "sam_predictor", None)
    if predictor is not None:
        return predictor

    if not USE_SAM:
        return None
    if not _sam_available:
        return None
    if not os.path.exists(SAM_CHECKPOINT):
        return None

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=_DEVICE)

    predictor = SamPredictor(sam)
    _thread_local.sam_predictor = predictor

    if logger:
        logger.info(f"[SAM] loaded in thread: type={SAM_MODEL_TYPE}, device={_DEVICE}, ckpt={SAM_CHECKPOINT}")
    else:
        print(f"[SAM] loaded in thread: type={SAM_MODEL_TYPE}, device={_DEVICE}, ckpt={SAM_CHECKPOINT}")
    return predictor

def refine_bbox_with_sam(bgr: np.ndarray, init_bbox):
    predictor = get_sam_predictor()
    if predictor is None:
        return None, {"reason": "sam_not_available"}

    h, w = bgr.shape[:2]
    bgr_no_glare = reduce_specular_glare(bgr)
    rgb = cv2.cvtColor(bgr_no_glare, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    x1, y1, x2, y2 = init_bbox
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True
    )

    best_idx = None
    best_val = -1e9

    for i in range(masks.shape[0]):
        m = masks[i].astype(np.uint8)
        val = float(scores[i]) + 0.8 * score_label_mask(m, w, h)
        if val > best_val:
            best_val = val
            best_idx = i

    if best_idx is None:
        return None, {"reason": "sam_no_valid_mask", "scores": scores.tolist()}

    mask = masks[best_idx].astype(np.uint8).copy()

    section_height = h / 40
    cutoff_section = 38
    cutoff_y = int(cutoff_section * section_height)
    mask[cutoff_y:, :] = 0

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, {
            "reason": "sam_empty_mask_after_cutoff",
            "scores": scores.tolist(),
            "best_idx": int(best_idx),
            "best_val": float(best_val),
            "mask": mask
        }

    rx1, rx2 = int(xs.min()), int(xs.max())
    ry1, ry2 = int(ys.min()), int(ys.max())

    rx1, ry1, rx2, ry2 = clamp_bbox(
        rx1, ry1, rx2, ry2, w, h,
        pad=int(min(w, h) * 0.005)
    )
    refined_bbox = (rx1, ry1, rx2, ry2)

    return refined_bbox, {
        "scores": scores.tolist(),
        "best_idx": int(best_idx),
        "best_val": float(best_val),
        "mask": mask
    }

def create_edge_assist_image(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 40, 140)

    assist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assist[edges > 0] = (255, 255, 255)

    return assist, edges, enhanced

def get_s3_client():
    client = getattr(_thread_local, "s3_client", None)
    if client is not None:
        return client

    client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    _thread_local.s3_client = client
    return client


def get_db_connection():
    connection = getattr(_thread_local, "db_connection", None)

    if connection is not None:
        try:
            connection.ping(reconnect=True)
            return connection
        except Exception:
            try:
                connection.close()
            except Exception:
                pass

    connection = pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=3306,
    )
    _thread_local.db_connection = connection
    return connection


def reset_thread_local_resources():
    connection = getattr(_thread_local, "db_connection", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass
        finally:
            delattr(_thread_local, "db_connection")

    if hasattr(_thread_local, "s3_client"):
        delattr(_thread_local, "s3_client")

def check_s3_permissions():
    logger = get_logger()
    try:
        s3 = get_s3_client()

        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{(S3_LABEL_PREFIX or 'label').strip('/')}/__permission_test__.txt",
            Body=b"permission test",
            ContentType="text/plain",
            ACL="public-read",
        )

        if logger:
            logger.info(f"[S3] bucket 접근 및 PutObject OK: {S3_BUCKET_NAME}")
        else:
            print(f"[S3] bucket 접근 및 PutObject OK: {S3_BUCKET_NAME}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"[S3] 권한 확인 실패: {e}")
        else:
            print(f"[S3] 권한 확인 실패: {e}")
        return False

def test_s3_connection():
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"[S3] 연결 성공: bucket={S3_BUCKET_NAME}, region={AWS_REGION}")
        return True
    except Exception as e:
        print(f"[S3] 연결 실패: {e}")
        return False

def test_s3_upload():
    try:
        s3 = get_s3_client()
        test_key = f"{(S3_LABEL_PREFIX or 'label').strip('/')}/__s3_connection_test__.txt"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=test_key,
            Body=b"test",
            ContentType="text/plain",
            ACL="public-read",
        )
        print(f"[S3] 업로드 테스트 성공: {test_key}")
        return True
    except Exception as e:
        print(f"[S3] 업로드 테스트 실패: {e}")
        return False

def extract_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    filename = os.path.basename(path)
    if not filename:
        raise ValueError(f"파일명을 추출할 수 없습니다: {url}")
    return filename

def make_label_s3_key(original_url: str) -> str:
    filename = extract_filename_from_url(original_url)
    prefix = (S3_LABEL_PREFIX or "label").strip("/")
    return f"{prefix}/{filename}"

def upload_bytes_to_s3(image_bytes: bytes, s3_key: str, content_type: str = "image/png") -> str | None:
    logger = get_logger()
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType=content_type,
            ACL="public-read",
        )
        return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except (BotoCoreError, ClientError, Exception) as e:
        if logger:
            logger.error(f"[S3] upload failed: {e}")
        else:
            print(f"[S3] upload failed: {e}")
        return None

def update_winelabel_crop(wine_id: int, image_url: str):
    logger = get_logger()
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "UPDATE wine SET winelabel_crop = %s WHERE id = %s"
            cursor.execute(sql, (image_url, wine_id))
        connection.commit()

        if logger:
            logger.info(f"[DB] updated wine.id={wine_id} -> {image_url}")
        else:
            print(f"[DB] updated wine.id={wine_id} -> {image_url}")

    except pymysql.MySQLError as e:
        if logger:
            logger.error(f"[DB] update failed: {e}")
        else:
            print(f"[DB] update failed: {e}")

def encode_bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()

# -----------------------------
# 디버그 저장
# -----------------------------
def save_debug_bundle(index: int,
                      bgr: np.ndarray,
                      assist: np.ndarray,
                      enhanced_gray: np.ndarray,
                      edges: np.ndarray,
                      clahe_image: np.ndarray,
                      url: str,
                      gemini_bbox,
                      refined_bbox,
                      gemini_text: str | None,
                      sam_mask: np.ndarray | None):
    os.makedirs(DEBUG_DIR, exist_ok=True)

    base = safe_filename(url)
    prefix = f"{index:03d}_{base}"

    overlay = bgr.copy()

    if gemini_bbox is not None:
        x1, y1, x2, y2 = gemini_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(overlay, "Gemini", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if refined_bbox is not None:
        x1, y1, x2, y2 = refined_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(overlay, "SAM", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_overlay.png"), overlay)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_clahe.png"), clahe_image)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_assist.png"), assist)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_edges.png"), edges)

    if sam_mask is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_mask.png"), (sam_mask * 255).astype(np.uint8))

    if gemini_text:
        with open(os.path.join(DEBUG_DIR, f"{prefix}_gemini_response.txt"), "w", encoding="utf-8") as f:
            f.write(gemini_text)

# -----------------------------
# 메인 엔트리
# -----------------------------
def crop_labels(wine_id: int, image: np.ndarray, data_url: str, original_url: str, index: int, debug: bool = True):
    logger = get_logger()

    if image is None or not hasattr(image, "shape"):
        raise ValueError("image must be a valid OpenCV image (numpy array).")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    out_idx = index + 1
    uploaded_s3_url = None
    uploaded_s3_key = None
    out_path = None

    start_total = time.time()

    preprocessed_image = preprocess_image(image)

    clahe_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    debug_clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

    t1 = time.time()
    assist_image, edges, enhanced_gray = create_edge_assist_image(preprocessed_image)
    if logger:
        logger.info(f"[TIME] wine_id={wine_id} edge_assist={time.time() - t1:.2f}s")

    t2 = time.time()
    gemini_bbox, info = detect_bbox_with_gemini(preprocessed_image, assist_image, data_url)
    if logger:
        logger.info(f"[TIME] wine_id={wine_id} gemini={time.time() - t2:.2f}s")

    gemini_text = info.get("response") if isinstance(info, dict) else None

    if gemini_bbox is None:
        if logger:
            logger.warning(f"[{out_idx:03d}] Gemini bbox not found wine_id={wine_id}")
            logger.info(f"[TIME] wine_id={wine_id} total={time.time() - start_total:.2f}s")
        else:
            print(f"[{out_idx:03d}] ❌ Gemini bbox not found")

        return {
            "s3_url": None,
            "s3_key": None,
            "local_path": None,
        }

    padding = 0
    x1, y1, x2, y2 = gemini_bbox
    x1, y1, x2, y2 = clamp_bbox(
        x1, y1, x2, y2 + padding,
        preprocessed_image.shape[1],
        preprocessed_image.shape[0]
    )

    refined_bbox = None
    sam_mask = None

    t3 = time.time()
    if USE_SAM:
        rb, sam_info = refine_bbox_with_sam(preprocessed_image, (x1, y1, x2, y2))
        if logger:
            logger.info(f"[TIME] wine_id={wine_id} sam={time.time() - t3:.2f}s")

        if rb is not None:
            refined_bbox = rb

        if isinstance(sam_info, dict) and sam_info.get("mask") is not None:
            sam_mask = sam_info["mask"]

            scores = []
            section_height = preprocessed_image.shape[0] / 40

            for sec_idx in range(40):
                if sec_idx >= 37:
                    score = -1e9
                else:
                    section_mask = np.zeros_like(sam_mask)
                    sy1 = int(sec_idx * section_height)
                    sy2 = int((sec_idx + 1) * section_height)
                    section_mask[sy1:sy2, :] = sam_mask[sy1:sy2, :]
                    score = score_label_mask(
                        section_mask,
                        preprocessed_image.shape[1],
                        preprocessed_image.shape[0]
                    )
                scores.append(score)
    else:
        if logger:
            logger.info(f"[TIME] wine_id={wine_id} sam=skipped")

    final_bbox = refined_bbox if refined_bbox is not None else (x1, y1, x2, y2)
    x1, y1, x2, y2 = final_bbox
    x1, y1, x2, y2 = clamp_bbox(
        x1, y1, x2, y2,
        preprocessed_image.shape[1],
        preprocessed_image.shape[0],
        pad=0
    )

    t4 = time.time()
    crop = image[y1:y2, x1:x2].copy()
    if logger:
        logger.info(f"[TIME] wine_id={wine_id} crop={time.time() - t4:.2f}s")

    if crop.size == 0:
        if logger:
            logger.warning(f"[{out_idx:03d}] Empty crop image wine_id={wine_id}")
            logger.info(f"[TIME] wine_id={wine_id} total={time.time() - start_total:.2f}s")
        else:
            print(f"[{out_idx:03d}] Empty crop image")

        return {
            "s3_url": None,
            "s3_key": None,
            "local_path": None,
        }

    out_path = None

    # S3 업로드
    try:
        uploaded_s3_key = make_label_s3_key(data_url)
        crop_bytes = encode_bgr_to_png_bytes(crop)

        # 기존 코드
        # uploaded_s3_url = upload_bytes_to_s3(
        #     image_bytes=crop_bytes,
        #     s3_key=uploaded_s3_key,
        #     content_type="image/png"
        # )

        # 수정 코드: 테스트 시 S3 업로드 막고 mock 성공 처리
        upload_started_at = time.time()
        if UPLOAD_TO_S3:
            uploaded_s3_url = upload_bytes_to_s3(
                image_bytes=crop_bytes,
                s3_key=uploaded_s3_key,
                content_type="image/png"
            )
        else:
            uploaded_s3_url = f"mock://{uploaded_s3_key}"
            if logger:
                logger.info(f"[{out_idx:03d}] S3 upload skipped (UPLOAD_TO_S3=0), mock url={uploaded_s3_url}")
            else:
                print(f"[{out_idx:03d}] S3 upload skipped (UPLOAD_TO_S3=0), mock url={uploaded_s3_url}")
        if logger:
            logger.info(f"[TIME] wine_id={wine_id} s3_upload={time.time() - upload_started_at:.2f}s")

        method = "sam" if refined_bbox is not None else "gemini"

        if uploaded_s3_url:
            if logger:
                logger.info(f"[{out_idx:03d}] | wine_id={wine_id} Uploaded label to S3: {uploaded_s3_url} ({method})")
            else:
                print(f"[{out_idx:03d}] | wine_id={wine_id} Uploaded label to S3: {uploaded_s3_url} ({method})")

            json_value = json.dumps({original_url: uploaded_s3_key}, ensure_ascii=False)

            # 기존 코드
            # update_winelabel_crop(wine_id, json_value)

            # 수정 코드: 테스트 시 DB 저장 막기
            if SAVE_TO_DB:
                db_update_started_at = time.time()
                update_winelabel_crop(wine_id, json_value)
                if logger:
                    logger.info(f"[TIME] wine_id={wine_id} db_update={time.time() - db_update_started_at:.2f}s")
                if logger:
                    logger.info(f"[{out_idx:03d}] DB updated")
                else:
                    print(f"[{out_idx:03d}] DB updated")
            else:
                if logger:
                    logger.info(f"[{out_idx:03d}] DB update skipped (SAVE_TO_DB=0)")
                else:
                    print(f"[{out_idx:03d}] DB update skipped (SAVE_TO_DB=0)")
        else:
            if logger:
                logger.warning(f"[{out_idx:03d}] S3 upload failed")
            else:
                print(f"[{out_idx:03d}] S3 upload failed")

    except Exception as e:
        if logger:
            logger.exception(f"[{out_idx:03d}] S3 upload error: {e}")
        else:
            print(f"[{out_idx:03d}] S3 upload error: {e}")

    if logger:
        logger.info(f"[TIME] wine_id={wine_id} total={time.time() - start_total:.2f}s")

    # 디버그 저장 비활성화
    # if debug:
    #     save_debug_bundle(
    #         index=out_idx,
    #         bgr=image,
    #         assist=assist_image,
    #         enhanced_gray=enhanced_gray,
    #         edges=edges,
    #         clahe_image=debug_clahe_image,
    #         url=data_url,
    #         gemini_bbox=gemini_bbox,
    #         refined_bbox=refined_bbox,
    #         gemini_text=gemini_text,
    #         sam_mask=sam_mask
    #     )

    return {
        "s3_url": uploaded_s3_url,
        "s3_key": uploaded_s3_key,
        "local_path": out_path,
    }
