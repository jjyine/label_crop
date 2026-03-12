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
import boto3
from urllib.parse import urlparse
from botocore.exceptions import BotoCoreError, ClientError

# -----------------------------
# ENV / 설정
# -----------------------------
load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_LABEL_PREFIX = os.getenv("S3_LABEL_PREFIX")

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
# Google Generative AI client
# -----------------------------
try:
    from google.genai import Client  # Google Generative AI 클라이언트 임포트ㅐ
    from google.genai import types
    _gemini_client = Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
    _gemini_client = None

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
def extract_image_area(image_url: str) -> int:
    m = re.search(r"(\d+)x(\d+)", image_url)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        return w * h

    m = re.search(r"x(\d+)", image_url)
    if m:
        size = int(m.group(1))
        return size * size

    return 0


def select_best_image(image_data):
    vivino_candidates = []
    other_candidates = []

    for image_url, s3_key in image_data.items():
        area = extract_image_area(image_url)
        item = (area, image_url, s3_key)

        if "vivino.com" in image_url:
            vivino_candidates.append(item)
        else:
            other_candidates.append(item)

    if vivino_candidates:
        _, selected_image, selected_key = max(vivino_candidates, key=lambda x: x[0])
        return selected_image, selected_key

    if other_candidates:
        _, selected_image, selected_key = max(other_candidates, key=lambda x: x[0])
        return selected_image, selected_key

    return None, None

def fetch_data(limit=1, total_results=None, start_id=380):
    connection = None
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=3306,
        )

        last_id = start_id
        fetched_results = 0

        while True:
            if total_results is not None and fetched_results >= total_results:
                break

            with connection.cursor() as cursor:
                sql = """
                    SELECT id, s3_images
                    FROM wine
                    WHERE s3_images IS NOT NULL
                      AND id > %s
                    ORDER BY id ASC
                    LIMIT %s
                """
                cursor.execute(sql, (last_id, limit))
                results = cursor.fetchall()

                if not results:
                    print("No more images to fetch.")
                    break

                for row in results:
                    wine_id = row[0]
                    raw_s3_images = row[1]

                    # 다음 페이지 기준 id 갱신
                    last_id = wine_id

                    # 1) s3_images JSON 파싱
                    try:
                        if not raw_s3_images or not str(raw_s3_images).strip():
                            print(f"[SKIP] wine.id={wine_id} - s3_images is empty")
                            continue

                        image_data = json.loads(raw_s3_images)

                        if not isinstance(image_data, dict) or not image_data:
                            print(f"[SKIP] wine.id={wine_id} - s3_images json is empty or not dict")
                            continue

                    except json.JSONDecodeError as e:
                        print(f"[SKIP] wine.id={wine_id} - invalid s3_images json: {e}")
                        continue
                    except Exception as e:
                        print(f"[SKIP] wine.id={wine_id} - failed to parse s3_images: {e}")
                        continue

                    # 2) 사용할 이미지 선택
                    try:
                        selected_url, selected_key = select_best_image(image_data)
                        if not selected_url or not selected_key:
                            print(f"[SKIP] wine.id={wine_id} - no selectable image found")
                            continue
                    except Exception as e:
                        print(f"[SKIP] wine.id={wine_id} - image selection failed: {e}")
                        continue

                    final_image_url = f"https://vin-social.s3.amazonaws.com/{selected_key}"
                    print("Fetching image from URL:", final_image_url)

                    # 3) 이미지 다운로드
                    try:
                        response = requests.get(final_image_url, timeout=15)
                        response.raise_for_status()
                    except requests.RequestException as e:
                        print(f"[SKIP] wine.id={wine_id} - image download failed: {e}")
                        continue
                    except Exception as e:
                        print(f"[SKIP] wine.id={wine_id} - unexpected download error: {e}")
                        continue

                    # 4) 이미지 열기 / OpenCV 변환
                    try:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"[SKIP] wine.id={wine_id} - image decode/convert failed: {e}")
                        continue

                    fetched_results += 1
                    yield wine_id, img_cv, final_image_url, selected_url

                    if total_results is not None and fetched_results >= total_results:
                        break

    except pymysql.MySQLError as e:
        print("DB connect failed:", e)
    except Exception as e:
        print("Unexpected fetch_data error:", e)
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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    assist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assist[edges > 0] = (255, 255, 255)

    return assist, edges, enhanced

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

def check_s3_permissions():
    try:
        s3 = get_s3_client()

        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"[S3] bucket 접근 OK: {S3_BUCKET_NAME}")

        test_key = f"{(S3_LABEL_PREFIX or 'label').strip('/')}/__permission_test__.txt"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=test_key,
            Body=b"permission test",
            ContentType="text/plain"
        )

        print(f"[S3] PutObject OK: {test_key}")
        return True

    except Exception as e:
        print(f"[S3] 권한 확인 실패: {e}")
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

def encode_bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()

def upload_bytes_to_s3(image_bytes: bytes, s3_key: str, content_type: str = "image/png") -> str | None:
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType=content_type,
        )
        return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except (BotoCoreError, ClientError, Exception) as e:
        print(f"[S3] upload failed: {e}")
        return None

def find_label_candidate_box_from_edges(edges: np.ndarray):
    h, w = edges.shape[:2]

    work = edges.copy()
    work = cv2.morphologyEx(edges,
                        cv2.MORPH_CLOSE,
                        np.ones((3,3), np.uint8),
                        iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (w * h) * 0.01:
            continue

        aspect = cw / max(1, ch)
        if not (0.55 <= aspect <= 3.8):
            continue

        cx = x + cw / 2
        cy = y + ch / 2
        center_bonus = 1.0 - (abs(cx - w / 2) / (w / 2)) * 0.7
        vertical_bonus = 1.0 - (abs(cy - h * 0.62) / (h * 0.62)) * 0.7

        if cy < h * 0.22:
            vertical_bonus *= 0.2

        score = area * max(0.05, center_bonus) * max(0.05, vertical_bonus)

        if score > best_score:
            best_score = score
            best = (x, y, x + cw, y + ch)

    if best is None:
        return None, work

    x1, y1, x2, y2 = best
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.12)
    x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, w, h, pad=0)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return np.array([x1, y1, x2, y2], dtype=np.int32), work

# -----------------------------
# 2) Google Generative AI -> bbox refine (candidate box 기반)
# -----------------------------
import time

def gemini_refine_bbox(color_bgr: np.ndarray, assist_bgr: np.ndarray, candidate_box_xyxy: np.ndarray | None):
    if _gemini_client is None:
        print("[DEBUG] Gemini client is not initialized.")
        return None, None

    h, w = color_bgr.shape[:2]
    
    # 1. 이미지를 Base64 데이터 URL로 인코딩
    color_data = encode_bgr_to_data_url_png(color_bgr)
    assist_data = encode_bgr_to_data_url_png(assist_bgr)

    cand_txt = "none"
    if candidate_box_xyxy is not None:
        x1, y1, x2, y2 = map(int, candidate_box_xyxy.tolist())
        cand_txt = f"[{x1}, {y1}, {x2}, {y2}]"

    instruction = f"""
    You are given TWO images: (1) Original COLOR, (2) Edge assist.
    Initial candidate box = {cand_txt}
    Return ONLY valid JSON with a tight bounding box around the wine label:
    {{"label_bboxes": [{{"x1": int, "y1": int, "x2": int, "y2": int}}]}}
    Image size: width={w}, height={h}
    """

    # 2. 재시도 로직 설정
    max_retries = 3
    wait_time = 5  # 429 에러 시 대기 시간 (초)

    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Calling Gemini API (Attempt {attempt+1})...")
            time.sleep(1.2)
            
            resp = _gemini_client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=[
                    instruction,
                    Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)),
                    Image.fromarray(cv2.cvtColor(assist_bgr, cv2.COLOR_BGR2RGB))
                ],      
                config=types.GenerateContentConfig(
                    temperature=0.0,  
                    response_mime_type="application/json" 
                )
            )
            
            # 응답 텍스트 추출
            resp_text = resp.text
            print("[DEBUG] Gemini API response received.")
            break  
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if attempt < max_retries - 1:
                    print(f"[RETRY] 쿼타 초과(429). {wait_time}초 후 다시 시도합")
                    time.sleep(wait_time)
                    wait_time *= 2 
                    continue
            
            print(f"[ERROR] Gemini API call failed: {e}")
            return None, None

    # 3. 결과 파싱
    try:
        data = safe_json_extract(resp_text)
        bboxes = data.get("label_bboxes", [])
        if not bboxes:
            return None, resp_text

        # 가장 큰 박스 선택 로직
        cand = []
        for b in bboxes:
            x1, y1, x2, y2 = clamp_xyxy(b["x1"], b["y1"], b["x2"], b["y2"], w, h)
            cand.append((x1, y1, x2, y2))
        
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in cand]
        best = cand[int(np.argmax(areas))]
        return best, resp_text

    except Exception as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        return None, resp_text
    
# -----------------------------
# 3) SAM refine (Google bbox 기반)
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
               gemini_bbox: tuple[int, int, int, int] | None,
               sam_bbox: tuple[int, int, int, int] | None,
               sam_mask: np.ndarray | None):
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

    if gemini_bbox is not None:
        x1, y1, x2, y2 = gemini_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 노랑: Google Generative AI
        cv2.putText(overlay, "gemini", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if sam_bbox is not None:
        x1, y1, x2, y2 = sam_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 초록: SAM
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

# -----------------------------
# edges -> Google Generative AI -> SAM 파이프라인
# -----------------------------
def detect_label_edges_gemini_sam(bgr: np.ndarray, url: str):
    assist, edges, enhanced = create_edge_assist_image(bgr)
    candidate_box, edges_work = find_label_candidate_box_from_edges(edges)

    edges_bbox = None
    if candidate_box is not None:
        x1, y1, x2, y2 = map(int, candidate_box.tolist())
        edges_bbox = (x1, y1, x2, y2)

    # 2) Google Generative AI refine (edges candidate hint)
    gemini_bbox, gemini_response = gemini_refine_bbox(bgr, assist, candidate_box)

    base_bbox = gemini_bbox if gemini_bbox is not None else edges_bbox

    # 3) SAM refine (base_bbox 기반)
    sam_bbox = None
    sam_mask = None
    if base_bbox is not None and USE_SAM:
        rb, mask = refine_bbox_with_sam(bgr, base_bbox)
        if rb is not None:
            sam_bbox = rb
        sam_mask = mask

    final_bbox = sam_bbox if sam_bbox is not None else base_bbox

    return final_bbox, {
        "assist": assist,
        "edges": edges,
        "edges_work": edges_work,
        "candidate_box": candidate_box,
        "gemini_bbox": gemini_bbox,
        "gemini_response": gemini_response,
        "sam_bbox": sam_bbox,
        "sam_mask": sam_mask,
    }

def update_winelabel_crop(wine_id: int, s3_key: str):
    connection = None
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=3306,
        )
        with connection.cursor() as cursor:
            sql = "UPDATE wine SET winelabel_crop = %s WHERE id = %s"
            cursor.execute(sql, (s3_key, wine_id))
        connection.commit()
        print(f"[DB] updated wine.id={wine_id} -> {s3_key}")
    except pymysql.MySQLError as e:
        print(f"[DB] update failed: {e}")
    finally:
        if connection:
            connection.close()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    print("[DEBUG] OUT_DIR:", OUT_DIR)
    print("[DEBUG] DEBUG_DIR:", DEBUG_DIR)
    print("[DEBUG] Google Generative AI:", "ON" if _gemini_client else "OFF")
    print("[DEBUG] SAM:", "ON" if (USE_SAM and _sam_available and os.path.exists(SAM_CHECKPOINT)) else "OFF")

    if not check_s3_permissions():
        raise RuntimeError("S3 권한 확인 실패")

    for i, (wine_id, img_cv, url, original_url) in enumerate(fetch_data(limit=1, total_results=10), start=1):
        final_bbox, dbg = detect_label_edges_gemini_sam(img_cv, url)

        # 디버그 항상 저장
        # save_debug(
        #     idx=i,
        #     url=url,
        #     bgr=img_cv,
        #     assist=dbg["assist"],
        #     edges=dbg["edges"],
        #     edges_work=dbg["edges_work"],
        #     candidate_box=dbg["candidate_box"],
        #     gemini_bbox=dbg["gemini_bbox"],
        #     sam_bbox=dbg["sam_bbox"],
        #     sam_mask=dbg["sam_mask"],
        # )

        if final_bbox is None:
            print(f"[{i:03d}] label not found:", url)
            continue

        crop = crop_by_xyxy(img_cv, final_bbox)

        if crop.size == 0:
            print(f"[{i:03d}] empty crop: {url}")
            continue

        s3_key = make_label_s3_key(url)
        crop_bytes = encode_bgr_to_png_bytes(crop)
        s3_url = upload_bytes_to_s3(crop_bytes, s3_key, content_type="image/png")

        method = "sam" if dbg["sam_bbox"] is not None else ("gemini" if dbg["gemini_bbox"] is not None else "edges")

        if s3_url:
            json_value = json.dumps({original_url: s3_key}, ensure_ascii=False)
            update_winelabel_crop(wine_id, json_value)
            print(f"[{i:03d}] uploaded: {s3_url} ({method})")
        else:
            print(f"[{i:03d}] S3 upload failed: {url}")

        time.sleep(1.0)