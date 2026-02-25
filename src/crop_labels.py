import os
import re
import json
import base64
import cv2
import numpy as np

# -----------------------------
# 환경 설정
# -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SAM 사용 여부 및 체크포인트
USE_SAM = os.getenv("USE_SAM", "1") == "1"
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")  # vit_b / vit_l / vit_h
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth")  # 상대/절대 경로 모두 가능

# -----------------------------
# OpenAI client
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
# 경로 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(BASE_DIR, "out")
DEBUG_DIR = os.path.join(BASE_DIR, "out_debug")
OUT_LABEL_TEMPLATE = "cropped_label{:03d}.png"


# -----------------------------
# 유틸
# -----------------------------
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
# 하이라이트(반사) 제거: Specular Glare Reduction
# -----------------------------
def reduce_specular_glare(bgr: np.ndarray) -> np.ndarray:
    """
    밝고(V↑) 채도 낮은(S↓) 하이라이트 영역을 마스크로 잡아 inpaint로 메운다.
    목적: 반사/조명으로 생기는 강한 edge를 줄여 OpenAI bbox + SAM 마스크 안정화.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 하이라이트 후보: 밝고(V↑) 채도 낮음(S↓)
    glare = cv2.inRange(hsv, (0, 0, 210), (180, 60, 255))

    # 조금 부드럽게 확장/정리
    k = np.ones((5, 5), np.uint8)
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, k, iterations=1)
    glare = cv2.dilate(glare, k, iterations=1)

    # inpaint로 하이라이트 제거
    out = cv2.inpaint(bgr, glare, 3, cv2.INPAINT_TELEA)
    return out


# -----------------------------
# ✅ SAM 멀티마스크 중 "라벨다운" 마스크 선택을 위한 스코어링
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

    # 비율 기반 점수
    ar = bw / float(bh + 1e-6)
    ar_pen = 1.0 if (0.35 <= ar <= 3.2) else 0.5

    # y1에 기반한 위치 점수
    position_score = 1.0 - min(1.0, y1 / (0.75 * h))

    return (2.0 * fill + 0.6 * position_score) * ar_pen

# -----------------------------
# Edge assist image 생성
# -----------------------------
def create_edge_assist_image(bgr: np.ndarray):
    """
    Gray 기반 + edge를 흰색으로 강조한 '보조 이미지' 생성.
    ✅ glare 제거를 먼저 적용해서(반사/조명 edge 감소) assist/edges가 라벨 윤곽에 더 집중하도록 한다.
    """
    bgr_no_glare = reduce_specular_glare(bgr)

    gray = cv2.cvtColor(bgr_no_glare, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    edges = cv2.Canny(enhanced, 40, 140)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    assist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assist[edges > 0] = (255, 255, 255)

    return assist, edges, enhanced


def find_label_roi_from_edges(edges: np.ndarray, image_shape, debug: bool = False):
    h, w = image_shape[:2]

    e = edges.copy()
    if e.dtype != np.uint8:
        e = e.astype(np.uint8)
    if e.max() == 1:
        e = (e * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    e2 = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel, iterations=1)
    e2 = cv2.dilate(e2, kernel, iterations=1)

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
# OpenAI Vision: 컬러+보조 이미지로 bbox 받기
# -----------------------------
def detect_bbox_with_openai(color_bgr: np.ndarray, assist_bgr: np.ndarray, original_url: str):
    if _openai_client is None:
        return None, {"reason": "openai_client_not_available"}

    h, w = color_bgr.shape[:2]

    color_data_url = encode_bgr_to_data_url_png(color_bgr)
    assist_data_url = encode_bgr_to_data_url_png(assist_bgr)

    instruction = f"""
You are given TWO images:

(1) Original COLOR image of a wine bottle.
(2) Edge-enhanced GRAYSCALE assist image, where label boundaries are emphasized.

Task:
Find the front paper label on the bottle body.

Use BOTH images:
- Use the assist image to locate precise label edges.
- Use the color image to understand what is actually the paper label (not glass/background).

STRICT:
- Do NOT include bottle glass.
- Do NOT include bottle neck/capsule.
- Do NOT include background.
- Return a TIGHT bounding box around the paper label only.

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
                {"type": "input_image", "image_url": color_data_url},
                {"type": "input_image", "image_url": assist_data_url},
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

    data = safe_json_extract(text)
    bboxes = data.get("label_bboxes", [])

    if not isinstance(bboxes, list) or len(bboxes) == 0:
        return None, {"reason": "openai_no_bbox", "response": text, "parsed": data}

    cand = []
    for b in bboxes:
        if not all(k in b for k in ("x1", "y1", "x2", "y2")):
            continue
        x1, y1, x2, y2 = clamp_bbox(
            b["x1"], b["y1"], b["x2"], b["y2"],
            w, h,
            pad=int(min(w, h) * 0.01)
        )
        cand.append((x1, y1, x2, y2))

    if not cand:
        return None, {"reason": "openai_bbox_invalid", "response": text, "parsed": data}

    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in cand]
    best = cand[int(np.argmax(areas))]

    return best, {"response": text, "parsed": data, "source_url": original_url}


# -----------------------------
# SAM refine: bbox -> mask -> tighter bbox
# -----------------------------
def get_sam_predictor():
    global _sam_predictor

    if _sam_predictor is not None:
        return _sam_predictor

    if not USE_SAM:
        return None
    if not _sam_available:
        return None

    ckpt = SAM_CHECKPOINT
    if not os.path.isabs(ckpt):
        ckpt_candidate = os.path.join(BASE_DIR, ckpt)
        if os.path.exists(ckpt_candidate):
            ckpt = ckpt_candidate

    if not os.path.exists(ckpt):
        return None

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt)
    sam.to(device=_DEVICE)
    _sam_predictor = SamPredictor(sam)
    print(f"[SAM] loaded: type={SAM_MODEL_TYPE}, device={_DEVICE}, ckpt={ckpt}")
    return _sam_predictor


def refine_bbox_with_sam(bgr: np.ndarray, init_bbox):
    predictor = get_sam_predictor()
    if predictor is None:
        return None, {"reason": "sam_not_available"}

    h, w = bgr.shape[:2]

    # SAM에도 glare 제거 이미지를 쓰면 반사로 인한 오인식이 줄어듦
    bgr_no_glare = reduce_specular_glare(bgr)

    rgb = cv2.cvtColor(bgr_no_glare, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    x1, y1, x2, y2 = init_bbox
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True
    )

    # ✅ 개선: score + "라벨다운 형태" 점수로 best mask 선택
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

    mask = masks[best_idx].astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, {
            "reason": "sam_empty_mask",
            "scores": scores.tolist(),
            "best_idx": int(best_idx),
            "best_val": float(best_val),
            "mask": mask
        }

    rx1, rx2 = int(xs.min()), int(xs.max())
    ry1, ry2 = int(ys.min()), int(ys.max())

    rx1, ry1, rx2, ry2 = clamp_bbox(rx1, ry1, rx2, ry2, w, h, pad=int(min(w, h) * 0.005))
    refined_bbox = (rx1, ry1, rx2, ry2)

    return refined_bbox, {
        "scores": scores.tolist(),
        "best_idx": int(best_idx),
        "best_val": float(best_val),
        "mask": mask
    }


# -----------------------------
# 디버그 저장
# -----------------------------
def save_debug_bundle(index: int,
                      bgr: np.ndarray,
                      assist: np.ndarray,
                      enhanced_gray: np.ndarray,
                      edges: np.ndarray,
                      url: str,
                      openai_bbox,
                      refined_bbox,
                      openai_text: str | None,
                      sam_mask: np.ndarray | None):
    os.makedirs(DEBUG_DIR, exist_ok=True)

    base = safe_filename(url)
    prefix = f"{index:03d}_{base}"

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_orig.png"), bgr)

    bgr_no_glare = reduce_specular_glare(bgr)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_no_glare.png"), bgr_no_glare)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_assist.png"), assist)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_enhanced_gray.png"), enhanced_gray)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_edges.png"), edges)

    overlay = bgr.copy()

    if openai_bbox is not None:
        x1, y1, x2, y2 = openai_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(overlay, "openai", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    if refined_bbox is not None:
        x1, y1, x2, y2 = refined_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(overlay, "sam", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_overlay.png"), overlay)

    if sam_mask is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_mask.png"), (sam_mask * 255).astype(np.uint8))

        mask_overlay = bgr.copy()
        m = sam_mask.astype(bool)
        mask_overlay[m] = cv2.addWeighted(mask_overlay[m], 0.3, np.full_like(mask_overlay[m], 255), 0.7, 0)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_overlay.png"), mask_overlay)

    if openai_text:
        with open(os.path.join(DEBUG_DIR, f"{prefix}_openai_response.txt"), "w", encoding="utf-8") as f:
            f.write(openai_text)


# -----------------------------
# 메인 엔트리
# -----------------------------
def crop_labels(image: np.ndarray, data_url: str, index: int, debug: bool = True):
    if image is None or not hasattr(image, "shape"):
        raise ValueError("image must be a valid OpenCV image (numpy array).")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    out_idx = index + 1

    # 전체 파이프라인에서 glare 제거
    image_no_glare = reduce_specular_glare(image)

    # assist 이미지 생성
    assist, edges, enhanced_gray = create_edge_assist_image(image_no_glare)

    # 초기 ROI 찾기
    roi_bbox, roi_info = find_label_roi_from_edges(edges, image_no_glare.shape, debug=debug)

    if roi_bbox is None:
        openai_bbox, info = detect_bbox_with_openai(image_no_glare, assist, data_url)
        openai_text = info.get("response") if isinstance(info, dict) else None
        roi_used = False
    else:
        rx1, ry1, rx2, ry2 = roi_bbox
        roi_color = image_no_glare[ry1:ry2, rx1:rx2].copy()
        roi_assist = assist[ry1:ry2, rx1:rx2].copy()

        openai_bbox_roi, info = detect_bbox_with_openai(roi_color, roi_assist, data_url)
        openai_text = info.get("response") if isinstance(info, dict) else None

        openai_bbox = offset_bbox(openai_bbox_roi, dx=rx1, dy=ry1) if openai_bbox_roi else None
        roi_used = True

    if openai_bbox is None:
        print(f"[{out_idx:03d}] ❌ OpenAI bbox not found")
        return

    # SAM refine
    refined_bbox = None
    sam_mask = None

    if USE_SAM:
        rb, sam_info = refine_bbox_with_sam(image_no_glare, openai_bbox)
        if rb is not None:
            refined_bbox = rb
        if isinstance(sam_info, dict) and sam_info.get("mask") is not None:
            sam_mask = sam_info["mask"]

    # 초기 크롭 박스 설정 (30% 여유)
    final_bbox = refined_bbox if refined_bbox is not None else openai_bbox
    pad = int(min(image.shape[:2]) * 0.25)  # 여유를 30%로 설정
    x1, y1, x2, y2 = final_bbox
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, image.shape[1], image.shape[0], pad=pad)

    # 크롭 및 저장
    crop = image[y1:y2, x1:x2].copy()
    out_path = os.path.join(OUT_DIR, OUT_LABEL_TEMPLATE.format(out_idx))
    
    if cv2.imwrite(out_path, crop):
        print(f"[{out_idx:03d}] ✅ Saved label crop: {out_path} (initial crop with padding)")
    else:
        print(f"[{out_idx:03d}] ❌ Failed to save label crop.")

    # 세밀한 조정을 위해 반복
    for i in range(2):  # 2회 반복하여 크롭 범위를 줄임
        # 여유를 줄여서 새로운 크롭 박스 설정
        pad = int(min(image.shape[:2]) * 0.1)  # 여유를 10%로 설정
        x1, y1, x2, y2 = final_bbox
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, image.shape[1], image.shape[0], pad=pad)

        crop = image[y1:y2, x1:x2].copy()
        out_path = os.path.join(OUT_DIR, OUT_LABEL_TEMPLATE.format(out_idx))

        if cv2.imwrite(out_path, crop):
            print(f"[{out_idx:03d}] ✅ Saved refined label crop: {out_path} (iteration {i+1})")
            break  # 성공적으로 저장하면 반복 중지
    else:
        print(f"[{out_idx:03d}] ❌ Failed to save refined label crop.")

    if debug:
        save_debug_bundle(index=out_idx, bgr=image, assist=assist, enhanced_gray=enhanced_gray,
                          edges=edges, url=data_url, openai_bbox=openai_bbox,
                          refined_bbox=refined_bbox, openai_text=openai_text, sam_mask=sam_mask)