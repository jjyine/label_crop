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
# Edge assist image 생성
# -----------------------------
def create_edge_assist_image(bgr: np.ndarray):
    """
    원본 컬러를 유지하지 않고,
    Gray 기반 + edge를 흰색으로 강조한 '보조 이미지' 생성.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # edge
    edges = cv2.Canny(enhanced, 40, 140)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # gray -> 3채널로 만들고, edge 위치를 흰색으로 표시
    assist = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assist[edges > 0] = (255, 255, 255)

    return assist, edges, enhanced


def find_label_roi_from_edges(
    edges: np.ndarray,
    image_shape,
    debug: bool = False
):
    """
    edges(0/255)에서 라벨 후보 ROI bbox를 찾는다.
    반환: (x1,y1,x2,y2) or None, info dict
    """
    h, w = image_shape[:2]

    # edges는 0/255 또는 0/1일 수 있으니 0/255로 정규화
    e = edges.copy()
    if e.dtype != np.uint8:
        e = e.astype(np.uint8)
    if e.max() == 1:
        e = (e * 255).astype(np.uint8)

    # 노이즈 줄이기: close로 끊긴 선을 좀 잇고, dilate로 덩어리화
    kernel = np.ones((5, 5), np.uint8)
    e2 = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel, iterations=1)
    e2 = cv2.dilate(e2, kernel, iterations=1)

    # contour 추출
    contours, _ = cv2.findContours(e2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, {"reason": "no_contours"}

    # 후보 필터링 + 스코어링
    candidates = []
    img_area = float(w * h)

    # 라벨은 보통 병 몸통 중앙 근처에 있고, 너무 작지 않으며, 가로세로 비율이 극단적이지 않음
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area < img_area * 0.02:   # 너무 작은 것 제거 (2% 미만)
            continue
        if area > img_area * 0.65:   # 너무 큰 것 제거 (병 전체/배경일 가능성)
            continue

        ar = cw / float(ch + 1e-6)

        # 라벨 비율 가드 (상황 따라 조절)
        # 와인 라벨은 대체로 0.5~2.5 사이에 많이 있음 (세로형/가로형 모두)
        if ar < 0.35 or ar > 3.2:
            continue

        # 중앙성 점수: 라벨은 보통 중앙에 가까움
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        dist = ((cx - w/2.0)**2 + (cy - h/2.0)**2) ** 0.5
        center_score = 1.0 - min(1.0, dist / (0.75 * (w**2 + h**2) ** 0.5))

        # contour가 얼마나 "사각형에 가까운지": contour area / bbox area
        cnt_area = float(cv2.contourArea(cnt))
        rect_fill = cnt_area / (area + 1e-6)  # 1에 가까울수록 꽉 참

        # 최종 점수(가중치): 면적 + 중앙성 + 사각형성
        area_score = min(1.0, area / (img_area * 0.25))  # 25% 정도면 만점
        score = 0.45 * area_score + 0.35 * center_score + 0.20 * rect_fill

        candidates.append(((x, y, x+cw, y+ch), score, {"ar": ar, "rect_fill": rect_fill, "area": area}))

    if not candidates:
        return None, {"reason": "no_candidates_after_filter"}

    # 점수 최고 후보 선택
    candidates.sort(key=lambda t: t[1], reverse=True)
    best_bbox, best_score, meta = candidates[0]

    # ROI는 조금 여유롭게 확장 (라벨 경계 누락 방지)
    pad = int(min(w, h) * 0.06)  # 6% 정도 (원하면 0.03~0.10 조절)
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

    # output_text 우선, 없으면 방어적으로 추출
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

    # 여러개면 가장 큰 면적 선택
    cand = []
    for b in bboxes:
        if not all(k in b for k in ("x1", "y1", "x2", "y2")):
            continue
        x1, y1, x2, y2 = clamp_bbox(b["x1"], b["y1"], b["x2"], b["y2"], w, h, pad=int(min(w, h) * 0.01))
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

    # 체크포인트 경로: 상대경로면 프로젝트 루트 기준으로도 한번 확인
    ckpt = SAM_CHECKPOINT
    if not os.path.isabs(ckpt):
        # 프로젝트 루트 기준
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

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    x1, y1, x2, y2 = init_bbox
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, {"reason": "sam_empty_mask", "scores": scores.tolist(), "mask": mask}

    rx1, rx2 = int(xs.min()), int(xs.max())
    ry1, ry2 = int(ys.min()), int(ys.max())

    rx1, ry1, rx2, ry2 = clamp_bbox(rx1, ry1, rx2, ry2, w, h, pad=int(min(w, h) * 0.005))
    refined_bbox = (rx1, ry1, rx2, ry2)

    return refined_bbox, {"scores": scores.tolist(), "mask": mask}




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
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_assist.png"), assist)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_enhanced_gray.png"), enhanced_gray)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_edges.png"), edges)

    # overlay: openai bbox(노랑), sam refined(초록)
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

    # SAM mask 저장
    if sam_mask is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_mask.png"), (sam_mask * 255).astype(np.uint8))

        # mask overlay도 추가 저장
        mask_overlay = bgr.copy()
        m = sam_mask.astype(bool)
        # 마스크 영역 약간 밝게 표시
        mask_overlay[m] = cv2.addWeighted(mask_overlay[m], 0.3, np.full_like(mask_overlay[m], 255), 0.7, 0)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{prefix}_sam_overlay.png"), mask_overlay)

    # OpenAI 응답 원문 저장
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

    if debug and index == 0:
        print("[DEBUG] cwd:", os.getcwd())
        print("[DEBUG] OUT_DIR:", os.path.abspath(OUT_DIR))
        print("[DEBUG] DEBUG_DIR:", os.path.abspath(DEBUG_DIR))
        print("[DEBUG] OpenAI:", "ON" if _openai_client else "OFF")
        print("[DEBUG] SAM:", "ON" if (USE_SAM and _sam_available) else "OFF (or not installed)")

    # 1) assist 이미지 생성
    assist, edges, enhanced_gray = create_edge_assist_image(image)

    # ✅ 2) edge 기반 ROI 찾기
    roi_bbox, roi_info = find_label_roi_from_edges(edges, image.shape, debug=debug)

    # ROI를 못 찾으면 기존 방식(전체 이미지로 OpenAI) fallback
    if roi_bbox is None:
        openai_bbox, info = detect_bbox_with_openai(image, assist, data_url)
        openai_text = info.get("response") if isinstance(info, dict) else None
        roi_used = False
        roi_offset = (0, 0)
    else:
        rx1, ry1, rx2, ry2 = roi_bbox
        roi_color = image[ry1:ry2, rx1:rx2].copy()
        roi_assist = assist[ry1:ry2, rx1:rx2].copy()

        # ✅ OpenAI는 ROI 안에서만 bbox 찾기
        openai_bbox_roi, info = detect_bbox_with_openai(roi_color, roi_assist, data_url)
        openai_text = info.get("response") if isinstance(info, dict) else None

        # ROI 좌표계를 원본 좌표계로 복원
        openai_bbox = offset_bbox(openai_bbox_roi, dx=rx1, dy=ry1) if openai_bbox_roi else None
        roi_used = True
        roi_offset = (rx1, ry1)

    if openai_bbox is None:
        if debug:
            # 디버그 저장(ROI 정보는 텍스트로라도 남기고 싶으면 openai_text에 추가 가능)
            save_debug_bundle(
                index=out_idx,
                bgr=image,
                assist=assist,
                enhanced_gray=enhanced_gray,
                edges=edges,
                url=data_url,
                openai_bbox=None,
                refined_bbox=None,
                openai_text=openai_text,
                sam_mask=None
            )
        why = "ROI used but OpenAI bbox not found" if roi_used else "OpenAI bbox not found"
        print(f"[{out_idx:03d}] ❌ {why}. roi_info={roi_info if roi_bbox is not None else 'None'}")
        return

    # 3) SAM refine (가능하면) — 기존 그대로
    refined_bbox = None
    sam_mask = None

    if USE_SAM:
        rb, sam_info = refine_bbox_with_sam(image, openai_bbox)
        if rb is not None:
            refined_bbox = rb
        if isinstance(sam_info, dict) and sam_info.get("mask") is not None:
            sam_mask = sam_info["mask"]

    final_bbox = refined_bbox if refined_bbox is not None else openai_bbox

    # 4) crop + 저장 (원본 컬러)
    x1, y1, x2, y2 = final_bbox
    crop = image[y1:y2, x1:x2].copy()

    out_path = os.path.join(OUT_DIR, OUT_LABEL_TEMPLATE.format(out_idx))
    ok = cv2.imwrite(out_path, crop)

    if ok:
        method = "openai+sam" if refined_bbox is not None else "openai_only"
        method += "+roi" if roi_used else ""
        print(f"[{out_idx:03d}] ✅ Saved label crop: {out_path} ({method})")
    else:
        print(f"[{out_idx:03d}] ❌ Failed to save label crop: {out_path}")

    # 5) debug 저장 — 기존 그대로
    if debug:
        # openai_text에 ROI 정보를 조금 덧붙여 저장하면 분석 편함
        extra = ""
        if roi_bbox is not None:
            extra = f"\n\n[ROI]\nroi_bbox={roi_bbox}\nroi_info={roi_info}\nroi_offset={roi_offset}\n"
        save_debug_bundle(
            index=out_idx,
            bgr=image,
            assist=assist,
            enhanced_gray=enhanced_gray,
            edges=edges,
            url=data_url,
            openai_bbox=openai_bbox,
            refined_bbox=refined_bbox,
            openai_text=(openai_text + extra) if openai_text else extra,
            sam_mask=sam_mask
        )