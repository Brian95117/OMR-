# 啟動：
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -U pip setuptools wheel
#   pip install streamlit opencv-python numpy pandas openpyxl pdf2image pillow
#   (macOS) brew install poppler
#   streamlit run omr_ui.py

import os
import io
import zipfile
import tempfile
import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import cv2
import streamlit as st

# PDF 支援
try:
    from pdf2image import convert_from_path
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# 支援到 10 選（A~J）
LETTERS = list("ABCDE")

# ================= 基礎工具 =================


def ensure_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def normalize_answer_token(x: str) -> str:
    """標準化答案：允許多選 AC/BDE；NFKC→大寫→僅保留 A~J→去重排序。"""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).upper()
    letters = re.findall(r"[A-J]", s)
    if not letters:
        return ""
    return "".join(sorted(set(letters)))


def pdf_to_bgr_list(file_or_fp) -> List[np.ndarray]:
    if not HAS_PDF:
        raise RuntimeError("讀取 PDF 需要 pdf2image + poppler。")
    if hasattr(file_or_fp, "read"):
        file_or_fp.seek(0)
        # 優先使用 getbuffer 以提升相容性
        data = file_or_fp.getbuffer()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            pages = convert_from_path(tmp_path, dpi=300)
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
    else:
        pages = convert_from_path(str(file_or_fp), dpi=300)
    return [cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2BGR) for p in pages]


def imread_any(file_or_path) -> np.ndarray:
    """讀單一影像或 PDF 第一頁為 BGR numpy（模板或單張影像用）。"""
    if isinstance(file_or_path, np.ndarray):
        return file_or_path
    if hasattr(file_or_path, "read"):
        pos = file_or_path.tell()
        file_or_path.seek(0)
        data = np.frombuffer(file_or_path.read(), np.uint8)
        file_or_path.seek(pos)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            name = getattr(file_or_path, "name", "").lower()
            if name.endswith(".pdf"):
                pages = pdf_to_bgr_list(file_or_path)
                if not pages:
                    raise RuntimeError("PDF 轉影像失敗。")
                return pages[0]
            raise RuntimeError("無法讀取影像/檔案。")
        return img
    path = str(file_or_path).lower()
    ext = os.path.splitext(path)[1]
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
        img = cv2.imdecode(np.fromfile(
            file_or_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"無法讀取影像：{file_or_path}")
        return img
    if ext == ".pdf":
        pages = pdf_to_bgr_list(file_or_path)
        if not pages:
            raise RuntimeError(f"PDF 轉影像失敗：{file_or_path}")
        return pages[0]
    raise RuntimeError(f"不支援的檔案格式：{file_or_path}")


def apply_roi(img: np.ndarray, top_p: float, bottom_p: float, left_p: float, right_p: float):
    H, W = img.shape[:2]
    x0 = int(W * left_p)
    x1 = int(W * right_p)
    y0 = int(H * top_p)
    y1 = int(H * bottom_p)
    x0 = max(0, min(x0, W-1))
    x1 = max(1, min(x1, W))
    y0 = max(0, min(y0, H-1))
    y1 = max(1, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return img.copy(), (0, 0, W, H)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1-x0, y1-y0)


def rect_clip(x, y, w, h, W, H):
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W-x))
    h = max(1, min(h, H-y))
    return x, y, w, h

# ================= 圓圈偵測與排序 =================


def auto_detect_bubbles_from_template(template_img: np.ndarray,
                                      dp=1.2, minDist=18, param1=80, param2=20, minRadius=8, maxRadius=22):
    def hough_detect(img):
        gray = ensure_gray(img)
        gray = cv2.medianBlur(gray, 3)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )
        bboxes = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for (cx, cy, r) in circles:
                x, y = int(cx - r), int(cy - r)
                w = h = int(2*r)
                bboxes.append((x, y, w, h))
        return bboxes

    def contour_detect(img):
        gray = ensure_gray(img)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        _, th = cv2.threshold(
            eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(
            th, cv2.MORPH_OPEN, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)), 1
        )
        cnts, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        area_min = max(30, (minRadius*2)**2*0.15)
        area_max = (maxRadius*2)**2*1.2
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < area_min or area > area_max:
                continue
            peri = cv2.arcLength(c, True)
            circ = 0 if peri == 0 else 4.0*np.pi*(area/(peri*peri))
            ar = w/float(h) if h != 0 else 0
            if 0.75 <= ar <= 1.35 and circ >= 0.55:
                bboxes.append((x, y, w, h))
        return bboxes

    b = hough_detect(template_img)
    if len(b) < 5:
        b = contour_detect(template_img)
    if not b:
        raise RuntimeError("沒有偵測到圓圈；請調整參數或提高掃描解析度。")
    return b


def sort_bubbles_into_grid(bboxes, questions, choices):
    centers = [(x+w/2, y+h/2, (x, y, w, h)) for (x, y, w, h) in bboxes]
    centers.sort(key=lambda t: t[1])
    rows = []
    row_tol = None
    for c in centers:
        if not rows:
            rows.append([c])
            row_tol = None
            continue
        avg_y = float(np.mean([it[1] for it in rows[-1]]))
        tol = 8 if row_tol is None else row_tol
        if abs(c[1]-avg_y) <= tol:
            rows[-1].append(c)
            row_tol = 8
        else:
            rows.append([c])
            row_tol = None
    clean = [r for r in rows if 2 <= len(r) <= choices*2]
    for r in clean:
        r.sort(key=lambda t: t[0])
    question_rows = []
    for r in clean:
        groups, g = [], [r[0]]
        for i in range(1, len(r)):
            if abs(r[i][0]-r[i-1][0]) < 30:
                g.append(r[i])
            else:
                groups.append(g)
                g = [r[i]]
        groups.append(g)
        for g in groups:
            if len(g) >= choices:
                g = g[:choices]
                g.sort(key=lambda t: t[0])
                question_rows.append([t[2] for t in g])
    if len(question_rows) < questions:
        raise RuntimeError(f"偵測到的題列數不足：{len(question_rows)} < {questions}")
    return question_rows[:questions]

# ================= 對位與評分（單選/多選） =================


def feature_align(src_img: np.ndarray, dst_img: np.ndarray) -> np.ndarray:
    sgray, dgray = ensure_gray(src_img), ensure_gray(dst_img)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(sgray, None)
    kp2, des2 = orb.detectAndCompute(dgray, None)
    if des1 is None or des2 is None:
        return cv2.resize(src_img, (dgray.shape[1], dgray.shape[0]))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return cv2.resize(src_img, (dgray.shape[1], dgray.shape[0]))
    matches = sorted(matches, key=lambda x: x.distance)[:500]
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(src_img, (dgray.shape[1], dgray.shape[0]))
    return cv2.warpPerspective(src_img, H, (dgray.shape[1], dgray.shape[0]))


def score_sheet_single(aligned_img: np.ndarray, template: Dict[str, Any], fill_threshold: float = 0.72):
    gray = ensure_gray(aligned_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = 1.0 - (gray.astype(np.float32)/255.0)
    picks = []
    for row in template["bubbles"]:
        scores = []
        for (x, y, w, h) in row:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(inv.shape[1], x+w), min(inv.shape[0], y+h)
            roi = inv[y0:y1, x0:x1]
            s = float(np.mean(roi)) if roi.size else 0.0
            scores.append(s)
        if not scores:
            picks.append("-")
            continue
        best_i = int(np.argmax(scores))
        best = scores[best_i]
        if best < 0.15:
            picks.append("-")
            continue
        tmp = scores.copy()
        tmp.pop(best_i)
        second = max(tmp) if tmp else 0.0
        ratio = best/(second+1e-6) if second != 0 else 99.0
        picks.append(LETTERS[best_i] if ratio >=
                     fill_threshold and best_i < len(LETTERS) else "?")
    return picks


def score_sheet_multi(aligned_img: np.ndarray, template: Dict[str, Any],
                      abs_min: float = 0.15, rel_to_max: float = 0.6):
    gray = ensure_gray(aligned_img)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = 1.0 - (gray.astype(np.float32)/255.0)
    picks = []
    for row in template["bubbles"]:
        scores = []
        for (x, y, w, h) in row:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(inv.shape[1], x+w), min(inv.shape[0], y+h)
            roi = inv[y0:y1, x0:x1]
            s = float(np.mean(roi)) if roi.size else 0.0
            scores.append(s)
        if not scores:
            picks.append("")
            continue
        m = max(scores)
        chosen = []
        for i, sc in enumerate(scores):
            if sc >= abs_min and (m == 0 or sc >= m*rel_to_max):
                if i < len(LETTERS):
                    chosen.append(LETTERS[i])
        picks.append("".join(chosen))
    return picks


def grade_one(input_img_or_np, template_img: np.ndarray, template: Dict[str, Any],
              answer_key: List[str], fill_threshold: float = 0.72,
              allow_multi: bool = False, multi_abs_min: float = 0.15, multi_rel_max: float = 0.6,
              grade_policy: str = "全對給1分（少一個或多一個都0分）"):
    # 將學生 ROI 影像對齊到「模板 ROI 尺寸」
    src = input_img_or_np if isinstance(
        input_img_or_np, np.ndarray) else imread_any(input_img_or_np)
    aligned = feature_align(src, template_img)
    N = min(len(template.get("bubbles", [])), len(answer_key))
    sub_template = {**template, "bubbles": template["bubbles"][:N]}
    details = []
    total_score = 0.0

    if not allow_multi:
        picks = score_sheet_single(
            aligned, sub_template, fill_threshold=fill_threshold)
        for i in range(N):
            pick = normalize_answer_token(picks[i]) if i < len(picks) else ""
            ans = normalize_answer_token(
                answer_key[i]) if i < len(answer_key) else ""
            ok = int(pick == ans and ans != "")
            sc = float(ok)
            total_score += sc
            details.append({"Q": i+1, "Pick": pick or "-",
                           "Ans": ans or "-", "Correct": ok, "Score": sc})
    else:
        picks_multi = score_sheet_multi(
            aligned, sub_template, abs_min=multi_abs_min, rel_to_max=multi_rel_max)
        for i in range(N):
            pick = normalize_answer_token(
                picks_multi[i]) if i < len(picks_multi) else ""
            ans = normalize_answer_token(
                answer_key[i]) if i < len(answer_key) else ""
            p_set, a_set = set(pick), set(ans)
            if grade_policy.startswith("全對"):
                ok = int(p_set == a_set and len(a_set) > 0)
                sc = float(ok)
            else:
                k = max(1, len(a_set))
                hit = len(p_set & a_set)
                wrong = len(p_set - a_set)
                sc = max(0.0, min(1.0, (hit - wrong)/k))
                ok = int(sc >= 0.9999)
            total_score += sc
            details.append({"Q": i+1, "Pick": "".join(sorted(p_set)) or "-", "Ans": "".join(
                sorted(a_set)) or "-", "Correct": ok, "Score": round(sc, 4)})

    percent = round(100.0*total_score/max(1, N), 2)
    return {"detail": details, "score": round(total_score, 4), "total": N, "percent": percent}

# ================= 手動網格與總預覽 =================


def make_manual_grid(H: int, W: int, start_x: int, start_y: int, bubble_w: int, bubble_h: int,
                     dx: int, dy: int, n_rows: int, n_cols: int):
    bubbles = []
    for r in range(n_rows):
        row = []
        y = start_y + r*dy
        for c in range(n_cols):
            x = start_x + c*dx
            row.append((int(x), int(y), int(bubble_w), int(bubble_h)))
        bubbles.append(row)
    return bubbles


def draw_master_preview(full_img_bgr: np.ndarray, roi_box,
                        ans_bubbles_rel, cls_bubbles_rel, sid_bubbles_rel,
                        show_idx=True, show_roi=True, scale=1.0):
    colors = {"roi": (0, 0, 255), "b_ans": (0, 255, 0), "b_cls": (
        0, 200, 255), "b_sid": (255, 0, 255), "text": (50, 220, 50)}
    vis = full_img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    rx, ry, rw, rh = roi_box
    if show_roi and rw > 0 and rh > 0:
        cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), colors["roi"], 2)

    def draw_rel(img, bubbles_rel, offx, offy, color, prefix):
        if bubbles_rel is None:
            return
        for r, row in enumerate(bubbles_rel):
            for d, (x, y, w, h) in enumerate(row):
                X, Y = offx+int(x), offy+int(y)
                cv2.rectangle(img, (X, Y), (X+w, Y+h), color, 1)
                if show_idx:
                    cv2.putText(
                        img, f"{prefix}{r+1}:{d}", (X, max(0, Y-3)), font, 0.35, color, 1, cv2.LINE_AA)
    draw_rel(vis, ans_bubbles_rel, rx, ry, colors["b_ans"], "Q")
    draw_rel(vis, cls_bubbles_rel, rx, ry, colors["b_cls"], "C")
    draw_rel(vis, sid_bubbles_rel, rx, ry, colors["b_sid"], "S")
    cv2.putText(vis, "ROI=red  ANS=green  CLASS=yellow-blue  SID=magenta",
                (10, 18), font, 0.55, colors["text"], 2, cv2.LINE_AA)
    if scale != 1.0:
        H, W = vis.shape[:2]
        vis = cv2.resize(vis, (int(W*scale), int(H*scale)),
                         interpolation=cv2.INTER_AREA)
    return vis

# ================= 答案表讀取（自動 + 除錯） =================


def read_answers_auto(ans_file, vertical_only: bool = False) -> Tuple[List[str], Optional[pd.DataFrame]]:
    def last_valid(arr: List[str]) -> List[str]:
        # 去尾端空白
        last = -1
        for i, v in enumerate(arr):
            if re.fullmatch(r"[A-J]+", v or ""):
                last = i
        return arr[:last+1] if last >= 0 else []

    def try_df(df: pd.DataFrame) -> List[str]:
        df2 = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if df2.empty:
            return []
        cols_lower = [str(c).strip().lower() for c in df2.columns]

        def col_idx(names):
            for i, c in enumerate(cols_lower):
                if c in names:
                    return i
            return None
        q_idx = col_idx(["q", "題號", "question", "no", "index", "題目"])
        a_idx = col_idx(["answer", "ans", "答案"])
        # Q+Ans 兩欄
        if a_idx is not None and q_idx is not None:
            pairs = []
            for _, r in df2.iloc[:, [q_idx, a_idx]].iterrows():
                try:
                    qn = int(str(r.iloc[0]).strip())
                except:
                    continue
                a = normalize_answer_token(r.iloc[1])
                if qn >= 1:
                    pairs.append((qn, a))
            if pairs:
                max_q = max(q for q, _ in pairs)
                arr = [""]*max_q
                for q, a in pairs:
                    if 1 <= q <= max_q:
                        arr[q-1] = a
                return last_valid(arr)
        # 單欄縱向
        if df2.shape[1] >= 1:
            col0 = [normalize_answer_token(v) for v in df2.iloc[:, 0].tolist()]
            if any(col0):
                return last_valid(col0)
        if not vertical_only:
            # 橫向第一列
            if df2.shape[0] >= 1:
                row0 = [normalize_answer_token(v)
                        for v in df2.iloc[0, :].tolist()]
                if any(row0):
                    return last_valid(row0)
            # 橫向兩列（第一列題號、第二列答案）
            if df2.shape[0] >= 2:
                first = [str(v).strip() for v in df2.iloc[0, :].tolist()]
                second = [normalize_answer_token(v)
                          for v in df2.iloc[1, :].tolist()]
                pairs = []
                for i, t in enumerate(second):
                    try:
                        qn = int(first[i])
                    except:
                        qn = i+1
                    pairs.append((qn, t))
                if pairs:
                    max_q = max(q for q, _ in pairs)
                    arr = [""]*max_q
                    for q, a in pairs:
                        if 1 <= q <= max_q:
                            arr[q-1] = a
                    return last_valid(arr)
        return []

    debug_df = None
    if hasattr(ans_file, "read"):
        ans_file.seek(0)
        data = ans_file.getbuffer()
        xls = pd.ExcelFile(io.BytesIO(data))
    else:
        xls = pd.ExcelFile(ans_file)
    for sheet in xls.sheet_names:
        for header in [0, None]:
            try:
                df = xls.parse(sheet, header=header)
                if debug_df is None:
                    debug_df = df.copy()
                ans = try_df(df)
                if ans:
                    return ans, df
            except Exception:
                continue
    return [], debug_df


def read_excel_sheets(ans_file):
    if hasattr(ans_file, "read"):
        ans_file.seek(0)
        data = ans_file.getbuffer()
        xls = pd.ExcelFile(io.BytesIO(data))
    else:
        xls = pd.ExcelFile(ans_file)
    return xls, xls.sheet_names


def parse_answers_from_df(df: pd.DataFrame, has_header: bool, answer_col: int,
                          use_q_col: bool = False, q_col: int = 0) -> List[str]:
    # has_header 目前不使用，但保留介面相容
    df2 = df.copy()
    data = df2.iloc[:, answer_col].tolist()
    if use_q_col:
        qseries = df2.iloc[:, q_col].tolist()
        pairs = []
        for q, a in zip(qseries, data):
            try:
                qn = int(str(q).strip())
            except:
                continue
            tok = normalize_answer_token(a)
            pairs.append((qn, tok))
        if not pairs:
            return []
        max_q = max(q for q, _ in pairs)
        arr = [""]*max_q
        for q, a in pairs:
            if 1 <= q <= max_q:
                arr[q-1] = a
        return arr
    else:
        return [normalize_answer_token(v) for v in data]

# ================= 學號→姓名 對照讀取 =================


def read_sid_name_map(xlsx_file) -> Dict[str, str]:
    """
    從 .xlsx 萃取 {student_id(str) -> name(str)} 對照。
    """
    import io as _io
    import pandas as _pd

    def _norm_sid(x) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        if re.fullmatch(r"\d+\.0", s):
            s = s[:-2]
        s = s.replace(" ", "").replace("\u3000", "")
        return s

    def _pick_cols(df: _pd.DataFrame) -> Optional[Tuple[int, int]]:
        cols_lower = [str(c).strip().lower() for c in df.columns]
        sid_names = ["student_id", "sid", "id", "學號"]
        name_names = ["name", "student_name", "姓名"]
        sid_idx = next((i for i, c in enumerate(
            cols_lower) if c in sid_names), None)
        name_idx = next((i for i, c in enumerate(
            cols_lower) if c in name_names), None)
        if sid_idx is not None and name_idx is not None:
            return sid_idx, name_idx
        return None

    if hasattr(xlsx_file, "read"):
        xlsx_file.seek(0)
        data = xlsx_file.getbuffer()
        xls = _pd.ExcelFile(_io.BytesIO(data))
    else:
        xls = _pd.ExcelFile(xlsx_file)

    mapping: Dict[str, str] = {}
    for sheet in xls.sheet_names:
        for header in [0, None]:
            try:
                df = xls.parse(sheet, header=header, dtype=str)
                df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
                if df.empty or df.shape[1] < 2:
                    continue

                pair_idx = _pick_cols(df) if header == 0 else (0, 1)
                if pair_idx is None:
                    pair_idx = (0, 1)

                sid_col, name_col = pair_idx
                for sid_raw, name_raw in zip(df.iloc[:, sid_col], df.iloc[:, name_col]):
                    sid = _norm_sid(sid_raw)
                    name = "" if name_raw is None else str(name_raw).strip()
                    if sid:
                        mapping[sid] = name
            except Exception:
                continue
    return mapping


def render_cc_footer():
    st.markdown(
        """
        <style>
        /* 固定貼底的頁尾條 */
        .cc-footer {
          position: fixed;
          left: 0; right: 0; bottom: 0;
          padding: 8px 14px;
          font-size: 13px;
          line-height: 1.6;
          border-top: 1px solid rgba(0,0,0,.1);
          background: rgba(255,255,255,.95);
          z-index: 9999;
        }
        /* 暗色模式的邊線/背景微調 */
        @media (prefers-color-scheme: dark) {
          .cc-footer { 
            background: rgba(20,20,20,.92);
            border-top: 1px solid rgba(255,255,255,.15);
          }
        }
        /* 讓頁面內容不要被蓋住（預留頁尾高度） */
        .block-container { padding-bottom: 60px; }
        </style>

        <div class="cc-footer">
          <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank">
            <img alt="Creative Commons License" style="border-width:0;vertical-align:middle;height:22px"
                 src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png">
          </a>
          <span style="margin-left:8px;">
            <strong>omr system</strong> © 2025 pchen — 版權所有
            <a href="https://pchen.info/" target="_blank">pchen</a>
            is licensed under
            <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank">
              CC BY-NC-ND 4.0
            </a>.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================= Streamlit UI =================
st.set_page_config(
    page_title="OMR 自動閱卷",
    layout="wide",
    initial_sidebar_state="collapsed"  # ✅ 預設收起側邊欄
)
st.title("📄 OMR（電腦卡）自動閱卷")

# （可選）柔和樣式：讓側邊欄間距與邊框更清爽
st.markdown("""
<style>
h1, h2, h3 { margin-bottom: .3rem; }
[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
[data-testid="stSidebar"] .stExpander { border-radius: 10px; }
[data-testid="stSidebar"] .stExpander > details {
  border: 1px solid rgba(49,51,63,0.2);
  padding: 0.25rem 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---- 百分比→比例 的讀值工具（避免覆寫 widget state） ----


def get_pct(key: str, default_pct: float) -> float:
    """回傳以百分比表示的數值（0~100），若不存在回傳預設值。"""
    return float(st.session_state.get(key, default_pct))


def roi_bounds_from_state():
    """從 *_pct keys 取出 ROI，轉成 0~1 比例"""
    top = get_pct("k_roi_top_pct", 5) / 100.0
    bottom = get_pct("k_roi_bottom_pct", 95) / 100.0
    left = get_pct("k_roi_left_pct", 5) / 100.0
    right = get_pct("k_roi_right_pct", 95) / 100.0
    return top, bottom, left, right


def master_scale_from_state():
    """從 *_pct 取出預覽縮放，轉 0~1"""
    return get_pct("k_s_master_pct", 100) / 100.0

# ---- 以函式建側邊欄：分群折疊，沿用原本 state，但用 *_pct keys ----


def build_sidebar():
    with st.sidebar:
        st.caption("🧩 進階設定（點開區塊調整）")

        # 共用 ROI（使用 *_pct keys；不覆寫 session_state）
        with st.expander("📐 共用 ROI（%）（紅框）", expanded=False):
            st.slider("上界(%)", 0, 49, 5, 1, key="k_roi_top_pct")
            st.slider("下界(%)", 51, 100, 95, 1, key="k_roi_bottom_pct")
            st.slider("左界(%)", 0, 49, 5, 1, key="k_roi_left_pct")
            st.slider("右界(%)", 51, 100, 95, 1, key="k_roi_right_pct")

        # 答案表讀取偏好
        with st.expander("📑 答案表讀取偏好", expanded=False):
            st.checkbox("只讀直欄（忽略橫向）", value=True, key="k_ans_vertical_only")
            st.checkbox("開啟『答案表除錯模式』", value=True, key="k_ans_debug_mode")

        # 答案格
        with st.expander("🫧 答案格", expanded=False):
            st.radio("建立方式", ["自動偵測", "手動訓練"], index=1, key="k_ans_mode_radio")
            st.slider("單選：填黑優勢比門檻", 0.5, 1.5, 0.72, 0.01, key="k_fill")

            if st.session_state["k_ans_mode_radio"] == "自動偵測":
                st.number_input("題數", 1, 500, 100, 1, key="k_ans_q")
                st.number_input("每題選項數", 2, 10, 5, 1, key="k_ans_c")
                st.number_input("Hough dp", 1.0, 3.0, 1.2, 0.1, key="k_h_dp")
                st.number_input("Hough minDist", 5.0, 60.0,
                                18.0, 1.0, key="k_h_minDist")
                st.number_input("Hough param1", 10.0, 300.0,
                                80.0, 1.0, key="k_h_p1")
                st.number_input("Hough param2", 5.0, 100.0,
                                20.0, 1.0, key="k_h_p2")
                st.number_input("Hough minRadius", 3,
                                100, 8, 1, key="k_h_rmin")
                st.number_input("Hough maxRadius", 5, 120,
                                24, 1, key="k_h_rmax")
            else:
                st.number_input("欄數（columns）", 1, 6, 4, 1, key="k_man_cols")
                st.number_input("每欄題數", 1, 300, 25, 1, key="k_man_qpc")
                st.number_input("每題選項數", 2, 10, 5, 1, key="k_man_choices")
                st.number_input("起始 X（答案）", 0, 5000, 153, 1, key="k_man_ax")
                st.number_input("起始 Y（答案）", 0, 5000, 1101, 1, key="k_man_ay")
                st.number_input("泡泡寬（答案）", 4, 200, 50, 1, key="k_man_aw")
                st.number_input("泡泡高（答案）", 4, 200, 50, 1, key="k_man_ah")
                st.number_input("同題水平間距（答案）", 5, 400, 78, 1, key="k_man_adx")
                st.number_input("題目垂直間距（答案）", 5, 400, 79, 1, key="k_man_ady")
                st.number_input("欄與欄水平間距（答案）", 10, 2000,
                                567, 1, key="k_man_acdx")

        # 多選題設定
        with st.expander("✅ 多選題設定", expanded=False):
            st.checkbox("允許多選題（答案可含多個字母）", value=True, key="k_allow_multi")
            st.slider("多選：每泡泡最低填黑（絕對）", 0.05, 0.6,
                      0.15, 0.01, key="k_multi_abs")
            st.slider("多選：相對題內最高比率", 0.3, 1.0, 0.6, 0.01, key="k_multi_rel")
            st.selectbox(
                "多選評分方式",
                ["全對給1分（少一個或多一個都0分）", "部分給分：對一個給 1/k；選錯扣 1/k（不低於0）"],
                index=0, key="k_grade_policy"
            )

        # 班級（在 ROI 內）
        with st.expander("🏫 班級（在 ROI 內）", expanded=False):
            st.number_input("位數（rows）", 1, 6, 2, 1, key="k_cls_rows")
            st.number_input("起始 X", 0, 5000, 575, 1, key="k_cls_x")
            st.number_input("起始 Y", 0, 5000, 337, 1, key="k_cls_y")
            st.number_input("泡泡寬", 4, 200, 50, 1, key="k_cls_w")
            st.number_input("泡泡高", 4, 200, 50, 1, key="k_cls_h")
            st.number_input("同位數水平間距", 5, 600, 73, 1, key="k_cls_dx")
            st.number_input("位數垂直間距", 5, 600, 84, 1, key="k_cls_dy")

        # 學號（在 ROI 內）
        with st.expander("🆔 學號（在 ROI 內）", expanded=False):
            st.number_input("位數（rows）", 1, 6, 2, 1, key="k_sid_rows")
            st.number_input("起始 X", 0, 5000, 575, 1, key="k_sid_x")
            st.number_input("起始 Y", 0, 5000, 567, 1, key="k_sid_y")
            st.number_input("泡泡寬", 4, 200, 50, 1, key="k_sid_w")
            st.number_input("泡泡高", 4, 200, 50, 1, key="k_sid_h")
            st.number_input("同位數水平間距", 5, 600, 73, 1, key="k_sid_dx")
            st.number_input("位數垂直間距", 5, 600, 84, 1, key="k_sid_dy")

        # 總預覽顯示（使用 *_pct key）
        with st.expander("🖼️ 總預覽顯示", expanded=False):
            st.slider("總預覽寬度(px)", 400, 2000, 1100, 10, key="k_w_master")
            st.slider("總預覽縮放（%）", 10, 300, 100, 5, key="k_s_master_pct")
            st.checkbox("顯示 ROI 邊框（紅）", value=True, key="k_show_roi")


# 建立側邊欄
build_sidebar()

# ================= 上傳與解析 =================
st.subheader("1) 上傳空白答案卡 (PDF/PNG/JPG)")
template_file = st.file_uploader("Template", type=[
                                 "pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"], key="k_file_template")

st.subheader("2) 上傳答案表（支援多選，如 AC）")
ans_file = st.file_uploader("Answer key", type=["xlsx"], key="k_file_ans")

# 2b) 學號→姓名對照表（可選）
st.subheader("2b) 上傳『學號→姓名』對照表（.xlsx，可選）")
sidmap_file = st.file_uploader("Student ID ↔ Name map", type=[
                               "xlsx"], key="k_file_sidmap")

sid2name: Dict[str, str] = {}
if sidmap_file is not None:
    try:
        sid2name = read_sid_name_map(sidmap_file)
        st.success(f"✅ 已讀入學號對照：{len(sid2name)} 筆")
        if sid2name:
            preview_items = list(sid2name.items())[:10]
            st.write("範例預覽（前 10 筆）:", preview_items)
        else:
            st.warning("沒有解析到任何學號→姓名對照，請檢查欄位或工作表。")
    except Exception as e:
        st.error(f"學號對照讀取失敗：{e}")

# ===== 答案表除錯或自動解析 =====
parsed_answers: List[str] = []
force_qn: Optional[int] = None

if ans_file is not None:
    if st.session_state["k_ans_debug_mode"]:
        st.markdown("### ✅ 答案表除錯模式")
        try:
            xls, sheets = read_excel_sheets(ans_file)
            sheet = st.selectbox("選擇工作表", sheets, index=0, key="k_dbg_sheet")
            df_auto = xls.parse(sheet, header=0)
            df_noh = xls.parse(sheet, header=None)
            st.caption("👀 預覽（上：第一列為標題；下：無標題）")
            st.dataframe(df_auto.head(
                30), use_container_width=True, key="k_dbg_df_auto")
            st.dataframe(df_noh.head(30), use_container_width=True,
                         key="k_dbg_df_noh")

            header_mode = st.radio(
                "標題模式", ["第一列為標題", "無標題"], index=0, key="k_dbg_header")
            df_use = df_auto if header_mode == "第一列為標題" else df_noh
            ncol = df_use.shape[1]
            if ncol == 0:
                st.warning("此表格沒有欄位。")
            else:
                ans_col = st.number_input("『答案欄』索引（0 開始）", 0, max(
                    0, ncol-1), 0, 1, key="k_dbg_ans_col")
                use_qcol = st.checkbox(
                    "我有『題號欄』", value=False, key="k_dbg_has_q")
                q_col = 0
                if use_qcol:
                    q_col = st.number_input("題號欄索引（0 開始）", 0, max(
                        0, ncol-1), 0, 1, key="k_dbg_q_col")
                raw_answers = parse_answers_from_df(
                    df_use, has_header=(header_mode == "第一列為標題"),
                    answer_col=int(ans_col), use_q_col=use_qcol, q_col=int(q_col)
                )

                def trim_tail(arr: List[str]) -> List[str]:
                    last = -1
                    for i, v in enumerate(arr):
                        if re.fullmatch(r"[A-J]+", v or ""):
                            last = i
                    return arr[:last+1] if last >= 0 else []
                parsed_answers = trim_tail(raw_answers)
                st.write(f"解析到有效題數：**{len(parsed_answers)}**")
                if parsed_answers:
                    st.write("前 20 題預覽：", parsed_answers[:20])
                else:
                    st.warning("此設定解析不到 A~J（或多選 AC），請換欄/換表/換標題模式。")

                force_qn = st.number_input(
                    "（選用）強制題數（優先於其他）", 1, 500,
                    value=(len(parsed_answers) if parsed_answers else 1),
                    step=1, key="k_force_qn_dbg"
                )
        except Exception as e:
            st.error(f"答案表除錯讀取失敗：{e}")
            parsed_answers = []
            force_qn = None
    else:
        try:
            parsed_answers, _ = read_answers_auto(
                ans_file, vertical_only=st.session_state["k_ans_vertical_only"])
            st.markdown("#### ✅ 答案表解析結果（自動）")
            st.write(f"解析到有效題數：**{len(parsed_answers)}**")
            if parsed_answers:
                st.write("前 20 題預覽：", parsed_answers[:20])
            else:
                st.warning("解析不到 A~J（或多選 AC）。可開啟『答案表除錯模式』手動指定。")
            force_qn = st.number_input(
                "（選用）強制題數（優先於解析/模板）", 1, 500,
                value=(len(parsed_answers) if parsed_answers else 1),
                step=1, key="k_force_qn_auto"
            )
        except Exception as e:
            st.warning(f"答案表解析失敗：{e}")

# ================= 學生卷上傳（含多頁 PDF） =================
st.subheader("3) 上傳學生卷（可多檔，支援多頁 PDF）")
student_files = st.file_uploader(
    "Student sheets",
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=True, key="k_file_students"
)


def expand_student_inputs(files) -> List[Tuple[np.ndarray, str]]:
    out = []
    if not files:
        return out
    for f in files:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith(".pdf"):
            pages = pdf_to_bgr_list(f)
            base = os.path.splitext(name)[0]
            for i, img in enumerate(pages, start=1):
                out.append((img, f"{base}_p{str(i).zfill(3)}.png"))
        else:
            try:
                img = imread_any(f)
                out.append((img, name))
            except Exception:
                continue
    return out


# ================= 單圖總預覽 =================
st.markdown("### 4) 總預覽ROI")
if template_file is not None:
    try:
        template_file.seek(0)
        full_img = imread_any(template_file)

        # 取得 ROI 比例
        roi_top, roi_bottom, roi_left, roi_right = roi_bounds_from_state()

        roi_img, roi_box = apply_roi(
            full_img, roi_top, roi_bottom, roi_left, roi_right)

        # 建立答案圈
        if st.session_state["k_ans_mode_radio"] == "自動偵測":
            try:
                bboxes = auto_detect_bubbles_from_template(
                    roi_img,
                    dp=st.session_state.get("k_h_dp", 1.2),
                    minDist=st.session_state.get("k_h_minDist", 18.0),
                    param1=st.session_state.get("k_h_p1", 80.0),
                    param2=st.session_state.get("k_h_p2", 20.0),
                    minRadius=int(st.session_state.get("k_h_rmin", 8)),
                    maxRadius=int(st.session_state.get("k_h_rmax", 24))
                )
                ans_bubbles_rel = sort_bubbles_into_grid(
                    bboxes,
                    int(st.session_state.get("k_ans_q", 100)),
                    int(st.session_state.get("k_ans_c", 5))
                )
                questions = int(st.session_state.get("k_ans_q", 100))  # 保留但未使用
                choices_count = int(st.session_state.get("k_ans_c", 5))
            except Exception as e:
                st.warning(f"答案圈自動偵測失敗：{e}")
                ans_bubbles_rel = None
                choices_count = int(st.session_state.get("k_ans_c", 5))
        else:
            questions = int(
                st.session_state["k_man_cols"]*st.session_state["k_man_qpc"])
            bubbles = []
            for col_idx in range(int(st.session_state["k_man_cols"])):
                base_x = int(
                    st.session_state["k_man_ax"] + col_idx*int(st.session_state["k_man_acdx"]))
                for qi in range(int(st.session_state["k_man_qpc"])):
                    y = int(st.session_state["k_man_ay"] +
                            qi*int(st.session_state["k_man_ady"]))
                    row = []
                    for ci in range(int(st.session_state["k_man_choices"])):
                        x = int(base_x + ci*int(st.session_state["k_man_adx"]))
                        row.append((x, y, int(st.session_state["k_man_aw"]), int(
                            st.session_state["k_man_ah"])))
                    bubbles.append(row)
            ans_bubbles_rel = bubbles
            choices_count = int(st.session_state["k_man_choices"])

        # 若有答案或強制題數，預覽裁切
        if ans_bubbles_rel:
            target_qn = None
            if force_qn is not None:
                target_qn = int(force_qn)
            elif parsed_answers:
                target_qn = len(parsed_answers)
            if target_qn and target_qn > 0:
                ans_bubbles_rel = ans_bubbles_rel[:target_qn]

        # 班級/學號（相對 ROI）
        cls_bubbles_rel = make_manual_grid(
            roi_img.shape[0], roi_img.shape[1],
            int(st.session_state["k_cls_x"]), int(st.session_state["k_cls_y"]),
            int(st.session_state["k_cls_w"]), int(st.session_state["k_cls_h"]),
            int(st.session_state["k_cls_dx"]), int(
                st.session_state["k_cls_dy"]),
            int(st.session_state["k_cls_rows"]), 10
        )
        sid_bubbles_rel = make_manual_grid(
            roi_img.shape[0], roi_img.shape[1],
            int(st.session_state["k_sid_x"]), int(st.session_state["k_sid_y"]),
            int(st.session_state["k_sid_w"]), int(st.session_state["k_sid_h"]),
            int(st.session_state["k_sid_dx"]), int(
                st.session_state["k_sid_dy"]),
            int(st.session_state["k_sid_rows"]), 10
        )

        master = draw_master_preview(
            full_img, roi_box, ans_bubbles_rel, cls_bubbles_rel, sid_bubbles_rel,
            show_idx=True,
            show_roi=st.session_state.get("k_show_roi", True),
            scale=master_scale_from_state()
        )
        st.image(cv2.cvtColor(master, cv2.COLOR_BGR2RGB),
                 caption="總預覽ROI",
                 width=int(st.session_state.get("k_w_master", 1100)),
                 use_column_width=False)

        # ✅ 預覽建立完成後，把結果寫入 session_state（供開始閱卷與重建）
        st.session_state["_ans_bubbles_rel"] = ans_bubbles_rel
        st.session_state["_template_roi_img"] = roi_img.copy()
        st.session_state["_tpl_roi_size"] = (
            roi_img.shape[1], roi_img.shape[0])  # (W, H)
        st.session_state["_choices_count"] = int(choices_count)
        st.session_state["_questions_count"] = len(
            ans_bubbles_rel) if ans_bubbles_rel else 0

    except Exception as e:
        st.warning(f"預覽失敗：{e}")
else:
    st.info("請先上傳空白答案卡以顯示單圖總預覽。")

# ================= 輔助：必要時重建泡泡佈局 =================


def _rebuild_bubbles_if_needed():
    if st.session_state.get("_ans_bubbles_rel") is not None \
       and st.session_state.get("_template_roi_img") is not None:
        return  # 已有可用資料

    if template_file is None:
        raise RuntimeError("尚未上傳模板，無法建立答案圈。")

    template_file.seek(0)
    full_img = imread_any(template_file)
    roi_top, roi_bottom, roi_left, roi_right = roi_bounds_from_state()
    tpl_roi_img, _ = apply_roi(
        full_img, roi_top, roi_bottom, roi_left, roi_right)

    if st.session_state["k_ans_mode_radio"] == "自動偵測":
        bboxes = auto_detect_bubbles_from_template(
            tpl_roi_img,
            dp=st.session_state.get("k_h_dp", 1.2),
            minDist=st.session_state.get("k_h_minDist", 18.0),
            param1=st.session_state.get("k_h_p1", 80.0),
            param2=st.session_state.get("k_h_p2", 20.0),
            minRadius=int(st.session_state.get("k_h_rmin", 8)),
            maxRadius=int(st.session_state.get("k_h_rmax", 24))
        )
        ans_bubbles_rel = sort_bubbles_into_grid(
            bboxes,
            int(st.session_state.get("k_ans_q", 100)),
            int(st.session_state.get("k_ans_c", 5))
        )
        choices_count = int(st.session_state.get("k_ans_c", 5))
    else:
        bubbles = []
        for col_idx in range(int(st.session_state["k_man_cols"])):
            base_x = int(st.session_state["k_man_ax"] +
                         col_idx*int(st.session_state["k_man_acdx"]))
            for qi in range(int(st.session_state["k_man_qpc"])):
                y = int(st.session_state["k_man_ay"] +
                        qi*int(st.session_state["k_man_ady"]))
                row = []
                for ci in range(int(st.session_state["k_man_choices"])):
                    x = int(base_x + ci*int(st.session_state["k_man_adx"]))
                    row.append((x, y, int(st.session_state["k_man_aw"]), int(
                        st.session_state["k_man_ah"])))
                bubbles.append(row)
        ans_bubbles_rel = bubbles
        choices_count = int(st.session_state["k_man_choices"])

    # 依『強制題數/解析答案長度』裁切
    target_qn = None
    # 取用目前 UI 的強制題數
    if st.session_state.get("k_ans_debug_mode") and st.session_state.get("k_force_qn_dbg"):
        target_qn = int(st.session_state["k_force_qn_dbg"])
    elif (not st.session_state.get("k_ans_debug_mode")) and st.session_state.get("k_force_qn_auto"):
        target_qn = int(st.session_state["k_force_qn_auto"])
    else:
        # 退而求其次，若 globals 內已有 parsed_answers，拿其長度
        _pa = globals().get("parsed_answers", [])
        if _pa:
            target_qn = len(_pa)

    if target_qn and ans_bubbles_rel:
        ans_bubbles_rel = ans_bubbles_rel[:target_qn]

    st.session_state["_ans_bubbles_rel"] = ans_bubbles_rel
    st.session_state["_template_roi_img"] = tpl_roi_img
    st.session_state["_tpl_roi_size"] = (
        tpl_roi_img.shape[1], tpl_roi_img.shape[0])
    st.session_state["_choices_count"] = int(choices_count)
    st.session_state["_questions_count"] = len(
        ans_bubbles_rel) if ans_bubbles_rel else 0


# ================= 開始閱卷 =================
st.markdown("### 5) 開始閱卷")
start_btn = st.button("開始閱卷", use_container_width=True, key="k_btn_start")

if start_btn:
    if not template_file or not student_files:
        st.error("請先完整上傳：模板、（可選）答案表、至少一份學生卷（可為多頁 PDF）。")
    else:
        try:
            # 若預覽未跑過或 state 遺失，這裡自動重建
            _rebuild_bubbles_if_needed()

            # 取出已建立的狀態
            ans_bubbles_rel = st.session_state["_ans_bubbles_rel"]
            template_img = st.session_state["_template_roi_img"]
            choices_count = int(st.session_state.get("_choices_count", 5))

            if not ans_bubbles_rel:
                raise RuntimeError("答案圈尚未建立，請調整參數或改用手動訓練。")

            # 取得答案鍵（除錯模式解析 > 自動解析 > 空陣列）
            answer_key: List[str] = parsed_answers[:] if parsed_answers else []
            if not answer_key and ans_file is not None:
                try:
                    auto_ans, _ = read_answers_auto(
                        ans_file, vertical_only=st.session_state["k_ans_vertical_only"])
                    answer_key = auto_ans
                except Exception:
                    answer_key = []

            # 目標題數（模板泡泡數、強制題數、答案鍵長度 取最小）
            qn_candidates = [len(ans_bubbles_rel)]
            force_val = st.session_state.get("k_force_qn_dbg") if st.session_state.get(
                "k_ans_debug_mode") else st.session_state.get("k_force_qn_auto")
            if force_val:
                qn_candidates.append(int(force_val))
            if answer_key:
                qn_candidates.append(len(answer_key))
            qn = min(qn_candidates) if qn_candidates else 0
            if qn <= 0:
                raise RuntimeError("無法決定題數；請確認答案表或手動輸入『強制題數』。")

            # 統一切齊
            ans_bubbles_rel = ans_bubbles_rel[:qn]
            answer_key = (answer_key[:qn] if answer_key else [""]*qn)

            # 保存班級/學號設定（相對模板ROI，用於讀取數字）
            roi_top, roi_bottom, roi_left, roi_right = roi_bounds_from_state()
            tpl_roi_img = template_img  # 已是 ROI 影像
            cls_rel_tpl = make_manual_grid(
                tpl_roi_img.shape[0], tpl_roi_img.shape[1],
                int(st.session_state["k_cls_x"]), int(
                    st.session_state["k_cls_y"]),
                int(st.session_state["k_cls_w"]), int(
                    st.session_state["k_cls_h"]),
                int(st.session_state["k_cls_dx"]), int(
                    st.session_state["k_cls_dy"]),
                int(st.session_state["k_cls_rows"]), 10
            )
            sid_rel_tpl = make_manual_grid(
                tpl_roi_img.shape[0], tpl_roi_img.shape[1],
                int(st.session_state["k_sid_x"]), int(
                    st.session_state["k_sid_y"]),
                int(st.session_state["k_sid_w"]), int(
                    st.session_state["k_sid_h"]),
                int(st.session_state["k_sid_dx"]), int(
                    st.session_state["k_sid_dy"]),
                int(st.session_state["k_sid_rows"]), 10
            )

            # 建立模板描述
            answer_template = {
                "width": tpl_roi_img.shape[1],
                "height": tpl_roi_img.shape[0],
                "questions": qn,
                "choices": choices_count,
                "bubbles": ans_bubbles_rel
            }
            template_img = tpl_roi_img.copy()

            # 展開學生卷頁面
            expanded_pages = expand_student_inputs(student_files)
            if not expanded_pages:
                raise RuntimeError("未展開到任何頁面，請確認學生卷檔案是否有效。")

            agg_wrong = [0]*qn
            # 錯誤選項細分統計（每題 A~J）
            wrong_choice_counts = [{L: 0 for L in LETTERS} for _ in range(qn)]
            wrong_totals = [0]*qn

            n_students = 0
            rows, detail_csvs = [], []

            with st.spinner(f"批次評分中…（來源頁數：{len(expanded_pages)}）"):
                for img_bgr, display_name in expanded_pages:
                    Hf, Wf = img_bgr.shape[:2]

                    # 使用比例裁切 ROI（在學生圖上）
                    roi_top, roi_bottom, roi_left, roi_right = roi_bounds_from_state()
                    rx0 = int(round(Wf * roi_left))
                    rx1 = int(round(Wf * roi_right))
                    ry0 = int(round(Hf * roi_top))
                    ry1 = int(round(Hf * roi_bottom))
                    rx0, ry0, rw, rh = rect_clip(
                        rx0, ry0, rx1-rx0, ry1-ry0, Wf, Hf)
                    stu_roi = img_bgr[ry0:ry0+rh, rx0:rx0+rw].copy()

                    # 將「班級、學號」的相對模板座標縮放到學生 ROI 座標，用於讀數字
                    def scale_rel(rel, tpl_box_size, target_size):
                        tw, th = tpl_box_size
                        sw, sh = target_size
                        sx = sw/max(1, tw)
                        sy = sh/max(1, th)
                        out = []
                        for row in rel:
                            out.append(
                                [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x, y, w, h) in row])
                        return out

                    cls_rel_stu = scale_rel(
                        cls_rel_tpl, (answer_template["width"], answer_template["height"]), (rw, rh))
                    sid_rel_stu = scale_rel(
                        sid_rel_tpl, (answer_template["width"], answer_template["height"]), (rw, rh))

                    # 讀班級/學號（每列取最大值的數字）
                    def read_digits(img_roi, rel_bubbles, rows=2):
                        gray = ensure_gray(img_roi)
                        gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        inv = 1.0 - (gray.astype(np.float32)/255.0)
                        digits = []
                        for row in rel_bubbles:
                            scores = []
                            for (x, y, w, h) in row:
                                x0, y0 = max(0, x), max(0, y)
                                x1, y1 = min(
                                    inv.shape[1], x+w), min(inv.shape[0], y+h)
                                roi = inv[y0:y1, x0:x1]
                                scores.append(float(np.mean(roi))
                                              if roi.size else 0.0)
                            digits.append(
                                str(int(np.argmax(scores))) if scores else "?")
                        return "".join(digits[:rows])

                    cls_val = read_digits(stu_roi, cls_rel_stu, rows=int(
                        st.session_state["k_cls_rows"]))
                    sid_val = read_digits(stu_roi, sid_rel_stu, rows=int(
                        st.session_state["k_sid_rows"]))

                    # 透過對照表取得姓名
                    stu_name = sid2name.get(sid_val, "")

                    # ✅ 評分（支援多選）— 使用「模板座標」answer_template，不再縮放 bubbles
                    res = grade_one(
                        stu_roi, template_img, answer_template,
                        answer_key, fill_threshold=st.session_state.get(
                            "k_fill", 0.72),
                        allow_multi=st.session_state["k_allow_multi"],
                        multi_abs_min=st.session_state["k_multi_abs"],
                        multi_rel_max=st.session_state["k_multi_rel"],
                        grade_policy=st.session_state["k_grade_policy"]
                    )

                    # 聚合：錯題總數與錯誤選項比例
                    for item in res["detail"]:
                        q_idx = int(item["Q"])-1
                        if 0 <= q_idx < len(agg_wrong):
                            is_correct = int(item["Correct"]) == 1
                            if not is_correct:
                                agg_wrong[q_idx] += 1
                                wrong_totals[q_idx] += 1
                                pick_set = set(re.findall(
                                    r"[A-J]", str(item.get("Pick", "")).upper()))
                                for L in pick_set:
                                    if L in LETTERS:
                                        wrong_choice_counts[q_idx][L] += 1

                    n_students += 1

                    rows.append({
                        "file": display_name, "class": cls_val, "student_id": sid_val,
                        "name": stu_name, "score": res["score"], "total": res["total"], "percent": res["percent"]
                    })

                    df_detail = pd.DataFrame(res["detail"])
                    # 依序插入：class -> student_id -> name
                    df_detail.insert(0, "class", cls_val)
                    df_detail.insert(1, "student_id", sid_val)
                    df_detail.insert(2, "name", stu_name)
                    bio = io.BytesIO()
                    df_detail.to_csv(bio, index=False, encoding="utf-8-sig")
                    bio.seek(0)
                    # 檔名含學號與姓名（若有）
                    save_name = (f"{sid_val}_{stu_name}_detail.csv"
                                 if stu_name else f"{os.path.splitext(display_name)[0]}_detail.csv")
                    detail_csvs.append((save_name, bio.read()))

            # 總表
            df = pd.DataFrame(rows)
            st.success(f"完成！共處理 {n_students} 份頁面/學生卷")
            st.dataframe(df, use_container_width=True, key="k_df_final")

            out_buf = io.BytesIO()
            df.to_csv(out_buf, index=False, encoding="utf-8-sig")
            out_buf.seek(0)
            st.download_button("⬇️ 下載 results.csv", data=out_buf, file_name="results.csv",
                               mime="text/csv", use_container_width=True, key="k_dl_results")

            if detail_csvs:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for name, data in detail_csvs:
                        z.writestr(name, data)
                zip_buf.seek(0)
                st.download_button("⬇️ 下載逐題明細 ZIP", data=zip_buf, file_name="details.zip",
                                   mime="application/zip", use_container_width=True, key="k_dl_details")

            # 未對上姓名的學號清單（若有上傳對照表）
            if sid2name:
                unmatched = sorted({r["student_id"]
                                   for r in rows if not r.get("name")})
                if unmatched:
                    st.warning(f"⚠️ 有 {len(unmatched)} 個學號未在對照表中找到姓名。")
                    _buf_un = io.BytesIO()
                    pd.DataFrame({"student_id": unmatched}).to_csv(
                        _buf_un, index=False, encoding="utf-8-sig")
                    _buf_un.seek(0)
                    st.download_button(
                        "⬇️ 下載未對上姓名的學號清單",
                        data=_buf_un,
                        file_name="unmatched_student_ids.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="k_dl_unmatched"
                    )

            # 每題錯誤統計（總錯人數）
            if n_students > 0:
                q_numbers = list(range(1, len(agg_wrong)+1))
                wrong_rates = [round(w/n_students*100.0, 2) for w in agg_wrong]
                df_wrong = pd.DataFrame(
                    {"Q": q_numbers, "WrongCount": agg_wrong, "WrongRate(%)": wrong_rates})
                st.subheader("📊 每題錯的人數（與錯誤率）")
                st.dataframe(df_wrong, use_container_width=True,
                             height=350, key="k_df_wrong")
                buf_wrong = io.BytesIO()
                df_wrong.to_csv(buf_wrong, index=False, encoding="utf-8-sig")
                buf_wrong.seek(0)
                st.download_button("⬇️ 下載 per_question_wrong.csv", data=buf_wrong,
                                   file_name="per_question_wrong.csv", mime="text/csv",
                                   use_container_width=True, key="k_dl_wrong")

                # 錯題選項分佈（在錯的人之中，各選項所占比例）
                rows_break = []
                for qi in range(len(q_numbers)):
                    total_wrong = wrong_totals[qi]
                    counts = wrong_choice_counts[qi]
                    perc = {L: (round(
                        counts[L]/total_wrong*100.0, 2) if total_wrong > 0 else 0.0) for L in LETTERS}
                    # 僅輸出到 J，避免表過寬
                    row_out = {
                        "Q": qi+1,
                        "WrongTotal": total_wrong,
                        **{f"{L}_count": counts[L] for L in LETTERS},
                        **{f"{L}_%": perc[L] for L in LETTERS}
                    }
                    rows_break.append(row_out)
                df_break = pd.DataFrame(rows_break)
                st.subheader("🧭 錯題選項分佈（在錯的人之中，各選項所占比例）")
                st.caption("若允許多選：一位學生同一題可能同時計入多個選項的錯誤次數。")
                st.dataframe(df_break, use_container_width=True,
                             height=400, key="k_df_break")

                buf_break = io.BytesIO()
                df_break.to_csv(buf_break, index=False, encoding="utf-8-sig")
                buf_break.seek(0)
                st.download_button(
                    "⬇️ 下載 per_question_wrong_choice_breakdown.csv",
                    data=buf_break,
                    file_name="per_question_wrong_choice_breakdown.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="k_dl_break"
                )
            else:
                st.info("沒有成功處理的學生卷，無法產生每題錯誤統計。")

        except Exception as e:
            st.exception(e)
render_cc_footer()
