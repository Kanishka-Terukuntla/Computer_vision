"""
SIFT Face Recognition Studio
Run from project root: streamlit run src/app.py
"""
from __future__ import annotations

import io
import pickle
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"
MODEL_PATH = ROOT / "sift_model.pkl"
FACE_SIZE = 200
TRAIN_TARGET = 10
TEST_TARGET = 2
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

for folder in (TRAIN_DIR, TEST_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SIFT Face Recognition Studio", page_icon="🧑", layout="wide")

# ----------------------------- Styling --------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background:rgba(120,120,120,.08); border:1px solid rgba(120,120,120,.12); padding:13px 16px; border-radius:13px;}
.hero {padding:1.1rem 1.3rem; border-radius:16px; background:linear-gradient(120deg, rgba(70,110,180,.18), rgba(30,160,140,.10)); border:1px solid rgba(120,140,180,.25); margin-bottom:1rem;}
.hero h1 {margin:0 0 .25rem 0;}
.small-muted {color:var(--text-color); opacity:.7; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Helpers --------------------------------------
@st.cache_resource
def get_sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT is unavailable. Install opencv-contrib-python.")
    return cv2.SIFT_create(nfeatures=1200)

@st.cache_resource
def get_face_detector():

    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    if not cascade_path.exists():
        return None

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        return None

    return detector
def clean_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return name.strip("_")[:40]

def decode_image(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)

def people_list():
    names = set()
    for base in (TRAIN_DIR, TEST_DIR):
        names.update(p.name for p in base.iterdir() if p.is_dir())
    return sorted(names)

def preprocess(bgr):

    if bgr is None:
        return None, False, None

    image = bgr.copy()

    h, w = image.shape[:2]

    # Resize large images
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale))
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Try Haar Cascade face detection
    detector = get_face_detector()

    faces = []

    if detector is not None:
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(45, 45)
        )

    found = len(faces) > 0

    # Crop the largest detected face
    if found:

        x, y, fw, fh = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        margin = int(0.12 * max(fw, fh))

        x0 = max(0, x - margin)
        y0 = max(0, y - margin)

        x1 = min(gray.shape[1], x + fw + margin)
        y1 = min(gray.shape[0], y + fh + margin)

        crop = gray[y0:y1, x0:x1]

    else:
        # Fallback: use the complete image
        crop = gray

    # Resize to standard dimensions
    crop = cv2.resize(
        crop,
        (200, 200)
    )

    # Improve local contrast using CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    crop = clahe.apply(crop)

    return crop, found, image

def describe(face):
    keypoints, descriptors = get_sift().detectAndCompute(face, None)
    return keypoints, descriptors

def good_match_count(query_desc, train_desc, ratio):
    if query_desc is None or train_desc is None or len(query_desc) < 2 or len(train_desc) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query_desc, train_desc, k=2)
    return sum(1 for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance)

def dataset_stats():
    rows = []
    for person in people_list():
        tr = len(list_images(TRAIN_DIR / person))
        te = len(list_images(TEST_DIR / person))
        rows.append({
            "Person": person, "Training": tr, "Testing": te,
            "Train target": f"{tr}/{TRAIN_TARGET}",
            "Test target": f"{te}/{TEST_TARGET}",
            "Status": "Ready" if tr >= TRAIN_TARGET and te >= TEST_TARGET else "Incomplete"
        })
    return pd.DataFrame(rows)

def save_uploaded_images(split, person, uploaded, skip_no_face):
    base = TRAIN_DIR if split == "Training" else TEST_DIR
    folder = base / person
    folder.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0
    for item in uploaded:
        bgr = decode_image(item.getvalue())
        if bgr is None:
            skipped += 1
            continue
        _, face_found, _ = preprocess(bgr)
        if skip_no_face and not face_found:
            skipped += 1
            continue
        filename = f"{time.time_ns()}_{saved:03d}.jpg"
        cv2.imwrite(str(folder / filename), bgr)
        saved += 1
    return saved, skipped

def train_model():
    records = []
    progress = st.progress(0, text="Preparing training images...")
    image_paths = [(pdir.name, f) for pdir in sorted(TRAIN_DIR.iterdir()) if pdir.is_dir()
                   for f in list_images(pdir)]
    for idx, (person, path) in enumerate(image_paths):
        bgr = cv2.imread(str(path))
        face, found, _ = preprocess(bgr)
        if face is not None:
            _, desc = describe(face)
            if desc is not None and len(desc) >= 2:
                records.append({"person": person, "file": path.name, "des": desc})
        progress.progress((idx + 1) / max(1, len(image_paths)),
                          text=f"Extracting SIFT: {idx + 1}/{len(image_paths)}")
    progress.empty()
    if not records:
        raise RuntimeError("No usable SIFT descriptors found. Check image files and quality.")
    payload = {
        "version": 1,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
        "ratio_default": 0.75
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    return payload

def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "records" in payload:
            return payload
        return None
    except Exception:
        return None

def predict_face(face, model, ratio, min_matches):
    _, query_desc = describe(face)
    scores = {}
    for record in model["records"]:
        count = good_match_count(query_desc, record["des"], ratio)
        person = record["person"]
        scores[person] = max(scores.get(person, 0), count)
    if not scores:
        return "Unknown", {}, 0
    best_person = max(scores, key=scores.get)
    best_score = scores[best_person]
    label = best_person if best_score >= min_matches else "Unknown"
    return label, scores, best_score

def evaluate_model(model, ratio, min_matches):
    results = []
    test_files = [(pdir.name, f) for pdir in sorted(TEST_DIR.iterdir()) if pdir.is_dir()
                  for f in list_images(pdir)]
    for actual, path in test_files:
        bgr = cv2.imread(str(path))
        face, found, _ = preprocess(bgr)
        if face is None:
            continue
        pred, scores, best = predict_face(face, model, ratio, min_matches)
        results.append({
            "Image": path.name, "Actual": actual, "Predicted": pred,
            "Good matches": best, "Face detected": found,
            "Correct": pred == actual, "Preview": face
        })
    return results

def invalidate_model():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

# ----------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.title("🧑 SIFT Face ID")
    st.caption("Classical computer vision • SIFT + Euclidean L2")
    st.divider()
    st.subheader("Recognition settings")
    ratio = st.slider("Lowe ratio threshold", 0.50, 0.95, 0.75, 0.01,
                      help="A match is retained when its nearest-neighbour distance is sufficiently smaller than the second-nearest.")
    min_matches = st.slider("Minimum good matches", 0, 30, 3,
                            help="Predictions below this match count are rejected as Unknown.")
    st.divider()
    stats = dataset_stats()
    total_train = int(stats["Training"].sum()) if not stats.empty else 0
    total_test = int(stats["Testing"].sum()) if not stats.empty else 0
    c1, c2 = st.columns(2)
    c1.metric("People", len(stats))
    c2.metric("Train images", total_train)
    c1.metric("Test images", total_test)
    model_state = load_model()
    st.caption("Model: " + ("Trained" if model_state else "Not trained"))

# ----------------------------- Main header ----------------------------------
st.markdown("""
<div class="hero">
  <h1>Face Recognition Studio</h1>
  <div class="small-muted">Build a face dataset, extract SIFT descriptors, recognize faces and evaluate test accuracy.</div>
</div>
""", unsafe_allow_html=True)

tab_data, tab_train, tab_rec, tab_eval, tab_about = st.tabs([
    "📁 Dataset Manager", "⚙️ Train Model", "🎯 Recognize Face",
    "📊 Evaluation", "🧠 How it works"
])

# ----------------------------- Dataset manager ------------------------------
with tab_data:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Add images to dataset")
        name_input = st.text_input("Person folder name", placeholder="Person_01")
        split = st.radio("Dataset split", ["Training", "Testing"], horizontal=True)
        source = st.radio("Image source", ["Upload images", "Camera"], horizontal=True)
        skip_no_face = st.checkbox("Skip image if Haar detector finds no face", value=False,
                                   help="Disable if your source images are already tightly cropped face images.")
        person = clean_name(name_input) if name_input else ""
        if source == "Upload images":
            uploaded = st.file_uploader("Choose one or more images",
                type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                accept_multiple_files=True, key="dataset_upload")
            if st.button("Save images", type="primary", disabled=not (person and uploaded)):
                saved, skipped = save_uploaded_images(split, person, uploaded, skip_no_face)
                invalidate_model()
                st.success(f"Saved {saved} image(s); skipped {skipped}. Retrain the model after dataset changes.")
                st.rerun()
        else:
            camera = st.camera_input("Capture a face image", key="dataset_camera")
            if st.button("Save camera photo", type="primary", disabled=not (person and camera)):
                saved, skipped = save_uploaded_images(split, person, [camera], skip_no_face)
                invalidate_model()
                if saved:
                    st.success(f"Saved photo for {person}.")
                else:
                    st.warning("Image could not be saved. Try another photo.")
                st.rerun()

        st.info("Target: 10 training images and 2 separate test images per person. Keep each person's folder name identical across both splits.")

    with right:
        st.subheader("Dataset status")
        stats = dataset_stats()
        if stats.empty:
            st.info("No person folders found yet. Add images or copy them into dataset/train and dataset/test.")
        else:
            st.dataframe(stats, hide_index=True, use_container_width=True)
            st.progress(min(1.0, total_train / 100), text=f"{total_train}/100 training images")
            st.progress(min(1.0, total_test / 20), text=f"{total_test}/20 testing images")
            available = people_list()
            selected = st.selectbox("Preview or manage a person", available)
            p1, p2 = st.columns(2)
            for col, title, base in ((p1, "Training", TRAIN_DIR), (p2, "Testing", TEST_DIR)):
                images = list_images(base / selected)
                with col:
                    st.markdown(f"**{title} ({len(images)})**")
                    if images:
                        thumbs = st.columns(2)
                        for i, image_path in enumerate(images[:6]):
                            thumbs[i % 2].image(str(image_path), caption=image_path.name,
                                                use_container_width=True)
                    else:
                        st.caption("No images yet.")
            if st.button(f"Delete {selected} from both splits", type="secondary"):
                for base in (TRAIN_DIR, TEST_DIR):
                    shutil.rmtree(base / selected, ignore_errors=True)
                invalidate_model()
                st.rerun()

# ----------------------------- Training --------------------------------------
with tab_train:
    st.subheader("SIFT feature extraction")
    st.write("The training stage extracts SIFT keypoints and 128-dimensional descriptors from every usable training image and saves them locally.")
    current = load_model()
    if current:
        st.success(f"Saved model found • {len(current['records'])} training images • {current.get('trained_at', 'unknown time')}")
    else:
        st.warning("No trained model found. Train after preparing the dataset.")
    ready_train = total_train > 0
    if st.button("🚀 Train / Retrain model", type="primary", disabled=not ready_train):
        try:
            with st.spinner("Detecting faces and extracting SIFT descriptors..."):
                payload = train_model()
            st.session_state["last_eval"] = None
            st.success(f"Training complete. Processed {len(payload['records'])} images.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    current = load_model()
    if current:
        records = current["records"]
        keypoint_counts = [len(r["des"]) for r in records]
        m1, m2, m3 = st.columns(3)
        m1.metric("Images used", len(records))
        m2.metric("Total descriptors", f"{sum(keypoint_counts):,}")
        m3.metric("Average descriptors / image", f"{np.mean(keypoint_counts):.1f}")
        people = sorted(set(r["person"] for r in records))
        st.subheader("Visualize SIFT keypoints")
        col_a, col_b = st.columns(2)
        selected_person = col_a.selectbox("Person", people, key="keypoint_person")
        person_images = list_images(TRAIN_DIR / selected_person)
        if person_images:
            selected_file = col_b.selectbox("Training image", person_images, format_func=lambda p: p.name)
            bgr = cv2.imread(str(selected_file))
            face, found, _ = preprocess(bgr)
            keypoints, _ = describe(face)
            visual = cv2.drawKeypoints(face, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            v1, v2 = st.columns(2)
            v1.image(face, caption="Preprocessed face", use_container_width=True)
            v2.image(visual, caption=f"{len(keypoints)} SIFT keypoints", channels="BGR",
                     use_container_width=True)

# ----------------------------- Recognition ----------------------------------
with tab_rec:
    st.subheader("Recognize a new face")
    model = load_model()
    if not model:
        st.info("Train the model first.")
    else:
        rec_source = st.radio("Input source", ["Upload image", "Camera"], horizontal=True, key="rec_source")
        data = None
        if rec_source == "Upload image":
            uploaded = st.file_uploader("Upload a face photograph", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                        key="recognition_upload")
            data = uploaded.getvalue() if uploaded else None
        else:
            camera = st.camera_input("Take a photo", key="recognition_camera")
            data = camera.getvalue() if camera else None

        if data:
            bgr = decode_image(data)
            face, face_found, original = preprocess(bgr)
            col1, col2 = st.columns([1, 1.4], gap="large")
            with col1:
                st.image(cv2.cvtColor(face, cv2.COLOR_GRAY2RGB),
                         caption="Preprocessed face" + ("" if face_found else " (no face detected; full image used)"),
                         use_container_width=True)
            with col2:
                label, scores, best = predict_face(face, model, ratio, min_matches)
                if label == "Unknown":
                    st.warning("Identity could not be determined confidently.")
                else:
                    st.success("Recognition completed")
                st.metric("Predicted identity", label)
                st.metric("Best good-match count", best)
                if scores:
                    score_df = pd.DataFrame(
                        [{"Person": p, "Good SIFT matches": s} for p, s in
                         sorted(scores.items(), key=lambda x: x[1], reverse=True)]
                    )
                    st.bar_chart(score_df.set_index("Person"))
                    st.dataframe(score_df, hide_index=True, use_container_width=True)

# ----------------------------- Evaluation -----------------------------------
with tab_eval:
    st.subheader("Evaluate using the held-out test dataset")
    model = load_model()
    if not model:
        st.info("Train the model before evaluation.")
    elif total_test == 0:
        st.info("No testing images found. Add 2 separate test images per person.")
    else:
        if st.button("▶ Run evaluation", type="primary"):
            try:
                with st.spinner("Matching all test images..."):
                    st.session_state["last_eval"] = evaluate_model(model, ratio, min_matches)
            except Exception as e:
                st.error(str(e))
        results = st.session_state.get("last_eval")
        if results:
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "Preview"} for r in results])
            if not df.empty:
                accuracy = float(df["Correct"].mean() * 100)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test accuracy", f"{accuracy:.2f}%")
                c2.metric("Correct", f"{int(df['Correct'].sum())}/{len(df)}")
                c3.metric("Incorrect", int((~df["Correct"]).sum()))
                c4.metric("Unknown", int((df["Predicted"] == "Unknown").sum()))

                st.subheader("Per-image results")
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.subheader("Confusion matrix")
                st.dataframe(pd.crosstab(df["Actual"], df["Predicted"]),
                             use_container_width=True)
                st.subheader("Test image previews")
                preview_cols = st.columns(5)
                for i, row in enumerate(results):
                    mark = "✅" if row["Correct"] else "❌"
                    preview_cols[i % 5].image(
                        cv2.cvtColor(row["Preview"], cv2.COLOR_GRAY2RGB),
                        caption=f"{mark} {row['Actual']} → {row['Predicted']}",
                        use_container_width=True
                    )
                st.download_button("⬇ Download evaluation results (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="sift_face_recognition_results.csv",
                    mime="text/csv")
                st.caption("Accuracy = correct predictions / evaluated test images × 100.")

# ----------------------------- Explanation ----------------------------------
with tab_about:
    st.subheader("How the SIFT recognition pipeline works")
    st.markdown("""
1. **Face preprocessing:** Haar Cascade attempts to detect the largest face. If no face is detected, the full image is used. The crop is resized to 200 × 200 grayscale pixels and CLAHE improves local contrast.
2. **SIFT feature extraction:** SIFT detects scale- and rotation-robust keypoints and computes a 128-value descriptor for each keypoint.
3. **Euclidean matching:** BFMatcher with `NORM_L2` compares descriptors using Euclidean distance.
4. **Lowe ratio test:** A descriptor match is retained when its nearest match is sufficiently better than its second-nearest match.
5. **Identity prediction:** For each identity, the highest good-match count among that person's training images is used. The identity with the highest count wins, subject to the minimum-match threshold.
6. **Evaluation:** Every held-out test image is predicted and compared with its folder label. Accuracy and a confusion matrix are displayed.
    """)
    st.warning("This is a classical SIFT feature-matching baseline, not a deep-learning classifier. Matching thresholds affect the Unknown rate and accuracy; report the settings used in your assignment.")
    st.markdown(f"**Project dataset path:** `{DATASET}`  \n**Saved model:** `{MODEL_PATH}`")
