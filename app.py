#######################################################################################
# Spotting Salmon App
# Written by: Rachel Lennon & co-pilot troubleshooting
#######################################################################################

####################################
# IMPORTS
####################################

import os
import io
import zipfile
import tempfile
import shutil
import atexit

import cv2
import pandas as pd
import numpy as np
import streamlit as st
from databricks.sdk import WorkspaceClient

####################################
# PAGE CONFIG
####################################

st.set_page_config(page_title="Spotting Salmon", layout="wide")
st.title("🐟 SpottingSalmon – Video Fish Counter")
st.subheader("🔍 Upload Monitoring Videos")

####################################
# HELPERS
####################################

def safe_rmtree(path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except:
        pass

def register_cleanup_once():
    if not st.session_state.get("_cleanup_registered"):
        def cleanup():
            safe_rmtree(st.session_state.get("temp_base_folder"))
        atexit.register(cleanup)
        st.session_state["_cleanup_registered"] = True

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def standardise_bbox_columns(df):
    """
    Map model bbox columns to xmin/ymin/xmax/ymax if needed.
    Only adds these columns when x1/y1/x2/y2 exist.
    """
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(df.columns):
        return df

    if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        df = df.copy()
        df["xmin"] = df["x1"]
        df["ymin"] = df["y1"]
        df["xmax"] = df["x2"]
        df["ymax"] = df["y2"]

    return df

def maybe_scale_bboxes(df, w, h):
    """If boxes look normalized (<=1.5), scale to pixel coords in-place."""
    needed = {"xmin","ymin","xmax","ymax"}
    if not needed.issubset(df.columns):
        return df
    max_x = pd.concat([df["xmin"], df["xmax"]], axis=1).max(axis=1).max()
    max_y = pd.concat([df["ymin"], df["ymax"]], axis=1).max(axis=1).max()
    if pd.notnull(max_x) and pd.notnull(max_y) and max_x <= 1.5 and max_y <= 1.5:
        df["xmin"] = (df["xmin"] * w).round()
        df["xmax"] = (df["xmax"] * w).round()
        df["ymin"] = (df["ymin"] * h).round()
        df["ymax"] = (df["ymax"] * h).round()
    return df

def draw_bboxes(frame, preds, w, h):
    """Draw bounding boxes + labels on a BGR frame."""
    count = 0
    for _, row in preds.iterrows():
        if not {"xmin","ymin","xmax","ymax"}.issubset(row.index):
            continue
        try:
            x1, y1, x2, y2 = map(int,
                                [row["xmin"], row["ymin"],
                                 row["xmax"], row["ymax"]])
        except Exception:
            continue

        # Clamp to bounds
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
        label = str(row["track_id"]) if "track_id" in preds.columns and pd.notnull(row.get("track_id")) else "fish"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        count += 1
    return count

####################################
# SAVE ZIP INTO UC
####################################

w = WorkspaceClient()
uploaded_file = st.file_uploader("Upload a ZIP file with monitoring videos")

if st.button("Save upload"):
    if not uploaded_file:
        st.error("Please upload a ZIP file.")
        st.stop()

    try:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        if not file_name.lower().endswith(".zip"):
            st.error("Must upload a ZIP.")
            st.stop()

        zf = zipfile.ZipFile(io.BytesIO(file_bytes))

        dest_folder = (
            "/Volumes/prd_dash_lab/dash_data_science_unrestricted/"
            "shared_external_volume/rachels_stuff/"
            f"{file_name.replace('.zip','')}"
        )

        # Temp folder for QC raw videos
        temp_base = tempfile.mkdtemp(prefix="qc_videos_")
        st.session_state["temp_base_folder"] = temp_base
        register_cleanup_once()

        uploaded_uc = []
        local_paths = []

        mp4s = [m for m in zf.namelist() if m.lower().endswith(".mp4")]
        if not mp4s:
            st.error("ZIP contained no .mp4 files.")
            st.stop()

        prog = st.progress(0.0)

        for i, member in enumerate(mp4s, start=1):
            fname = os.path.basename(member)
            fbytes = zf.read(member)

            # Upload to UC
            uc_path = f"{dest_folder}/{fname}"
            w.files.upload(uc_path, io.BytesIO(fbytes), overwrite=True)
            uploaded_uc.append(uc_path)

            # Local temp copy for QC
            local_path = os.path.join(temp_base, fname)
            with open(local_path, "wb") as f:
                f.write(fbytes)
            local_paths.append(local_path)

            prog.progress(i / len(mp4s))

        st.session_state["input_df"] = pd.DataFrame({"fish": uploaded_uc})
        st.session_state["uploaded_mp4_paths"] = uploaded_uc
        st.session_state["temp_mp4_paths"] = local_paths

        prog.empty()
        st.success("Upload complete.")

    except Exception as e:
        st.error(f"Error saving: {e}")

####################################
# MODEL INFERENCE
####################################

st.subheader("🔍 Run Inference")

if st.button("🚀 Start Inference"):
    if "input_df" not in st.session_state:
        st.error("No uploaded videos found. Upload first.")
        st.stop()

    input_df = st.session_state["input_df"]
    if input_df.empty:
        st.error("Upload videos first.")
        st.stop()

    payload = {"fish": input_df["fish"].tolist()}

    try:
        response = w.serving_endpoints.query(
            name="salmon_model_e",
            inputs=payload
        )
        resp = response.as_dict() if hasattr(response, "as_dict") else dict(response)

        preds = resp.get("predictions", [])
        if not preds:
            st.warning("Model returned no predictions.")
            st.stop()

        preds_df = pd.DataFrame(preds)

        # Coerce types to be safe
        preds_df = coerce_numeric(preds_df, ["frame","xmin","ymin","xmax","ymax","x1","y1","x2","y2"])

        # ONLY NEEDED FIX: map x1/y1/x2/y2 to xmin/ymin/xmax/ymax
        preds_df = standardise_bbox_columns(preds_df)

        ####################################
        # SUMMARY
        ####################################

        st.subheader("🐟 Fish Count Summary")

        if "track_id" in preds_df.columns:
            summary = (
                preds_df.groupby("video")["track_id"]
                .nunique()
                .reset_index(name="unique_fish")
            )
        else:
            summary = (
                preds_df.groupby("video")
                .size()
                .reset_index(name="detections")
            )

        st.dataframe(summary, use_container_width=True)

        ####################################
        # QC BLOCK
        ####################################

        st.subheader("🎥 QC: Review Input Videos")

        videos = preds_df["video"].dropna().unique()
        selected_video = st.selectbox("Select video:", videos)

        if not selected_video:
            st.stop()

        base = os.path.basename(selected_video)
        raw_path = os.path.join(st.session_state["temp_base_folder"], base)

        if not os.path.exists(raw_path):
            st.error("Local QC file missing.")
            st.stop()

        # Show raw
        st.markdown("### 🎞️ Original Video")
        with open(raw_path, "rb") as f:
            st.video(f.read())

        ####################################
        # ANNOTATED VIDEO (WEBM, robust matching)
        ####################################

        st.markdown("### 🟥 Annotated Video")

        vpreds = preds_df[preds_df["video"] == selected_video].copy()
        if vpreds.empty:
            st.info("No detections for this video.")
            st.stop()

        if "frame" not in vpreds.columns:
            st.info("'frame' column missing from predictions.")
            st.stop()

        # Open raw to get properties
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            st.error("Cannot open raw video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # If boxes are normalized, scale in-place for this video subset
        vpreds = maybe_scale_bboxes(vpreds, w_vid, h_vid)

        # Writer (WebM VP8)
        work_dir = tempfile.mkdtemp(prefix="annot_")
        out_path = os.path.join(work_dir, "annotated.webm")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (w_vid, h_vid))

        if not writer.isOpened():
            st.error("Failed to open VP8 writer.")
            cap.release()
            safe_rmtree(work_dir)
            st.stop()

        # Decide index base; still use a small window to be robust
        one_based = (pd.notnull(vpreds["frame"].min()) and int(vpreds["frame"].min()) >= 1)
        boxes_drawn_total = 0
        frame_num = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Match frames with a small ±1 window to tolerate off-by-one
            candidate = [frame_num-1, frame_num, frame_num+1]
            if one_based:
                candidate = [c+1 for c in candidate]  # shift if needed

            pf = vpreds[vpreds["frame"].isin(candidate)]

            boxes_drawn_total += draw_bboxes(frame, pf, w_vid, h_vid)
            writer.write(frame)

            frame_num += 1

        cap.release()
        writer.release()

        # Basic debug
        with st.expander("ℹ️ Annotation Debug"):
            st.write({
                "fps": fps, "width": w_vid, "height": h_vid,
                "total_frames": total_frames,
                "pred_rows_for_video": int(len(vpreds)),
                "boxes_drawn_total": int(boxes_drawn_total)
            })

        # Play annotated
        with open(out_path, "rb") as f:
            st.video(f.read(), format="video/webm")

        # Download button
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download Annotated WebM",
                data=f,
                file_name="annotated.webm",
                mime="video/webm"
            )

        safe_rmtree(work_dir)
        st.success("Annotation complete ✅")

    except Exception as e:
        st.error(f"Error during inference: {e}")