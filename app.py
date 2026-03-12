
#######################################################################################
# Spotting Salmon App
# Written by: Rachel Lennon & co-pilot troubleshooting
# Purpose: A UI for detecting and counting wild salmon from EA monitoring sites.
#######################################################################################

####################################
# SET UP
####################################

# Import packages
import os
import io
import zipfile
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
from databricks.sdk import WorkspaceClient
import tempfile, shutil
import subprocess
from PIL import Image, ImageDraw

####################################
# OPEN PAGE
####################################

# Title
st.set_page_config(page_title="Spotting Salmon", layout="wide")
st.title("🐟 SpottingSalmon – Video Fish Counter")

st.subheader("🔍 Upload Monitoring Videos")


####################################
# SAVE ZIP INTO UC VOLUME
####################################

w = WorkspaceClient()
uploaded_file = st.file_uploader("Upload a ZIP file with monitoring videos")

if st.button("Save upload"):
    if uploaded_file is None:
        st.error("Please upload a ZIP file.")
        st.stop()

    try:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name

        if not file_name.lower().endswith(".zip"):
            st.error("Please upload a ZIP file.")
            st.stop()

        zf = zipfile.ZipFile(io.BytesIO(file_bytes))

        # UC folder
        dest_folder = (
            "/Volumes/prd_dash_lab/dash_data_science_unrestricted/"
            "shared_external_volume/rachels_stuff/"
            f"{file_name.replace('.zip','')}"
        )

        # Temp folder for previews
        temp_base = tempfile.mkdtemp(prefix="qc_videos_")
        st.session_state["temp_base_folder"] = temp_base

        uploaded_mp4_paths = []
        temp_mp4_paths = []

        for member in zf.namelist():
            if not member.lower().endswith(".mp4"):
                continue

            fname = os.path.basename(member)
            if not fname:
                continue

            file_content = zf.read(member)

            # 1) Upload to UC
            uc_path = f"{dest_folder}/{fname}"
            w.files.upload(uc_path, io.BytesIO(file_content), overwrite=True)
            uploaded_mp4_paths.append(uc_path)

            # 2) Save local temp copy
            temp_path = os.path.join(temp_base, fname)
            with open(temp_path, "wb") as f:
                f.write(file_content)
            temp_mp4_paths.append(temp_path)

        if not uploaded_mp4_paths:
            st.error("ZIP contained no .mp4 files.")
            st.stop()

        st.session_state["input_df"] = pd.DataFrame({"fish": uploaded_mp4_paths})
        st.session_state["uploaded_base_folder"] = dest_folder
        st.session_state["uploaded_mp4_paths"] = uploaded_mp4_paths
        st.session_state["temp_mp4_paths"] = temp_mp4_paths

        st.success(f"Uploaded {len(uploaded_mp4_paths)} videos into UC + local temp copies!")

    except Exception as e:
        st.error(f"Error saving: {e}")
            
####################################
# Model Inference
####################################

st.subheader("🔍 Run Inference")

if st.button("🚀 Start Inference"):
    if "input_df" not in st.session_state:
        st.error("No saved file found. Upload and click 'Save changes' first.")
    else:
        input_df = st.session_state["input_df"]

        if input_df.empty:
            st.error("No videos prepared. Upload a ZIP and click 'Save changes' first.")
        else:
            payload = {
                "fish": input_df["fish"].astype(str).tolist()
            }

            try:
                # --- Call serving endpoint ---
                response = w.serving_endpoints.query(
                    name="salmon_model_e",
                    inputs=payload
                )

                resp_dict = response.as_dict() if hasattr(response, "as_dict") else dict(response)

                preds = resp_dict["predictions"]
                preds_df = pd.DataFrame(preds)

                 # --- Fish counts ---

                st.subheader("🐟 Fish Count Summary")
                if "track_id" in preds_df.columns:
                    fish_counts = (
                        preds_df.groupby("video")["track_id"]
                        .nunique()
                        .reset_index(name="unique_fish")
                        .sort_values("unique_fish", ascending=False)
                    )
                else:
                    fish_counts = (
                        preds_df.groupby("video")
                        .size()
                        .reset_index(name="detections")
                        .sort_values("detections", ascending=False)
                    )

                st.dataframe(fish_counts)

                # ----------------------------------------------------
                # QC BLOCK
                # ----------------------------------------------------

                st.subheader("🎥 QC: Review Input Videos")

                # Ensure predictions + paths are available
                if (
                    "input_df" in st.session_state 
                    and "uploaded_mp4_paths" in st.session_state
                    and "temp_base_folder" in st.session_state
                ):
                    videos = preds_df["video"].unique()
                    selected_video = st.selectbox("Select a video:", videos)

                    if selected_video:
                        # Local temp copy path
                        base_name = os.path.basename(selected_video)
                        temp_base = st.session_state["temp_base_folder"]
                        temp_path = os.path.join(temp_base, base_name)

                        if not os.path.exists(temp_path):
                            st.error("Cannot find local temp copy. Re-upload ZIP.")
                            st.stop()

                        # ------------------------------------------------
                        # Show raw original video
                        # ------------------------------------------------
                        st.markdown("### 🎞️ Original Video")
                        with open(temp_path, "rb") as f:
                            st.video(f.read(), format="video/mp4")

                        # ------------------------------------------------
                        # Show annotated video with bounding boxes
                        # ------------------------------------------------
                        st.markdown("### 🟥 Video with Model Predictions")

                        # Filter predictions for this video
                        video_preds = preds_df[preds_df["video"] == selected_video].copy()

                        if video_preds.empty:
                            st.info("No detections found for this video.")
                        else:
                            # Temporary working directory for annotated frames/video
                            work_dir = tempfile.mkdtemp(prefix="annot_frames_")
                            frames_dir = os.path.join(work_dir, "frames")
                            annotated_dir = os.path.join(work_dir, "annotated")
                            os.makedirs(frames_dir, exist_ok=True)
                            os.makedirs(annotated_dir, exist_ok=True)

                            # Extract frames with ffmpeg
                            extract_cmd = [
                                "ffmpeg",
                                "-i", temp_path,
                                os.path.join(frames_dir, "frame_%06d.jpg"),
                                "-hide_banner", "-loglevel", "error"
                            ]
                            subprocess.run(extract_cmd, check=True)

                            # -------------------------------------------
                            # Draw bounding boxes
                            # -------------------------------------------
                            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

                            for frame_file in frame_files:
                                frame_num = int(frame_file.split("_")[1].split(".")[0])

                                frame_path = os.path.join(frames_dir, frame_file)
                                img = Image.open(frame_path).convert("RGB")
                                draw = ImageDraw.Draw(img)

                                # Predictions for this frame
                                frame_preds = video_preds[video_preds["frame"] == frame_num]

                                for _, row in frame_preds.iterrows():
                                    x1, y1, x2, y2 = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

                                    # Draw bounding box
                                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                                    # Optional: draw label
                                    label = row.get("track_id", "fish")
                                    draw.text((x1, y1 - 10), str(label), fill="red")

                                # Save annotated frame
                                img.save(os.path.join(annotated_dir, frame_file))

                            # -------------------------------------------
                            # Reassemble into video
                            # -------------------------------------------
                            annotated_video_path = os.path.join(work_dir, "annotated.mp4")

                            annotate_cmd = [
                                "ffmpeg",
                                "-framerate", "25",
                                "-i", os.path.join(annotated_dir, "frame_%06d.jpg"),
                                "-c:v", "libx264",
                                "-pix_fmt", "yuv420p",
                                annotated_video_path,
                                "-hide_banner", "-loglevel", "error"
                            ]
                            subprocess.run(annotate_cmd, check=True)

                            # -------------------------------------------
                            # Display annotated video
                            # -------------------------------------------
                            with open(annotated_video_path, "rb") as f:
                                st.video(f.read(), format="video/mp4")

                            st.success("Annotated QC video generated!")

            except Exception as e:
                st.error(f"Error saving: {e}")