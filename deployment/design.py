import streamlit as st
import os
import io
import subprocess
from PIL import Image
from pdf2image import convert_from_bytes

# === CONFIG ===
IMAGE_DIR = "mc_ocr/data/inference/image"
BATCH_FILE = "inference.bat"

# Output folders
DETECT_VIZ = "mc_ocr/data/inference/text_detector/viz_imgs"
ROTATE_VIZ = "mc_ocr/data/inference/rotation_corrector/imgs"
OCR_VIZ = "mc_ocr/data/inference/text_classifier/viz_imgs"
OCR_TXT = "mc_ocr/data/inference/text_classifier/txt"

# === Initialize session state ===
session_defaults = {
    "converted_filename": "",
    "ocr_text": "",
    "ocr_img_path": "",
    "detect_img_path": "",
    "rotate_img_path": ""
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(page_title="OCR Pipeline Viewer", layout="centered")
st.title("📄 Vietnamese OCR Pipeline UI")

SUPPORTED_EXTS = ["jpg", "jpeg", "png", "webp", "tiff", "bmp", "heic", "pdf"]

def convert_to_jpg(uploaded_file, save_dir):
    filename_raw = uploaded_file.name
    name_wo_ext = os.path.splitext(filename_raw)[0]
    save_path = os.path.join(save_dir, f"{name_wo_ext}.jpg")

    file_bytes = uploaded_file.read()
    file_ext = filename_raw.split('.')[-1].lower()

    if file_ext == "pdf":
        # Convert first page of PDF to image
        images = convert_from_bytes(file_bytes)
        if images:
            img = images[0].convert("RGB")
            img.save(save_path, format="JPEG")
        else:
            raise ValueError("PDF has no pages")
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img.save(save_path, format="JPEG")

    return name_wo_ext + ".jpg"

def find_image_by_name(folder, filename):
    path = os.path.join(folder, filename)
    return path if os.path.exists(path) else None

def find_text_by_name(folder, filename):
    txt_name = os.path.splitext(filename)[0] + ".txt"
    path = os.path.join(folder, txt_name)
    return path if os.path.exists(path) else None

# === Upload and Convert ===
uploaded_file = st.file_uploader("🖼️ Upload an image or PDF", type=SUPPORTED_EXTS)
if uploaded_file:
    converted_filename = convert_to_jpg(uploaded_file, IMAGE_DIR)
    input_path = os.path.join(IMAGE_DIR, converted_filename)
    # st.image(Image.open(input_path), caption=f"📤 Converted: {converted_filename}", use_column_width=True)
    st.image(Image.open(input_path), use_column_width=True)
    st.session_state.converted_filename = converted_filename  # Save for reuse

    if st.button("▶️ Convert"):
        try:
            with st.spinner("⏳ Running..."):
                result = subprocess.run(
                    [BATCH_FILE, converted_filename],
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
            st.success("✅ Completed!")

            # === Save paths to session_state ===
            st.session_state.detect_img_path = find_image_by_name(DETECT_VIZ, converted_filename)
            st.session_state.rotate_img_path = find_image_by_name(ROTATE_VIZ, converted_filename)
            st.session_state.ocr_img_path = find_image_by_name(OCR_VIZ, converted_filename)
            ocr_text_path = find_text_by_name(OCR_TXT, converted_filename)

            # === Process OCR text ===
            st.session_state.ocr_text = ""
            if ocr_text_path:
                with open(ocr_text_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                only_text_lines = []
                for line in reversed(lines):
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        text = ','.join(parts[8:]).strip()  # Gộp phần text (nếu có dấu phẩy)
                        if text:
                            only_text_lines.append(text)

                st.session_state.ocr_text = '\n'.join(only_text_lines)

        except subprocess.CalledProcessError as e:
            st.error("❌ Execution failed!")
            st.text(e.stderr)

# === Show Results if available ===
if st.session_state.converted_filename:
    st.subheader("🧾 Results")

    with st.expander("🔍 Text Detection Result"):
        if st.session_state.detect_img_path:
            st.image(st.session_state.detect_img_path, caption="Text Detection Result", use_column_width=True)
        else:
            st.warning("Detection image not found.")

    with st.expander("🔄 Rotation Correction Result"):
        if st.session_state.rotate_img_path:
            st.image(st.session_state.rotate_img_path, caption="Rotation Correction Result", use_column_width=True)
                # Open and convert image to bytes
            with open(st.session_state.rotate_img_path, "rb") as img_file:
                img_bytes = img_file.read()

            # Add download button
            st.download_button(
                label="📥 Download Rotation Image",
                data=img_bytes,
                file_name="rotation_result.jpg",
                mime="image/jpeg"
            )
        else:
            st.warning("Rotation image not found.")

    with st.expander("✍️ OCR Output"):
        if st.session_state.ocr_img_path:
            st.image(st.session_state.ocr_img_path, caption="OCR Image Result", use_column_width=True)
        else:
            st.warning("OCR image not found.")

        if st.session_state.ocr_text:
            st.text_area("📋 OCR Text Output", st.session_state.ocr_text, height=300)
            st.download_button("📥 Download as .txt", st.session_state.ocr_text, file_name="ocr_output.txt")
            st.download_button("📄 Download as .doc", st.session_state.ocr_text, file_name="ocr_output.doc")
        else:
            st.warning("OCR text not available.")
