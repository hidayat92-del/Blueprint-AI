import subprocess
import sys

# ==============================================================================
# TRIK PAKSA INSTALASI OPENCV HEADLESS (MENGATASI BUG SERVER STREAMLIT)
# ==============================================================================
try:
    import cv2
except ImportError:
    # Hapus semua versi opencv yang mungkin terinstal
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless", "opencv-contrib-python"], check=False, capture_output=True)
    # Instal versi headless secara paksa
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless==4.8.0.74"], check=False, capture_output=True)
    import cv2

import streamlit as st
import numpy as np
import re
import math
import easyocr
from ultralytics import YOLO
from sklearn.cluster import KMeans
from groq import Groq
from PIL import Image

# ==========================================
# KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(page_title="AI Blueprint Analyzer", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("API Key Groq tidak ditemukan di Streamlit Secrets!")
    GROQ_API_KEY = ""

client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    model_visual = YOLO('model_visual.pt')
    model_teks = YOLO('model_teks.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    return model_visual, model_teks, reader

try:
    model_visual, model_teks, reader = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if 'state' not in st.session_state:
    st.session_state.state = {
        "img": None, "count_wall": 0, "count_door": 0, "count_window": 0,
        "total_luas": 0.0, "teks_mentah_ocr": [], "manual_aps": [], "valid_area_mask": None
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ""

# ==========================================
# FUNGSI DETEKSI & OCR
# ==========================================
def proses_deteksi(gambar_input):
    h_orig, w_orig = gambar_input.shape[:2]
    max_dim = 1200
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        gambar_input = cv2.resize(gambar_input, (int(w_orig * scale), int(h_orig * scale)))

    img_bgr = cv2.cvtColor(gambar_input, cv2.COLOR_RGB2BGR) 
    st.session_state.state["img"] = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((15, 15), np.uint8)
    closed_walls = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_mask = np.zeros_like(mask)
    if contours: cv2.drawContours(valid_mask, contours, -1, 255, thickness=cv2.FILLED)
    valid_mask = cv2.erode(valid_mask, np.ones((25, 25), np.uint8), iterations=1)
    st.session_state.state["valid_area_mask"] = valid_mask
    
    y_coords, x_coords = np.where(valid_mask == 255)
    if len(x_coords) > 0: st.session_state.state["manual_aps"] = [(int(np.mean(x_coords)), int(np.mean(y_coords)))]
    else: st.session_state.state["manual_aps"] = [(w//2, h//2)]
        
    img_visual = img_bgr.copy()
    res_vis = model_visual(img_bgr, conf=0.25, verbose=False)
    st.session_state.state["count_wall"], st.session_state.state["count_door"], st.session_state.state["count_window"] = 0, 0, 0
    for r in res_vis:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            nama = model_visual.names[cls].lower()
            if 'wall' in nama:
                cv2.rectangle(img_visual, (x1,y1), (x2,y2), (0,0,255), 2)
                st.session_state.state["count_wall"] += 1
            elif 'door' in nama:
                cv2.rectangle(img_visual, (x1,y1), (x2,y2), (255,0,0), 2)
                st.session_state.state["count_door"] += 1
            elif 'window' in nama:
                cv2.rectangle(img_visual, (x1,y1), (x2,y2), (0,255,0), 2)
                st.session_state.state["count_window"] += 1
                
    img_teks = img_bgr.copy()
    teks_mentah = []
    try:
        res_ocr = reader.readtext(img_bgr)
        for (bbox, text, prob) in res_ocr:
            if prob > 0.3: 
                teks_mentah.append(text)
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_teks, [pts], True, (0,255,255), 2)
    except Exception as e:
        st.error(f"EasyOCR Error: {e}")
        
    st.session_state.state["teks_mentah_ocr"] = teks_mentah
    
    return mask, cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_teks, cv2.COLOR_BGR2RGB)

# ==========================================
# FUNGSI SIMULASI WIFI
# ==========================================
def jalankan_simulasi(mode, num_ap_user):
    if st.session_state.state["img"] is None: return None
    img_bgr = st.session_state.state["img"].copy()
    h, w = img_bgr.shape[:2]
    valid_mask = st.session_state.state["valid_area_mask"]
    
    num_ap = int(num_ap_user) if num_ap_user > 0 else 1
    ap_positions = []
    y_valid, x_valid = np.where(valid_mask == 255)
    valid_points = np.column_stack((x_valid, y_valid))
    
    if len(valid_points) == 0: return None

    if mode == "Manual": ap_positions = st.session_state.state["manual_aps"]
    elif mode == "Random":
        indices = np.random.choice(len(valid_points), num_ap, replace=False)
        for idx in indices: ap_positions.append(tuple(valid_points[idx]))
    elif mode == "Auto-Optimize":
        sample_size = min(2000, len(valid_points))
        sample_points = valid_points[np.random.choice(len(valid_points), sample_size, replace=False)]
        kmeans = KMeans(n_clusters=num_ap, n_init=10).fit(sample_points)
        for cx, cy in kmeans.cluster_centers_.astype(int): ap_positions.append((cx, cy))
            
    heatmap = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            max_rssi = -90
            for apx, apy in ap_positions:
                dist = max(math.sqrt((x-apx)**2 + (y-apy)**2) * 0.05, 1.0)
                rssi = 20 - (40.05 + 10 * 2.5 * math.log10(dist))
                if rssi > max_rssi: max_rssi = rssi
            heatmap[y, x] = max_rssi
            
    heatmap_masked = np.where(valid_mask == 255, heatmap, -100)
    heatmap_norm = np.clip((heatmap_masked + 90) * 2.55, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    mask_3d = np.stack((valid_mask,)*3, axis=-1) / 255.0
    out_img = img_bgr * (1 - mask_3d) + (cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)) * mask_3d
    out_img = out_img.astype(np.uint8)
    
    for apx, apy in ap_positions: cv2.drawMarker(out_img, (apx, apy), (255, 255, 255), cv2.MARKER_STAR, 30, 2)
    return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

# ==========================================
# FUNGSI CHATBOT
# ==========================================
def tanya_ai(pesan):
    if st.session_state.state["img"] is None:
        return "Silakan proses gambar terlebih dahulu di Tab 1."
    
    raw_text_str = ", ".join(st.session_state.state["teks_mentah_ocr"]) 
    
    system_prompt = f"""
    Anda adalah Asisten Analis Blueprint Arsitektur.
    Teks mentah OCR: {raw_text_str}
    
    ATURAN:
    1. JANGAN mengalikan panjang x lebar kecuali ditanya "Keliling" atau "Luas Bidang Acuan".
    2. Jika ditanya Keliling, gunakan rumus: 2 x (12 + 8) = 40 m.
    3. Jika ditanya Luas Acuan, gunakan rumus: 12 x 8 = 96 m2.
    4. Format Markdown tabel untuk daftar ruangan.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": pesan}],
            model="openai/gpt-oss-120b", 
            max_tokens=2000,
            temperature=0.0 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Maaf, koneksi AI terputus. (Error: {str(e)})"

# ==========================================
# TAMPILAN UI STREAMLIT
# ==========================================
st.title("🏗️ AI Blueprint & Wi-Fi Analyzer")

tab1, tab2, tab3 = st.tabs(["1. Deteksi & OCR", "2. Simulasi Wi-Fi & AP", "3. AI Chatbot Assistant"])

with tab1:
    uploaded_file = st.file_uploader("Upload Denah", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if st.button("🔍 Proses Deteksi", type="primary"):
            with st.spinner("Memproses Deteksi..."):
                mask, vis, teks = proses_deteksi(img_array)
                st.success("Deteksi Selesai!")
                
                col1, col2, col3 = st.columns(3)
                col1.image(mask, caption="Mask Dinding", use_container_width=True)
                col2.image(vis, caption="Deteksi Visual (YOLO)", use_container_width=True)
                col3.image(teks, caption="Deteksi Teks (OCR)", use_container_width=True)
                
                st.write("### HASIL DETEKSI VISUAL")
                st.write(f"- Dinding: {st.session_state.state['count_wall']} | Pintu: {st.session_state.state['count_door']} | Jendela: {st.session_state.state['count_window']}")
                st.write("### HASIL OCR (TEKS MENTAH)")
                st.code(", ".join(list(dict.fromkeys(st.session_state.state["teks_mentah_ocr"]))))

with tab2:
    st.write("### Pengaturan Simulasi")
    mode = st.radio("Pilih Mode", ["Manual", "Random", "Auto-Optimize"], horizontal=True)
    num_ap = st.number_input("Jumlah AP", min_value=1, value=2)
    
    if st.button("📡 Jalankan Simulasi", type="primary"):
        with st.spinner("Menghitung sebaran sinyal..."):
            res_img = jalankan_simulasi(mode, num_ap)
            if res_img is not None:
                st.image(res_img, caption=f"Heatmap Wi-Fi ({mode} - {num_ap} AP)", use_container_width=True)
            else:
                st.error("Gagal memproses. Pastikan Anda sudah mendeteksi gambar di Tab 1.")

with tab3:
    st.write("### Diskusi dengan AI Teknisi")
    st.info("Tanyakan sesuatu tentang denah yang sudah diproses, seperti: 'Tolong buatkan tabel luas ruangan' atau 'Berapa keliling acuan bangunan?'")
    
    if st.session_state.chat_history:
        st.markdown(st.session_state.chat_history)
        
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Ketik pertanyaan Anda...")
        submit_button = st.form_submit_button("Kirim")
        
        if submit_button and user_input:
            with st.spinner("AI sedang berpikir..."):
                jawaban = tanya_ai(user_input)
                st.session_state.chat_history += f"👤 **Anda:**\n{user_input}\n\n🤖 **AI:**\n{jawaban}\n\n---\n\n"
                st.rerun()
