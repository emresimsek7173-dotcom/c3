import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import datetime
import os
import io

# --- FİZİKSEL SABİTLER ---
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
MAX_DEPTH = 10.0
GRID_RES = 180

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="C3 SAHA ANALİZ V2", layout="wide")
st.title("🛡️ C3 Manyetik Gradiometre Analiz Sistemi")

# --- SIDEBAR (KONTROL PANELİ) ---
st.sidebar.header("🛠️ Analiz Parametreleri")
gain = st.sidebar.slider("Kazanç (Gain)", 1.0, 1000.0, 100.0)
esik = st.sidebar.slider("Eşik (Threshold)", 0, 500, 20)
blur = st.sidebar.slider("Yumuşatma (Gaussian)", 0.0, 4.0, 0.8)
med_size = st.sidebar.slider("Nokta Filtre (Median)", 0, 9, 3, step=2)
mode = st.sidebar.selectbox("Analiz Modu", ("Raw", "Analytic", "Gradient"))

# --- YARDIMCI FONKSİYONLAR ---
def calculate_depth(peak_nt, shape_factor=2.8):
    grad = abs(peak_nt) / SENSOR_MESAFESI
    if grad < 0.005: return 0.0
    depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.05))
    return min(depth, MAX_DEPTH)

def detect_targets(zi, std_noise, xi, yi):
    threshold = std_noise * 3.0
    binary = np.abs(zi) > (threshold * (gain / 50))
    labeled, num_features = label(binary)
    targets = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        if np.sum(mask) < 3: continue
        coords = np.argwhere(mask)
        y_idx, x_idx = coords.mean(axis=0).astype(int)
        real_x = xi[min(x_idx, len(xi)-1)]
        real_y = yi[min(y_idx, len(yi)-1)]
        val = zi[mask].mean()
        targets.append({'id': i, 'amp': val, 'x': real_x, 'y': real_y})
    return sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

# --- VERİ YÜKLEME ---
uploaded_file = st.file_uploader("CSV Veri Dosyasını Seçin", type=["csv"])

if uploaded_file is not None:
    try:
        # Veri Temizleme (Orijinal mantık)
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        df['raw_diff'] = df['s1_z'] - df['s2_z']
        df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
        
        for r in df['satir'].unique():
            mask = df['satir'] == r
            if sum(mask) > 5:
                df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
        
        std_noise = df['clean_diff'].std()

        # Gridleme
        xi = np.linspace(df['satir'].min(), df['satir'].max(), GRID_RES)
        yi = np.linspace(df['sutun'].min(), df['sutun'].max(), GRID_RES)
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((df['satir'], df['sutun']), df['clean_diff'] * gain, (grid_x, grid_y), method='linear', fill_value=0)
        
        # Filtreler
        if med_size > 0: zi = median_filter(zi, size=med_size)
        zi = np.where(np.abs(zi) < esik, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # Modlar
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2 + zi**2)
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2)

        # Hedef Tespiti
        targets = detect_targets(zi, std_noise, xi, yi)

        # --- GÖRSELLEŞTİRME (SEKMELER) ---
        tab1, tab2 = st.tabs(["📊 2D Analiz & Kesit", "🧊 3D İnteraktif Görünüm"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 2D Harita
                fig2d, ax2d = plt.subplots(figsize=(10, 7))
                plt.style.use('dark_background')
                lim = max(np.abs(zi).max(), 1)
                norm = TwoSlopeNorm(vmin=-lim if mode=='Raw' else 0, vcenter=0 if mode=='Raw' else None, vmax=lim)
                
                im = ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                                origin='lower', cmap='turbo', norm=norm, aspect='auto')
                plt.colorbar(im, ax=ax2d, label="Manyetik Şiddet")
                
                # Hedef İşaretleme (İstediğin X ve H1, H2 etiketleri)
                for t in targets:
                    ax2d.plot(t['x'], t['y'], 'kx', markersize=10, markeredgewidth=2)
                    ax2d.text(t['x']+0.05, t['y']+0.05, f"H{t['id']}", color='white', 
                              fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
                
                ax2d.set_title(f"Saha 2D Haritası ({mode} Modu)")
                ax2d.set_xlabel("Yan Mesafe (Satır)")
                ax2d.set_ylabel("İleri Mesafe (Sütun)")
                st.pyplot(fig2d)

            with col2:
                # Hedef Tablosu
                st.subheader("🎯 Tespit Edilen Hedefler")
                target_data = []
                for t in targets:
                    d = calculate_depth(t['amp']/gain)
                    target_data.append({
                        "ID": f"H{t['id']}",
                        "X (Yan)": f"{t['x']:.2f}m",
                        "Y (İleri)": f"{t['y']:.2f}m",
                        "Derinlik": f"{d:.2f}m"
                    })
                st.table(pd.DataFrame(target_data))
                
                # Histogram
                fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                ax_hist.hist(zi.flatten(), bins=30, color='#00ff9d', alpha=0.6)
                ax_hist.set_title("Sinyal Dağılımı")
                st.pyplot(fig_hist)

        with tab2:
            st.subheader("🧊 3D Yüzey Analizi (Mouse ile Döndürülebilir)")
            # Plotly 3D Surface
            fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='Satır (Yan)',
                    yaxis_title='Sütun (İleri)',
                    zaxis_title='Şiddet'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                width=900, height=700
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # --- PDF RAPORLAMA ---
        st.sidebar.markdown("---")
        if st.sidebar.button("📄 PDF Raporu Oluştur"):
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Matplotlib grafiğini PDF için kaydet
            fig2d.savefig("temp_report.png", bbox_inches='tight', dpi=150)
            
            elements = [
                Paragraph("C3 Manyetik Gradiometre Analiz Raporu", styles['Title']),
                Paragraph(f"Tarih: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']),
                Spacer(1, 15),
                Image("temp_report.png", width=450, height=300),
                Spacer(1, 15),
                Paragraph("Tespit Edilen Anomaliler:", styles['Heading2'])
            ]
            
            # Hedefleri PDF'e ekle
            for t in target_data:
                elements.append(Paragraph(f"ID: {t['ID']} | Konum: {t['X (Yan)']}, {t['Y (İleri)']} | Tahmini Derinlik: {t['Derinlik']}", styles['Normal']))
            
            doc.build(elements)
            st.sidebar.success("✅ Rapor Hazır!")
            st.sidebar.download_button(
                label="📥 PDF İndir",
                data=pdf_buffer.getvalue(),
                file_name=f"C3_Analiz_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
            os.remove("temp_report.png")

    except Exception as e:
        st.error(f"⚠️ Bir hata oluştu: {e}")

else:
    st.info("💡 Analize başlamak için lütfen bir CSV dosyası yükleyin.")
