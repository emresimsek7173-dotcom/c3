import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label
from scipy.signal import detrend
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import datetime

# --- SENİN FİZİKSEL SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 450 # Çubukların kaybolmaması için yüksek çözünürlük

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # SENİN ÖZEL TEMİZLEME ALGORİTMAN
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Veri Hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def main():
    st.set_page_config(page_title="C3 Pro Ultimate", layout="wide")
    st.title("🛰️ C3 Gradiometre - Tam Teşekküllü Analiz Paneli")

    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    with st.sidebar:
        st.header("⚙️ Kontrol Merkezi")
        gain = st.slider("Hassasiyet (Gain)", 1, 1000, 300)
        filt = st.slider("Gürültü Eşiği", 0, 500, 25)
        st.divider()
        st.subheader("🖼️ Görüntü Ayarları")
        blur_val = st.slider("Blur (Yumuşatma)", 0.0, 5.0, 1.2)
        med_val = st.slider("Median (Nokta Temizliği)", 0, 9, 3)
        st.divider()
        mode = st.radio("Analiz Modu", ('Raw (Ham Veri)', 'Analytic (Dolu Tepecik)'))

    file = st.file_uploader("CSV Verisini Seçin", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        x_min, x_max = analiz.df['satir'].min(), analiz.df['satir'].max()
        y_min, y_max = analiz.df['sutun'].min(), analiz.df['sutun'].max()
        
        xi = np.linspace(x_min, x_max, GRID_RES)
        yi = np.linspace(y_min, y_max, GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # FİLTRELEME (Sırasıyla)
        if med_val > 0:
            m_size = int(med_val); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)

        if mode == 'Analytic (Dolu Tepecik)':
            # Boş boru yapmayan, senin istediğin sivri çubukları koruyan formül
            dx, dy = np.gradient(zi)
            zi = np.sqrt(zi**2 + dx**2 + dy**2)
        
        if blur_val > 0:
            zi = gaussian_filter(zi, sigma=blur_val)
            
        zi = np.where(np.abs(zi) < filt, 0, zi)

        # --- ANA EKRAN (2D HARİTA VE HISTOGRAM) ---
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            st.subheader("🗺️ 2D Manyetik Harita")
            fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
            plt.style.use('dark_background')
            im = ax_2d.imshow(zi, extent=[x_min-0.5, x_max+0.5, y_min-0.5, y_max+0.5], 
                           origin='lower', cmap='turbo', interpolation='bilinear', aspect='auto')
            
            # Grid çizgilerini (9 kare vb.) geri getirdim
            u_s, u_t = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
            for x in u_s: ax_2d.axvline(x - 0.5, color='white', linestyle='-', alpha=0.2)
            for y in u_t: ax_2d.axhline(y - 0.5, color='white', linestyle='-', alpha=0.2)
            
            plt.colorbar(im, label="nT")
            st.pyplot(fig_2d)

        with col_side:
            st.subheader("📊 Veri Dağılımı")
            fig_hist, ax_hist = plt.subplots(figsize=(4, 6))
            ax_hist.hist(zi.flatten(), bins=30, color='lime', alpha=0.7)
            ax_hist.set_yscale('log')
            st.pyplot(fig_hist)

        # --- HEDEF ANALİZİ VE DERİNLİK ---
        st.divider()
        threshold = (analiz.std_noise) * 4
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 4: continue
            val = zi[mask].mean()
            coords = np.argwhere(mask)
            y_i, x_i = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'x': xi[x_i], 'y': yi[y_i], 'd': d, 'amp': val})
        
        targets = sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🎯 Tespitler")
            for i, t in enumerate(targets):
                tur = "🔵 BOŞLUK" if t['amp'] < 0 else "🔴 METAL"
                st.info(f"**H#{i+1}** | Koordinat: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m** | {tur}")
        
        with c2:
            st.subheader("📋 Raporlama")
            if st.button("Haritalı PDF Oluştur"):
                # PDF Oluşturma fonksiyonu (İçinde resimle)
                buffer = io.BytesIO()
                pdf = canvas.Canvas(buffer, pagesize=A4)
                pdf.setFont("Helvetica-Bold", 16)
                pdf.drawString(50, 800, "C3 GRADIOMETRE SAHA RAPORU")
                
                # Resmi PDF'e ekle
                img_data = io.BytesIO()
                fig_2d.savefig(img_data, format='png')
                img_data.seek(0)
                pdf.drawImage(ImageReader(img_data), 50, 450, width=500, height=330)
                
                # Tabloyu ekle
                pdf.setFont("Helvetica", 10)
                y_pos = 420
                for i, t in enumerate(targets):
                    pdf.drawString(50, y_pos, f"H#{i+1} - Yan:{t['x']:.1f} Ileri:{t['y']:.1f} Derinlik:{t['d']:.2f}m")
                    y_pos -= 20
                
                pdf.showPage(); pdf.save()
                st.download_button("📥 PDF İndir", buffer.getvalue(), "C3_Rapor.pdf", "application/pdf")

        # --- 3D GÖRÜNÜM (Senin Çubukların Burası) ---
        st.divider()
        st.subheader("🌋 3D İnteraktif Analiz")
        fig_3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig_3d, use_container_width=True)

if __name__ == "__main__":
    main()
