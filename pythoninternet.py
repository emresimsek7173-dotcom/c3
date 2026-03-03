import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import io
import datetime

# --- AYARLAR ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 500 # PİKSELLEŞMEYİ ÖNLEMEK İÇİN ÇÖZÜNÜRLÜK ARTIRILDI

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # Veri ön işleme: Manyetik sapmaları temizle
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

def create_pdf_reportlab(targets, mode):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 50, "C3 GRADIOMETRE ANALIZ RAPORU")
    y = height - 120
    c.setFont("Helvetica", 11)
    for i, t in enumerate(targets):
        y -= 25
        tur = "BOSLUK" if t['amp'] < 0 else "METAL"
        c.drawString(50, y, f"Hedef #{i+1}: Yan:{t['x']:.1f} Ileri:{t['y']:.1f} Derinlik:{t['d']:.2f}m Tip:{tur}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="C3 Ultra-Net", layout="wide")
    st.title("🛰️ C3 Gradiometre - Ultra Net Görüntüleme")

    with st.sidebar:
        st.header("⚙️ Filtre Paneli")
        gain = st.slider("Hassasiyet (Gain)", 1, 1000, 350)
        filt = st.slider("Gürültü Eşiği", 0, 500, 20)
        
        st.divider()
        st.subheader("🖼️ Netleştirme Ayarları")
        blur_val = st.slider("Yumuşatma (Blur)", 0.0, 5.0, 0.8) # Netlik için biraz düşürdük
        med_val = st.slider("Median (Nokta Temizleme)", 0, 9, 3)
        sharpness = st.slider("Keskinlik Gücü", 0.0, 3.0, 1.2)
        
        st.divider()
        mode = st.radio("Mod", ('Raw (Normal)', 'Analytic (Noktasal)', 'Gradient (Kenar)'))

    file = st.file_uploader("CSV Verisini Seç", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        x_min, x_max = analiz.df['satir'].min(), analiz.df['satir'].max()
        y_min, y_max = analiz.df['sutun'].min(), analiz.df['sutun'].max()
        
        xi = np.linspace(x_min, x_max, GRID_RES)
        yi = np.linspace(y_min, y_max, GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        # Interpolasyon: 'cubic' yerine piksellenmeyi azaltan 'linear' ile 'cubic' harmanı mantığı
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # --- NETLEŞTİRME ZİNCİRİ ---
        if med_val > 0:
            m_size = int(med_val); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)
        
        if blur_val > 0:
            zi = gaussian_filter(zi, sigma=blur_val)

        # Unsharp Masking (Kenar Netleştirme)
        if sharpness > 0:
            blurred_img = gaussian_filter(zi, sigma=2.0)
            zi = zi + (zi - blurred_img) * sharpness
            
        zi = np.where(np.abs(zi) < filt, 0, zi)
        
        if mode == 'Analytic (Noktasal)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient (Kenar)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- GÖRSELLEŞTİRME ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100) # DPI artırıldı
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if 'Raw' in mode else Normalize(vmin=0, vmax=v_max)
        
        # 'bilinear' interpolation ekleyerek pikseller arası geçişi yumuşattık
        im = ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto', interpolation='bilinear')
        
        # Manuel Grid
        u_satir, u_sutun = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
        for x in u_satir: ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.2)
        for y in u_sutun: ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.2)
        
        ax.set_xticks(u_satir); ax.set_yticks(u_sutun)
        plt.colorbar(im, label="nT")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEFLER & PDF ---
        threshold = (analiz.std_noise) * 4
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 5: continue
            val, coords = zi[mask].mean(), np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'x': xi[min(x_idx, GRID_RES-1)], 'y': yi[min(y_idx, GRID_RES-1)], 'd': d, 'amp': val})
        
        targets = sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

        col1, col2 = st.columns([2, 1])
        with col1:
            for i, t in enumerate(targets):
                st.success(f"**Hedef #{i+1}** | Adım: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m**")
        with col2:
            if st.button("📄 Raporu İndir"):
                pdf_data = create_pdf_reportlab(targets, mode)
                st.download_button(label="📥 Tıkla ve İndir", data=pdf_data, file_name="C3_Rapor.pdf", mime="application/pdf")

        # 3D
        st.divider()
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
