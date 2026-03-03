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

# --- SENİN SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 300 

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def create_pdf_reportlab(targets, mode):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Başlık
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 50, "C3 GRADIOMETRE ANALIZ RAPORU")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height - 70, f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Mod: {mode}")
    
    # Tablo Çizimi
    y = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Hedef #")
    c.drawString(120, y, "Yan (Satir)")
    c.drawString(220, y, "Ileri (Sutun)")
    c.drawString(320, y, "Derinlik (m)")
    c.drawString(420, y, "Tip")
    
    c.line(50, y-5, 550, y-5)
    
    c.setFont("Helvetica", 11)
    for i, t in enumerate(targets):
        y -= 25
        tur = "BOSLUK" if t['amp'] < 0 else "METAL"
        c.drawString(50, y, f"#{i+1}")
        c.drawString(120, y, f"{t['x']:.1f}")
        c.drawString(220, y, f"{t['y']:.1f}")
        c.drawString(320, y, f"{t['d']:.2f} m")
        c.drawString(420, y, tur)
        
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "* Bu rapor C3 Gradiometre yazilimi tarafindan otomatik uretilmistir.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="C3 Ultimate ReportLab", layout="wide")
    st.title("🛰️ C3 Gradiometre - Adım & Derinlik Analizi")

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        gain = st.slider("Hassasiyet (Gain)", 1, 1000, 300)
        filt = st.slider("Esik Değeri", 0, 500, 30)
        mode = st.radio("Analiz Tipi", ('Raw (Bosluk ve Metal)', 'Analytic (Sadece Hedef)', 'Gradient (Kenarlar)'))
        st.divider()
        sharp = st.checkbox("Keskinlestirme Filtresi", value=True)

    file = st.file_uploader("CSV Verisini Sec reisim", type=['csv'])
    
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

        if sharp:
            blurred = gaussian_filter(zi, sigma=1)
            zi = zi + (zi - blurred) * 2.0
            
        zi = np.where(np.abs(zi) < filt, 0, zi)
        
        if mode == 'Analytic (Sadece Hedef)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient (Kenarlar)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- HARITA ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if 'Raw' in mode else Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        # OTOMATİK GRID
        u_satir, u_sutun = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
        for x in u_satir: ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.3)
        for y in u_sutun: ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.3)
        ax.set_xticks(u_satir); ax.set_yticks(u_sutun)
        
        plt.colorbar(im, label="Manyetik Siddet (nT)")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEF ANALİZİ ---
        threshold = (analiz.std_noise) * 4
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 3: continue
            val, coords = zi[mask].mean(), np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'x': xi[x_idx], 'y': yi[y_idx], 'd': d, 'amp': val})
        
        targets = sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

        # --- RAPORLAMA ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Hedef Analiz Raporu")
            for i, t in enumerate(targets):
                tip = "🔵 BOSLUK / EKSI" if t['amp'] < 0 else "🔴 METAL / ARTI"
                st.info(f"**Hedef #{i+1}** | Adım: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m** | {tip}")
        
        with col2:
            st.subheader("📄 PDF Indir")
            if st.button("Raporu Hazirla"):
                pdf_data = create_pdf_reportlab(targets, mode)
                st.download_button(label="📥 PDF Dosyasini Kaydet", data=pdf_data, 
                                   file_name=f"C3_Rapor_{datetime.datetime.now().strftime('%H%M')}.pdf", mime="application/pdf")

        # 3D
        st.divider()
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
