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
from reportlab.lib.utils import ImageReader
import io
import datetime

# --- AYARLAR ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 400 

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
            st.error(f"Veri Hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def create_full_pdf(targets, mode, fig_plot):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Başlık
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 40, "C3 GRADIOMETRE ANALIZ RAPORU")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height - 55, f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # HARİTAYI PDF'E EKLEME
    imgdata = io.BytesIO()
    fig_plot.savefig(imgdata, format='png', bbox_inches='tight', dpi=150)
    imgdata.seek(0)
    c.drawImage(ImageReader(imgdata), 50, height - 400, width=500, height=330)

    # TABLO
    y_table = height - 430
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y_table, "Hedef #")
    c.drawString(110, y_table, "Yan (Satir)")
    c.drawString(180, y_table, "Ileri (Sutun)")
    c.drawString(260, y_table, "Derinlik (m)")
    c.drawString(340, y_table, "Siddet (nT)")
    c.drawString(420, y_table, "TUR")
    c.line(50, y_table-5, 550, y_table-5)

    c.setFont("Helvetica", 10)
    for i, t in enumerate(targets):
        y_table -= 20
        tur = "BOSLUK" if t['amp'] < 0 else "METAL"
        c.drawString(50, y_table, f"#{i+1}")
        c.drawString(110, y_table, f"{t['x']:.1f}")
        c.drawString(180, y_table, f"{t['y']:.1f}")
        c.drawString(260, y_table, f"{t['d']:.2f} m")
        c.drawString(340, y_table, f"{t['amp']:.1f}")
        c.drawString(420, y_table, tur)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="C3 Master Pro", layout="wide")
    st.title("🛰️ C3 Gradiometre - Tam Donanımlı Analiz")

    with st.sidebar:
        st.header("⚙️ Kontrol Merkezi")
        gain = st.slider("Gain (Hassasiyet)", 1, 1000, 300)
        filt = st.slider("Gurultu Esigi", 0, 500, 25)
        st.divider()
        st.subheader("🖼️ Goruntu Ayarlari")
        blur_val = st.slider("Blur (Yumusatma)", 0.0, 5.0, 1.0)
        med_val = st.slider("Median (Parazit Sil)", 0, 9, 3)
        sharp_val = st.slider("Keskinlik (Sharpness)", 0.0, 3.0, 1.2)
        st.divider()
        mode = st.radio("Analiz Modu", ('Raw (Bosluk/Metal)', 'Analytic', 'Gradient'))

    file = st.file_uploader("CSV Verisini Sec", type=['csv'])
    
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

        # FİLTRE ZİNCİRİ
        if med_val > 0:
            m_size = int(med_val); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)
        if blur_val > 0:
            zi = gaussian_filter(zi, sigma=blur_val)
        if sharp_val > 0:
            blurred_ref = gaussian_filter(zi, sigma=2.0)
            zi = zi + (zi - blurred_ref) * sharp_val
            
        zi = np.where(np.abs(zi) < filt, 0, zi)
        
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- 2D ANA GÖRÜNTÜ ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8))
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if 'Raw' in mode else Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto', interpolation='bilinear')
        
        # MANUEL ADIM GRIDI (Hücreleri Gösteren)
        u_satir, u_sutun = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
        for x in u_satir: ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.25)
        for y in u_sutun: ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.25)
        ax.set_xticks(u_satir); ax.set_yticks(u_sutun)
        
        plt.colorbar(im, label="nT Siddet")
        st.pyplot(fig, use_container_width=True)

        # HEDEF TESPİTİ
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

        # RAPOR BÖLÜMÜ
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🎯 Tespit Edilen Hedefler")
            for i, t in enumerate(targets):
                tip = "🔵 BOSLUK / MEZAR" if t['amp'] < 0 else "🔴 METAL / YAPI"
                st.info(f"**Hedef #{i+1}** | Adım: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m** | {tip}")
        
        with c2:
            st.subheader("📋 PDF Rapor Al")
            if st.button("Görüntülü PDF Hazirla"):
                pdf_file = create_full_pdf(targets, mode, fig)
                st.download_button(label="📥 PDF'i Kaydet", data=pdf_file, 
                                   file_name=f"C3_Analiz_{datetime.datetime.now().strftime('%H%M')}.pdf", mime="application/pdf")

        # 3D GÖRÜNÜM
        st.divider()
        st.subheader("🌋 3D Manyetik Harita")
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
