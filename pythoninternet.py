import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from fpdf import FPDF
import io
import datetime

# --- SENİN DOKUNULMAZ FORMÜLLERİN ---
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

def create_pdf(targets, mode, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="C3 GRADIOMETRE ANALIZ RAPORU", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Tarih: {date_str} | Mod: {mode}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(40, 10, "Hedef #", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Yan (Satir)", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Ileri (Sutun)", 1, 0, 'C', 1)
    pdf.cell(50, 10, "Derinlik (m)", 1, 1, 'C', 1)
    
    for i, t in enumerate(targets):
        pdf.cell(40, 10, str(i+1), 1)
        pdf.cell(50, 10, f"{t['x']:.1f}", 1)
        pdf.cell(50, 10, f"{t['y']:.1f}", 1)
        pdf.cell(50, 10, f"{t['d']:.2f} m", 1, 1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "Not: Kirmizi bolgeler manyetik siddetin yuksek (pozitif), mavi bolgeler dusuk (negatif) oldugu yerleri gosterir. Bosluklar genellikle ani negatif dususlerle (mavi) belli olur.")
    
    return pdf.output(dest='S').encode('latin-1')

def main():
    st.set_page_config(page_title="C3 Ultimate", layout="wide")
    st.title("🛰️ C3 Gradiometre - Profesyonel Analiz & PDF Rapor")

    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")
        gain = st.slider("Manyetik Gain", 1, 1000, 250)
        filt = st.slider("Gurultu Esigi", 0, 500, 20)
        mode = st.radio("Analiz Modu", ('Raw (Bosluk+Metal)', 'Analytic (Net Hedef)', 'Gradient (Kenar)'))
        
        st.divider()
        st.subheader("🛠️ Goruntu Isleme")
        sharp = st.checkbox("Keskinlestirme (Sharpness)", value=True)
        med_filt = st.checkbox("Median (Kumlanma Giderici)", value=True)

    file = st.file_uploader("Telefondaki CSV Dosyasini Yukle", type=['csv'])
    
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

        # Gelişmiş Filtreler
        if med_filt: zi = median_filter(zi, size=3)
        if sharp:
            blurred_zi = gaussian_filter(zi, sigma=1)
            zi = zi + (zi - blurred_zi) * 1.5 # Unsharp mask
            
        zi = np.where(np.abs(zi) < filt, 0, zi)
        
        if mode == 'Analytic (Net Hedef)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient (Kenar)':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- HARITA ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if 'Raw' in mode else Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        # MANUEL GRID
        u_satir, u_sutun = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
        for x in u_satir: ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.3)
        for y in u_sutun: ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.3)
        ax.set_xticks(u_satir); ax.set_yticks(u_sutun)
        
        plt.colorbar(im, label="Manyetik Siddet (nT)")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEF ANALIZI ---
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

        # --- RAPORLAMA VE PDF ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Hedef Listesi")
            for i, t in enumerate(targets):
                tip = "🔵 BOŞLUK/EKSİ" if t['amp'] < 0 else "🔴 METAL/ARTI"
                st.info(f"**Hedef {i+1}** | Konum: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m** | Tip: {tip}")
        
        with col2:
            st.subheader("📋 Rapor Al")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if st.button("📄 PDF Raporu Olustur"):
                pdf_bytes = create_pdf(targets, mode, now)
                st.download_button(label="📥 PDF'i Indir", data=pdf_bytes, 
                                   file_name=f"C3_Rapor_{datetime.datetime.now().strftime('%H%M')}.pdf", mime="application/pdf")

        # 3D
        st.divider()
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
