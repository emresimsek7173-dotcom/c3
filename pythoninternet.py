import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
import io
import os

# --- FİZİKSEL AYARLAR (C3 STANDARTLARI) ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 200          

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.zi_cache = None
        self.grid_x = None
        self.grid_y = None
        self.fig = None
        self.info_text_str = ""
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            # Sayısal olmayanları temizle
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # S1_Z ve S2_Z farkını al (Gradiometre mantığı)
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            # Satır bazlı gürültü temizleme
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Veri yükleme hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

    def detect_targets(self, zi, gain_val):
        try:
            threshold = self.std_noise * 3.5 
            binary = np.abs(zi) > (threshold * (gain_val / 100))
            labeled, num_features = label(binary)
            targets = []
            
            xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
            yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)

            for i in range(1, num_features + 1):
                mask = labeled == i
                if np.sum(mask) < 5: continue
                coords = np.argwhere(mask)
                y_idx, x_idx = coords.mean(axis=0).astype(int)
                real_x = xi[min(x_idx, len(xi)-1)]
                real_y = yi[min(y_idx, len(yi)-1)]
                val = zi[mask].mean()
                targets.append({'id': i, 'amp': val, 'x': real_x, 'y': real_y})
            return sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]
        except:
            return []

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        self.fig, self.ax2d = plt.subplots(figsize=(10, 7))
        
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((self.df['satir'], self.df['sutun']), self.df['clean_diff'] * gain, (self.grid_x, self.grid_y), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi) # NaN değerleri 0 yap (Hata önleyici)

        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # --- MOD SEÇİMLERİ (HATASIZ ANALİTİK) ---
        if mode == 'Analytic':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        self.zi_cache = zi
        lim = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-lim if mode=='Raw' else 0, vcenter=0 if mode=='Raw' else None, vmax=lim)
        
        im = self.ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='turbo', norm=norm, aspect='auto')
        plt.colorbar(im, ax=self.ax2d, label="Siddet")
        
        self.ax2d.set_xlabel("Yan (X)")
        self.ax2d.set_ylabel("Ileri (Y)")
        self.ax2d.set_title(f"C3 Analizi - {mode}")

        targets = self.detect_targets(zi, gain)
        txt = f"SABIT AYAK: {YUKSEKLIK_SABITI*100}cm | SENSOR: {SENSOR_MESAFESI*100}cm\n\n"
        txt += "TESPITLER (X, Y, Derinlik):\n"
        txt += "-" * 30 + "\n"
        for t in targets:
            d = self.calculate_depth(t['amp']/gain)
            txt += f"H#{t['id']}: {t['x']:.1f}m, {t['y']:.1f}m -> {d:.2f}m\n"
        self.info_text_str = txt

def main():
    st.set_page_config(page_title="C3 Web Pro", layout="wide")
    st.title("🛰️ C3 Gradiometre Analiz İstasyonu")

    # SOL MENÜ (Sliderlar)
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        gain = st.slider("Gain (Kazanc)", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik Filtresi", 0, 1000, 30)
        blur = st.slider("Yumusatma (Blur)", 0.0, 5.0, 1.0)
        med = st.slider("Nokta Temizleme (Median)", 0, 9, 3)
        mode = st.radio("Analiz Modu", ('Raw', 'Analytic', 'Gradient'))
        st.divider()
        st.info("Mobil kullanıcılarda sol üstteki '>' simgesiyle bu menüye ulaşılabilir.")

    uploaded_file = st.file_uploader("Telefondaki CSV Dosyasını Seç", type=['csv'])

    if uploaded_file:
        analiz = C3AnalizSistemi(uploaded_file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🖼️ 2D Harita")
            st.pyplot(analiz.fig)
            
        with col2:
            st.subheader("🎯 Derinlik Hesabı")
            st.code(analiz.info_text_str)
            
            if st.button("📥 PDF RAPORU"):
                # Basit PDF Üretimi (Reportlab gerektirir)
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet
                buf = io.BytesIO()
                analiz.fig.savefig("temp.png")
                doc = SimpleDocTemplate(buf, pagesize=A4)
                elements = [Paragraph("C3 Analiz Raporu", getSampleStyleSheet()['Title']), Spacer(1,12), Image("temp.png", width=400, height=300), Spacer(1,12), Paragraph(analiz.info_text_str.replace('\n', '<br/>'), getSampleStyleSheet()['Normal'])]
                doc.build(elements)
                st.download_button("Raporu Indir", data=buf.getvalue(), file_name="C3_Rapor.pdf")

        # 3D GÖRÜNÜM
        st.divider()
        st.subheader("🌋 3D İnteraktif Görünüm")
        fig3d = go.Figure(data=[go.Surface(
            z=analiz.zi_cache, 
            x=analiz.grid_x[0, :], 
            y=analiz.grid_y[:, 0],
            colorscale='Turbo'
        )])
        fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.4)), height=600)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
