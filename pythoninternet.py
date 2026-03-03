import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
import io
import os

# --- SENİN ORİJİNAL SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 200          

class C3WebSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.zi_cache = None
        self.grid_x = None
        self.grid_y = None
        self.info_text_str = ""
        self.load_data(file_buffer)
        
    def load_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # Senin orijinal "raw_diff" ve "detrend" mantığın
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Veri işleme hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        # Senin orijinal derinlik formülün
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        # Senin bilgisayardaki 16:9 oranını koruduk
        self.fig, self.ax2d = plt.subplots(figsize=(10, 8)) 
        
        # --- EKSENLER: Tam senin kodundaki gibi ---
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        # Senin "cubic" gridleme mantığın
        zi = griddata((self.df['satir'], self.df['sutun']), 
                      self.df['clean_diff'] * gain, 
                      (self.grid_x, self.grid_y), 
                      method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # Modlar (Orijinal sobel mantığın)
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2 + zi**2)
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2)

        self.zi_cache = zi
        
        # Renk normu (Hata önleyici zırhlı versiyon)
        lim = max(np.abs(zi).max(), 1.0)
        if mode == 'Raw':
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        else:
            norm = Normalize(vmin=0, vmax=lim)
        
        # --- GÖRÜNTÜ: Senin "origin=lower" ve "extent" dizilimin ---
        im = self.ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                              origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        plt.colorbar(im, ax=self.ax2d, label="Manyetik Siddet")
        self.ax2d.set_xlabel("SATIR (Yana Kayma - Metre)")
        self.ax2d.set_ylabel("SUTUN (Ileri Gidis - Metre)")
        self.ax2d.set_title(f"C3 Analiz - {mode}")

        # Hedef Tespiti (Senin orijinal 3.5*std mantığın)
        threshold = (self.std_noise if self.std_noise > 0 else 1) * 3.5 
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num_features = label(binary)
        
        txt = f"SABIT AYAK: {YUKSEKLIK_SABITI*100}cm | SENSOR: {SENSOR_MESAFESI*100}cm\n\n"
        txt += "TESPITLER (Yan, Ileri, Derinlik):\n" + "-"*38 + "\n"
        
        targets = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) < 5: continue
            coords = np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            val = zi[mask].mean()
            targets.append({'amp': val, 'x': xi[min(x_idx, len(xi)-1)], 'y': yi[min(y_idx, len(yi)-1)]})
        
        for i, t in enumerate(sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]):
            d = self.calculate_depth(t['amp']/gain)
            txt += f"H#{i+1}: Yan:{t['x']:.1f}m, Ileri:{t['y']:.1f}m -> D:{d:.1f}m\n"
        self.info_text_str = txt

def main():
    st.set_page_config(page_title="C3 Pro Web", layout="wide")
    st.title("🛰️ C3 Orijinal Analiz İstasyonu")

    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))
        st.divider()
        st.write("Bilgisayardaki orijinal kod yapısıyla çalışıyor.")

    uploaded_file = st.file_uploader("CSV Verisi Yukle", type=['csv'])

    if uploaded_file:
        analiz = C3WebSistemi(uploaded_file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.pyplot(analiz.fig)
        with c2:
            st.subheader("🎯 Saha Verileri")
            st.code(analiz.info_text_str)
            
            # PDF Raporlama (Reportlab Entegrasyonu)
            if st.button("📥 PDF RAPOR AL"):
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet
                buf = io.BytesIO()
                analiz.fig.savefig("temp.png")
                doc = SimpleDocTemplate(buf, pagesize=A4)
                parts = [Paragraph("C3 Saha Raporu", getSampleStyleSheet()['Title']), Spacer(1,12), Image("temp.png", width=400, height=300), Spacer(1,12), Paragraph(analiz.info_text_str.replace('\n', '<br/>'), getSampleStyleSheet()['Normal'])]
                doc.build(parts)
                st.download_button("Dosyayı İndir", buf.getvalue(), "C3_Rapor.pdf")

        # Senin 3D Görünüm butonunu interaktif Plotly'ye çevirdim (Parmağınla döndür diye)
        st.divider()
        st.subheader("🌋 3D Manyetik Topografya")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, x=analiz.grid_x[0,:], y=analiz.grid_y[:,0], colorscale='Turbo')])
        fig3d.update_layout(scene=dict(xaxis_title='Yan', yaxis_title='Ileri', aspectratio=dict(x=1, y=1, z=0.5)), height=700)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
