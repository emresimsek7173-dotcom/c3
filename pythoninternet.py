import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, sobel
from scipy.signal import detrend

# --- SENİN BİLGİSAYAR KODUNDAKİ SABİTLER ---
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
GRID_RES = 200 # Senin orijinal çözünürlüğün

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # --- SENİN ORİJİNAL HESAPLAMA BLOĞUN ---
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            # Senin orijinal 5 adım kuralın (Buraya dokunmadım)
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            
            self.df = df
        except Exception as e:
            st.error(f"Veri yükleme hatası: {e}")

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        # Sadece 2D Harita - Kocaman Ekran
        self.fig, self.ax2d = plt.subplots(figsize=(16, 10))
        
        # Senin orijinal gridleme mantığın
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((self.df['satir'], self.df['sutun']), 
                      self.df['clean_diff'] * gain, 
                      (self.grid_x, self.grid_y), 
                      method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # --- YÖN DÜZELTME (Sola Bakması İçin) ---
        # Eğer bu yön hala yanlışsa, sadece tek bir satırla değiştireceğiz
        zi = np.rot90(zi, k=1) 
        ext = [yi.min(), yi.max(), xi.min(), xi.max()]

        # Senin orijinal filtreleme sıran
        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2 + zi**2)
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2)

        self.zi_cache = zi
        lim = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-lim if mode=='Raw' else 0, vcenter=0 if mode=='Raw' else None, vmax=lim)
        
        # Senin bilgisayardaki o yayvan (aspect='auto') görüntü
        im = self.ax2d.imshow(zi, extent=ext, origin='lower', cmap='turbo', norm=norm, aspect='auto')
        self.fig.colorbar(im, ax=self.ax2d)
        self.ax2d.set_xlabel("İLERİ (Sutun)"); self.ax2d.set_ylabel("YAN (Satir)")

def main():
    st.set_page_config(page_title="C3 Saf Analiz", layout="wide")
    st.title("🛰️ C3 Gradiometre - Orijinal Matematik")

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))

    file = st.file_uploader("CSV Seç", type=['csv'])
    if file:
        analiz = C3AnalizSistemi(file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        st.pyplot(analiz.fig, use_container_width=True)
        
        st.divider()
        st.subheader("🌋 3D Görünüm")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
