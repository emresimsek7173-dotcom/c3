import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend

# --- FİZİKSEL SABİTLER ---
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
GRID_RES = 300 # Çözünürlüğü iyice artırdım, cam gibi olsun

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_data(file_buffer)
        
    def load_data(self, file_buffer):
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
        except Exception as e:
            st.error(f"Veri hatası: {e}")

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        # PANELİ DEVLEŞTİRDİK
        self.fig, self.ax2d = plt.subplots(figsize=(18, 11))
        
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((self.df['satir'], self.df['sutun']), self.df['clean_diff'] * gain, (grid_x, grid_y), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # REİSİN İSTEDİĞİ SOLA BAKAN YÖN (SABİT)
        zi = np.flipud(zi.T) 
        ext = [yi.min(), yi.max(), xi.min(), xi.max()]

        if med > 0:
            m_size = int(med); m_size = m_size+1 if m_size%2==0 else m_size
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
        
        im = self.ax2d.imshow(zi, extent=ext, origin='lower', cmap='turbo', norm=norm, aspect='auto')
        self.fig.colorbar(im, ax=self.ax2d, label="Manyetik Şiddet")
        self.ax2d.set_title(f"C3 ANALİZ - {mode}", fontsize=20)
        self.ax2d.set_xlabel("İLERİ (Sutun)", fontsize=14)
        self.ax2d.set_ylabel("YAN (Satir)", fontsize=14)

def main():
    st.set_page_config(page_title="C3 Dev Ekran", layout="wide")
    st.title("🛰️ C3 Gradiometre - Büyük 2D Görünüm")

    with st.sidebar:
        st.header("⚙️ Kontroller")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Eşik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))
        st.divider()
        st.info("Ok yönü sola sabitlendi, 2D ekran büyütüldü.")

    file = st.file_uploader("CSV Yükle", type=['csv'])
    if file:
        analiz = C3AnalizSistemi(file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        
        # 2D HARİTA - DEV BOYUT
        st.pyplot(analiz.fig, use_container_width=True)
        
        st.divider()
        
        # 3D GÖRÜNÜM - ALTTA
        st.subheader("🌋 3D Topografya")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, colorscale='Turbo')])
        fig3d.update_layout(height=800)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
