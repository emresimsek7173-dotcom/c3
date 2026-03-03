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

# --- FİZİKSEL AYARLAR ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 250 # Çözünürlüğü biraz daha artırdım

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
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # Gradiometre farkı
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            # Global ofset temizleme
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            # Satır bazlı gürültü temizleme (Zebra efektini önler)
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 3:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Veri hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        # Eksenleri senin tarladaki duruşuna göre ayarlıyoruz
        self.fig, self.ax2d = plt.subplots(figsize=(8, 10)) # Boyuna bir grafik
        
        # Grid oluşturma: Satir=X (Yan), Sutun=Y (İleri)
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        # Veriyi ızgaraya oturt
        zi = griddata((self.df['satir'], self.df['sutun']), self.df['clean_diff'] * gain, (self.grid_x, self.grid_y), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        if med > 0:
            m_size = int(med); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # Mod işlemleri
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        self.zi_cache = zi
        
        # Renk ölçeği ayarı
        z_min, z_max = zi.min(), zi.max()
        if mode == 'Raw':
            abs_max = max(abs(z_min), abs(z_max), 0.1)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        else:
            norm = Normalize(vmin=0, vmax=max(z_max, 0.1))
        
        # İŞTE BURASI DÜZELTİLDİ: origin='lower' ve extent sırası
        im = self.ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                              origin='lower', cmap='turbo', norm=norm, aspect='equal') # aspect='equal' tarlayı kare tutar
        
        plt.colorbar(im, ax=self.ax2d, label="Manyetik Siddet", fraction=0.046, pad=0.04)
        
        self.ax2d.set_xlabel("YAN MESAFE (Satir - Metre)")
        self.ax2d.set_ylabel("ILERI MESAFE (Sutun - Metre)")
        self.ax2d.set_title(f"C3 Saha Analizi - {mode}")

        # Basit hedef tespiti
        threshold = (self.std_noise if self.std_noise > 0 else 1) * 3.5 
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        txt = f"Saha: {xi.max()-xi.min():.1f}x{yi.max()-yi.min():.1f}m\n"
        txt += "-"*30 + "\n"
        for i in range(1, min(num + 1, 6)):
            mask = labeled == i
            coords = np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            amp = zi[mask].mean()
            d = self.calculate_depth(amp/gain)
            txt += f"Hedef {i}: X:{xi[x_idx]:.1f}, Y:{yi[y_idx]:.1f} -> {d:.2f}m\n"
        self.info_text_str = txt

def main():
    st.set_page_config(page_title="C3 Pro Web", layout="wide")
    st.title("🛰️ C3 Gradiometre Analiz İstasyonu")
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        gain = st.slider("Kazanc", 0.1, 1000.0, 100.0)
        filt = st.slider("Filtre", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))
    
    uploaded_file = st.file_uploader("CSV Verisi Yukle", type=['csv'])
    if uploaded_file:
        analiz = C3AnalizSistemi(uploaded_file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        c1, c2 = st.columns([2, 1])
        with c1: st.pyplot(analiz.fig)
        with c2: 
            st.subheader("🎯 Analiz Sonuçları")
            st.code(analiz.info_text_str)
        
        st.divider()
        st.subheader("🌋 3D Gorunum")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, x=analiz.grid_x[0,:], y=analiz.grid_y[:,0], colorscale='Turbo')])
        fig3d.update_layout(scene=dict(xaxis_title='Yan', yaxis_title='Ileri', zaxis_title='Siddet', aspectratio=dict(x=1, y=1, z=0.4)), height=700)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__": main()
