import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend

# --- SENİN SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 300 # Çözünürlüğü biraz artırdım daha net olsun

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
            st.error(f"Hata: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def main():
    st.set_page_config(page_title="C3 Kesin Grid", layout="wide")
    st.title("🛰️ C3 Gradiometre - Adım Sayısı Kadar Kare")

    with st.sidebar:
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))

    file = st.file_uploader("CSV Dosyasını Yükle", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        
        # Grid Sınırları (Hücreleri tam oluşturmak için yarım adım genişletiyoruz)
        x_min, x_max = analiz.df['satir'].min(), analiz.df['satir'].max()
        y_min, y_max = analiz.df['sutun'].min(), analiz.df['sutun'].max()
        
        xi = np.linspace(x_min, x_max, GRID_RES)
        yi = np.linspace(y_min, y_max, GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        if med > 0:
            m_size = int(med); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- 2D HARİTA ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if mode == 'Raw' else Normalize(vmin=0, vmax=v_max)
        
        # imshow'u tam sınırlarına oturtuyoruz
        ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                  origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        # --- KESİN KARE SİSTEMİ (SINIRLARA ÇİZGİ) ---
        u_satir = sorted(analiz.df['satir'].unique())
        u_sutun = sorted(analiz.df['sutun'].unique())
        
        # Çizgileri adımların tam arasına (sınırına) çekiyoruz
        for x in u_satir:
            ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.axvline(x + 0.5, color='white', linestyle='-', alpha=0.8, linewidth=1.5)
        for y in u_sutun:
            ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.8, linewidth=1.5)
            ax.axhline(y + 0.5, color='white', linestyle='-', alpha=0.8, linewidth=1.5)
            
        ax.set_xticks(u_satir)
        ax.set_yticks(u_sutun)
        ax.set_xlabel("Yan Adım"); ax.set_ylabel("İleri Adım")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEF LİSTESİ ---
        st.divider()
        st.subheader("🎯 Hedef Koordinatları (Adım Bazlı)")
        threshold = (analiz.std_noise) * 3.5
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            val, coords = zi[mask].mean(), np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'amp': val, 'x': xi[x_idx], 'y': yi[y_idx], 'd': d})
        
        for i, t in enumerate(sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]):
            st.write(f"🟢 **Hedef {i+1}:** Yan: {t['x']:.1f}. Adım | İleri: {t['y']:.1f}. Adım | Derinlik: **{t['d']:.2f} m**")

        # --- 3D GÖRÜNÜM ---
        st.divider()
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
