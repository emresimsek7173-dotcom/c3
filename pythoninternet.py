import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend

# --- SENİN ORİJİNAL SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 200          

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
        # SENİN BİLGİSAYARDAKİ FORMÜLÜN
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def main():
    st.set_page_config(page_title="C3 Pro Full", layout="wide")
    st.title("🛰️ C3 Gradiometre - 2D/3D Adım Analizi")

    with st.sidebar:
        st.header("⚙️ Ayarlar")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))

    file = st.file_uploader("CSV Dosyasını Seç reisim", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        
        xi = np.linspace(analiz.df['satir'].min(), analiz.df['satir'].max(), GRID_RES)
        yi = np.linspace(analiz.df['sutun'].min(), analiz.df['sutun'].max(), GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, 
                      (gx, gy), method='cubic', fill_value=0)
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

        # --- 2D HARİTA (ADIM GRIDLİ) ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 10))
        
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if mode == 'Raw' else Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        # --- KAÇ ADIM VARSA O KADAR KARE ---
        u_satir = sorted(analiz.df['satir'].unique())
        u_sutun = sorted(analiz.df['sutun'].unique())
        ax.set_xticks(u_satir)
        ax.set_yticks(u_sutun)
        ax.grid(True, color='white', linestyle='--', alpha=0.5, linewidth=1)
            
        plt.colorbar(im, ax=ax, label="Şiddet")
        ax.set_xlabel("YAN ADIM (Satir)"); ax.set_ylabel("İLERİ ADIM (Sutun)")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEF LİSTESİ ---
        st.divider()
        st.subheader("🎯 Adım ve Derinlik Raporu")
        threshold = (analiz.std_noise) * 3.5
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 5: continue
            val = zi[mask].mean()
            coords = np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'amp': val, 'x': xi[x_idx], 'y': yi[y_idx], 'd': d})
        
        for i, t in enumerate(sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]):
            st.info(f"🟢 **Hedef {i+1}:** Yan: **{t['x']:.1f}**. Adım | İleri: **{t['y']:.1f}**. Adım | Derinlik: **{t['d']:.2f} m**")

        # --- 3D HARİTA (GERİ GELDİ) ---
        st.divider()
        st.subheader("🌋 3D İnteraktif Görünüm")
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.5)))
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
