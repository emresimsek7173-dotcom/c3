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
        self.zi_cache = None
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

def main():
    st.set_page_config(page_title="C3 Dev Ekran v2", layout="wide")
    st.title("🛰️ C3 Gradiometre - Saf 2D Analiz")

    with st.sidebar:
        st.header("⚙️ Kontroller")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))
        show_grid = st.checkbox("Adım Izgarasını Göster (+)")

    file = st.file_uploader("CSV Dosyasını Seç reisim", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        
        # Gridleme
        xi = np.linspace(analiz.df['satir'].min(), analiz.df['satir'].max(), GRID_RES)
        yi = np.linspace(analiz.df['sutun'].min(), analiz.df['sutun'].max(), GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, 
                      (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # --- ANALİTİK MOD HATASI ÇÖZÜLDÜ ---
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        # --- DEV 2D ÇİZİM ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Renk Ölçeklendirme Hatası Fix
        v_max = max(np.abs(zi).max(), 1.0)
        if mode == 'Raw':
            # Raw modda eksi ve artı değerler olduğu için vcenter=0 zorunlu
            norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max)
        else:
            # Analytic ve Gradient modda veriler hep artıdır, normal Normalize kullanmalıyız
            norm = Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto')
        
        # --- ADIM SAYISINA GÖRE AKILLI GRID ---
        if show_grid:
            # Kaç farklı satir ve sutun varsa o kadar çizgi çeker (Mesela 9 satir varsa 9 çizgi)
            unique_satir = np.sort(analiz.df['satir'].unique())
            unique_sutun = np.sort(analiz.df['sutun'].unique())
            ax.set_xticks(unique_satir)
            ax.set_yticks(unique_sutun)
            ax.grid(True, color='white', linestyle='--', alpha=0.3)
            
        plt.colorbar(im, ax=ax, label="Şiddet")
        ax.set_xlabel("YAN ADIM (Satir)"); ax.set_ylabel("İLERİ ADIM (Sutun)")
        st.pyplot(fig, use_container_width=True)

        # Hedefler ve 3D altta kalsın
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🎯 Hedef Tahminleri")
            threshold = (analiz.std_noise) * 3.5
            binary = np.abs(zi) > (threshold * (gain / 100))
            labeled, num = label(binary)
            for i in range(1, num + 1):
                mask = labeled == i
                if np.sum(mask) < 5: continue
                val = zi[mask].mean()
                d = analiz.calculate_depth(val/gain)
                st.write(f"Hedef #{i}: Derinlik {d:.2f}m")
        with col2:
            st.subheader("🌋 3D Görünüm")
            fig3d = go.Figure(data=[go.Surface(z=zi, colorscale='Turbo')])
            st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
