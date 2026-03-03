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
import datetime

# --- SENİN ORİJİNAL SABİTLERİN ---
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
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # ORİJİNAL HESAPLAMA
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            # SENİN 5 ADIMLI DETREND KURALIN
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

    def process(self, gain, filt, blur, med, mode):
        # Eksenleri senin kodundaki gibi hazırlıyoruz
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        # Gridleme
        zi = griddata((self.df['satir'], self.df['sutun']), 
                      self.df['clean_diff'] * gain, 
                      (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # REİSİN İSTEDİĞİ: SOLA BAKAN YÖN (90 Derece Sola Döndürülmüş)
        # Bu işlem senin bilgisayarındaki o "mavilik/sivrilik" yerini korur ama yönü çevirir
        zi = np.rot90(zi, k=1)
        ext = [yi.min(), yi.max(), xi.min(), xi.max()]

        # Orijinal Filtreler
        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # ANALYTIC MODU (Hata vermemesi için düzeltildi)
        if mode == 'Analytic':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        return zi, ext, xi, yi

def main():
    st.set_page_config(page_title="C3 Pro Web", layout="wide")
    st.title("🛰️ C3 Gradiometre - Orijinal Bilgisayar Analizi")

    with st.sidebar:
        st.header("⚙️ Kontroller")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))

    file = st.file_uploader("CSV Dosyasını Yükle", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        zi, ext, xi, yi = analiz.process(gain, filt, blur, med, mode)
        
        # --- 2D DEV EKRAN ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 10))
        lim = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-lim if mode=='Raw' else 0, vcenter=0 if mode=='Raw' else None, vmax=lim)
        
        im = ax.imshow(zi, extent=ext, origin='lower', cmap='turbo', norm=norm, aspect='auto')
        plt.colorbar(im, ax=ax, label="Şiddet")
        ax.set_xlabel("İLERİ (Sutun)"); ax.set_ylabel("YAN (Satir)")
        st.pyplot(fig, use_container_width=True)

        # --- HEDEF ANALİZİ VE RAPOR ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Tespit Edilen Hedefler")
            # Orijinal hedef tespit mantığı
            threshold = (analiz.std_noise) * 3.5
            binary = np.abs(zi) > (threshold * (gain / 100))
            labeled, num = label(binary)
            
            target_list = []
            for i in range(1, num + 1):
                mask = labeled == i
                if np.sum(mask) < 5: continue
                val = zi[mask].mean()
                d = analiz.calculate_depth(val/gain)
                target_list.append(f"H#{i}: Derinlik {d:.2f}m")
            
            if target_list:
                for t in target_list[:5]: st.write(t)
            else:
                st.write("Belirgin hedef bulunamadı.")

        with col2:
            st.subheader("🌋 3D Görünüm")
            fig3d = go.Figure(data=[go.Surface(z=zi, colorscale='Turbo')])
            fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
