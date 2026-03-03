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
            
            # ORİJİNAL HESAPLAMA (Dokunulmaz)
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            
            # SENİN 5 ADIMLI DETREND KURALIN (Dokunulmaz)
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

    def process_and_plot(self, gain, filt, blur, med, mode, show_grid):
        # Eksenleri senin kodundaki gibi hazırlıyoruz
        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        # Gridleme
        zi = griddata((self.df['satir'], self.df['sutun']), 
                      self.df['clean_diff'] * gain, 
                      (self.grid_x, self.grid_y), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # Orijinal Filtreler
        if med > 0:
            m_size = int(med)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < filt, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        # Analytic Modu (Hata vermemesi için düzeltildi)
        if mode == 'Analytic':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy) + np.square(zi))
        elif mode == 'Gradient':
            dx = sobel(zi, axis=1)
            dy = sobel(zi, axis=0)
            zi = np.sqrt(np.square(dx) + np.square(dy))

        self.zi_cache = zi

        # --- GÜZELLEŞTİRİLMİŞ ÇOKLU PANEL DÜZENİ (GS) ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1.5, 1, 1])
        
        ax2d = fig.add_subplot(gs[0:2, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.axis('off')

        # 2D ANA HARİTA
        lim = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-lim if mode=='Raw' else 0, vcenter=0 if mode=='Raw' else None, vmax=lim)
        im = ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='turbo', norm=norm, aspect='auto')
        fig.colorbar(im, ax=ax2d, label="Şiddet")
        ax2d.set_title(f"C3 ANALİZ - {mode}")
        ax2d.set_xlabel("YAN ADIM (Satir)"); ax2d.set_ylabel("İLERİ ADIM (Sutun)")

        # --- ADIM IZGARASI (GRID) BUTONU ---
        if show_grid:
            ax2d.grid(True, color='white', linestyle='--', alpha=0.5)

        # PROFİL KESİTİ (Otomatik tam orta noktadan)
        mid_idx = zi.shape[0] // 2
        ax_prof.plot(xi, zi[mid_idx, :], color='cyan')
        ax_prof.set_title(f"Orta Hat Kesit Analizi")
        ax_prof.set_xlabel("Yan Adım"); ax_prof.set_ylabel("Şiddet")
        ax_prof.grid(alpha=0.2)

        # HISTOGRAM (SIKLIK)
        ax_hist.hist(zi.flatten(), bins=40, color='#00ff9d', alpha=0.6)
        ax_hist.set_yscale('log')
        ax_hist.set_title("Veri Dağılım Sıklığı")

        # HEDEF ANALİZİ VE METİN (BİLGİ)
        threshold = (self.std_noise) * 3.5
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        
        info_txt = f"Saha: {xi.max():.1f}x{yi.max():.1f} Adım | Toplam Nokta: {len(self.df)}\n"
        info_txt += f"STD Gürültü: {self.std_noise:.4f}\n\n"
        info_txt += "TESPİT EDİLEN HEDEFLER:\n" + "-"*35 + "\n"
        
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 5: continue
            val = zi[mask].mean()
            coords = np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = self.calculate_depth(val/gain)
            targets.append({'amp': val, 'x': xi[min(x_idx, len(xi)-1)], 'y': yi[min(y_idx, len(yi)-1)], 'd': d})
        
        for i, t in enumerate(sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]):
            info_txt += f"H#{i+1}: Yan Adım:{t['x']:.1f}, İleri Adım:{t['y']:.1f} -> Derinlik: {t['d']:.2f}m\n"
        
        ax_info.text(0.05, 0.95, info_txt, fontsize=12, color='#00ff9d', family='monospace', verticalalignment='top')
        return fig

def main():
    st.set_page_config(page_title="C3 Pro Adım", layout="wide")
    st.title("🛰️ C3 Gradiometre - Adım Adım Analiz Paneli")

    with st.sidebar:
        st.header("⚙️ Kontroller")
        gain = st.slider("Gain", 0.1, 1000.0, 100.0)
        filt = st.slider("Esik", 0, 1000, 30)
        blur = st.slider("Blur", 0.0, 5.0, 1.0)
        med = st.slider("Median", 0, 9, 3)
        mode = st.radio("Mod", ('Raw', 'Analytic', 'Gradient'))
        st.divider()
        # --- IZGARA BUTONU ---
        show_grid = st.checkbox("Adım Izgarasını Göster (+)")
        st.info("Kırmızı alanlar tarlada 'sivri uç' demektir, dikkat reisim.")

    file = st.file_uploader("Telefondaki CSV Dosyasını Seç reisim", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        fig = analiz.process_and_plot(gain, filt, blur, med, mode, show_grid)
        
        # 2D Panel
        st.pyplot(fig, use_container_width=True)
        
        # 3D İnteraktif
        st.divider()
        st.subheader("🌋 3D İnteraktif Adım Topografyası")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, colorscale='Turbo')])
        fig3d.update_layout(height=700)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
