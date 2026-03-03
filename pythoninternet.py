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
import datetime

# --- FİZİKSEL SABİTLER (C3 Orijinal Değerlerin) ---
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
            st.error(f"Veri yükleme hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

    def process_and_plot(self, gain, filt, blur, med, mode):
        plt.style.use('dark_background')
        # Bilgisayardaki 16:9 benzeri geniş yerleşim
        self.fig = plt.figure(figsize=(16, 10))
        
        # GridSpec ile bilgisayardaki o çoklu panel yapısı
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        self.ax2d = self.fig.add_subplot(gs[0, 0])
        self.ax_hist = self.fig.add_subplot(gs[1, 0])
        self.ax_info = self.fig.add_subplot(gs[:, 1])
        self.ax_info.axis('off')

        xi = np.linspace(self.df['satir'].min(), self.df['satir'].max(), GRID_RES)
        yi = np.linspace(self.df['sutun'].min(), self.df['sutun'].max(), GRID_RES)
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((self.df['satir'], self.df['sutun']), self.df['clean_diff'] * gain, (self.grid_x, self.grid_y), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # --- YÖN DÜZELTME (Sivri Uç Aşağı Bakmalı) ---
        # Bilgisayar kodunla birebir aynı olması için burada transpoz veya yön ayarı
        # zi = zi.T # Eğer gerekirse bunu açabiliriz ama imshow yönü genelde çözer
        
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
        
        # HARİTA ÇİZİMİ
        im = self.ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap='turbo', norm=norm, aspect='auto')
        self.fig.colorbar(im, ax=self.ax2d, label="Siddet")
        self.ax2d.set_title(f"C3 SAHA ANALİZİ - {mode}")
        self.ax2d.set_xlabel("YAN (Satir)"); self.ax2d.set_ylabel("İLERİ (Sutun)")

        # HISTOGRAM
        self.ax_hist.hist(zi.flatten(), bins=50, color='#00ff9d', alpha=0.7)
        self.ax_hist.set_yscale('log')
        self.ax_hist.set_title("Manyetik Dağılım (Histogram)")

        # HEDEF ANALİZİ VE METİN
        threshold = (self.std_noise if self.std_noise > 0 else 1) * 3.5 
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num_features = label(binary)
        
        info_txt = f"SABIT AYAK: {YUKSEKLIK_SABITI*100}cm\nSENSOR FARK: {SENSOR_MESAFESI*100}cm\n\n"
        info_txt += "TESPİT EDİLEN HEDEFLER:\n" + "-"*25 + "\n"
        
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
            info_txt += f"H#{i+1}: X:{t['x']:.1f}m, Y:{t['y']:.1f}m\nDerinlik: {d:.2f}m\n\n"
        
        self.ax_info.text(0.05, 0.95, info_txt, fontsize=12, family='monospace', color='#00ff9d', verticalalignment='top')
        self.info_text_str = info_txt

def main():
    st.set_page_config(page_title="C3 Pro Full Analiz", layout="wide")
    st.title("🛰️ C3 Gradiometre - Tam Donanımlı Web Analiz")

    with st.sidebar:
        st.header("⚙️ Kontroller")
        gain = st.slider("Gain (Hassasiyet)", 0.1, 1000.0, 100.0)
        filt = st.slider("Eşik (Gürültü)", 0, 1000, 30)
        blur = st.slider("Yumuşatma (Blur)", 0.0, 5.0, 1.0)
        med = st.slider("Median (Nokta Filtre)", 0, 9, 3)
        mode = st.radio("Analiz Modu", ('Raw', 'Analytic', 'Gradient'))
        st.divider()
        st.write("Tüm fonksiyonlar aktif.")

    uploaded_file = st.file_uploader("Telefondaki CSV Dosyasını Seç", type=['csv'])

    if uploaded_file:
        analiz = C3AnalizSistemi(uploaded_file)
        analiz.process_and_plot(gain, filt, blur, med, mode)
        
        # 2D Panel
        st.pyplot(analiz.fig)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 PDF RAPORU OLUŞTUR"):
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet
                buf = io.BytesIO()
                analiz.fig.savefig("temp.png")
                doc = SimpleDocTemplate(buf, pagesize=A4)
                parts = [Paragraph("C3 Profesyonel Saha Raporu", getSampleStyleSheet()['Title']), Spacer(1,12), Image("temp.png", width=450, height=350), Spacer(1,12), Paragraph(analiz.info_text_str.replace('\n', '<br/>'), getSampleStyleSheet()['Normal'])]
                doc.build(parts)
                st.download_button("Raporu Bilgisayara İndir", buf.getvalue(), "C3_Rapor.pdf")
        
        # 3D Panel
        st.divider()
        st.subheader("🌋 İnteraktif 3D Manyetik Alan")
        fig3d = go.Figure(data=[go.Surface(z=analiz.zi_cache, x=analiz.grid_x[0,:], y=analiz.grid_y[:,0], colorscale='Turbo')])
        fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.5)), height=800)
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
