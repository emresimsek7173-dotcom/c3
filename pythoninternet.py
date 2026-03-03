import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label
from scipy.signal import detrend
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import datetime

# --- SENİN DEĞİŞMEZ SABİTLERİN ---
SENSOR_MESAFESI = 0.80  
YUKSEKLIK_SABITI = 0.20 
MAX_DEPTH = 10.0        
GRID_RES = 500 # En yüksek netlik

class C3AnalizSistemi:
    def __init__(self, file_buffer):
        self.df = None
        self.load_and_clean_data(file_buffer)
        
    def load_and_clean_data(self, file_buffer):
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
            df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            
            # Veri temizleme ve detrend (Senin orijinal mantığın)
            df['raw_diff'] = df['s1_z'] - df['s2_z']
            df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
            for r in df['satir'].unique():
                mask = df['satir'] == r
                if sum(mask) > 5:
                    df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
            self.df = df
            self.std_noise = df['clean_diff'].std()
        except Exception as e:
            st.error(f"Veri Hatası: {e}")

    def calculate_depth(self, peak_nt, shape_factor=3.0):
        grad = abs(peak_nt) / SENSOR_MESAFESI
        if grad < 0.01: return 0.0
        depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.1))
        return min(depth, MAX_DEPTH)

def create_pro_pdf(targets, mode, fig_plot):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Başlık ve Tarih
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 40, "C3 GRADIOMETRE PROFESYONEL ANALIZ")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height - 55, f"Rapor Tarihi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # HARİTA RESMİNİ PDF'E EKLE
    imgdata = io.BytesIO()
    fig_plot.savefig(imgdata, format='png', bbox_inches='tight', dpi=150)
    imgdata.seek(0)
    c.drawImage(ImageReader(imgdata), 50, height - 420, width=500, height=350)

    # HEDEF TABLOSU
    y = height - 450
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Hedef # | Yan (Satır) | İleri (Sütun) | Derinlik | Şiddet | Tür")
    c.line(50, y-5, 550, y-5)

    c.setFont("Helvetica", 10)
    for i, t in enumerate(targets):
        y -= 25
        tur = "BOSLUK / MEZAR" if t['amp'] < 0 else "METAL / YAPI"
        c.drawString(50, y, f"#{i+1}  |  {t['x']:.1f}  |  {t['y']:.1f}  |  {t['d']:.2f}m  |  {t['amp']:.1f} nT  |  {tur}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(page_title="C3 Ultimate Pro", layout="wide")
    st.title("🛰️ C3 Gradiometre - Profesyonel Yer Altı Görüntüleme")

    with st.sidebar:
        st.header("⚙️ Ana Kontroller")
        gain = st.slider("Manyetik Kazanç (Gain)", 1, 1000, 300)
        filt = st.slider("Hassasiyet Eşiği", 0, 500, 20)
        
        st.divider()
        st.subheader("🖼️ Görüntü Netleştirme")
        blur_val = st.slider("Blur (Yumuşatma)", 0.0, 10.0, 1.5)
        med_val = st.slider("Median (Parazit Silme)", 0, 9, 3)
        
        st.divider()
        mode = st.radio("Analiz Modu", ('Raw (Pozitif & Negatif)', 'Analytic (Dolu Tepecik Hedef)'))

    file = st.file_uploader("CSV Verinizi Yükleyin", type=['csv'])
    
    if file:
        analiz = C3AnalizSistemi(file)
        x_min, x_max = analiz.df['satir'].min(), analiz.df['satir'].max()
        y_min, y_max = analiz.df['sutun'].min(), analiz.df['sutun'].max()
        
        xi = np.linspace(x_min, x_max, GRID_RES)
        yi = np.linspace(y_min, y_max, GRID_RES)
        gx, gy = np.meshgrid(xi, yi)
        
        # Yüzey oluşturma
        zi = griddata((analiz.df['satir'], analiz.df['sutun']), 
                      analiz.df['clean_diff'] * gain, (gx, gy), method='cubic', fill_value=0)
        zi = np.nan_to_num(zi)

        # 1. PARAZİT SİLME (MEDIAN)
        if med_val > 0:
            m_size = int(med_val); m_size = m_size+1 if m_size%2==0 else m_size
            zi = median_filter(zi, size=m_size)

        # 2. DOLU TEPECİK ANALİTİĞİ (Sadece Kenar Değil, Tam Kütle)
        if mode == 'Analytic (Dolu Tepecik Hedef)':
            dx = np.gradient(zi, axis=1)
            dy = np.gradient(zi, axis=0)
            # Analitik Sinyal Genliği: Kütleyi dolu gösteren formül
            zi = np.sqrt(np.square(zi) + np.square(dx) + np.square(dy))
        
        # 3. BLUR (Yumuşatma)
        if blur_val > 0:
            zi = gaussian_filter(zi, sigma=blur_val)
            
        # 4. EŞİK FİLTRESİ
        zi = np.where(np.abs(zi) < filt, 0, zi)

        # --- GÖRSELLEŞTİRME ---
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        v_max = max(np.abs(zi).max(), 1.0)
        norm = TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max) if 'Raw' in mode else Normalize(vmin=0, vmax=v_max)
        
        im = ax.imshow(zi, extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], 
                       origin='lower', cmap='turbo', norm=norm, aspect='auto', interpolation='bilinear')
        
        # OTOMATİK KARE GRİD SİSTEMİ (Çizgiler adımların tam ortasından geçer)
        u_satir, u_sutun = sorted(analiz.df['satir'].unique()), sorted(analiz.df['sutun'].unique())
        for x in u_satir: ax.axvline(x - 0.5, color='white', linestyle='-', alpha=0.2)
        for y in u_sutun: ax.axhline(y - 0.5, color='white', linestyle='-', alpha=0.2)
        ax.set_xticks(u_satir); ax.set_yticks(u_sutun)
        
        plt.colorbar(im, label="Manyetik Şiddet (nT)")
        st.pyplot(fig, use_container_width=True)

        # HEDEF TESPİTİ VE DERİNLİK
        threshold = (analiz.std_noise) * 4
        binary = np.abs(zi) > (threshold * (gain / 100))
        labeled, num = label(binary)
        targets = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) < 5: continue
            val, coords = zi[mask].mean(), np.argwhere(mask)
            y_idx, x_idx = coords.mean(axis=0).astype(int)
            d = analiz.calculate_depth(val/gain)
            targets.append({'x': xi[x_idx], 'y': yi[y_idx], 'd': d, 'amp': val})
        
        targets = sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

        # RAPORLAMA
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🎯 Tespit Edilen Hedefler")
            for i, t in enumerate(targets):
                tip = "🔵 BOŞLUK" if t['amp'] < 0 else "🔴 METAL"
                st.info(f"**Hedef #{i+1}** | Adım: ({t['x']:.1f}, {t['y']:.1f}) | Derinlik: **{t['d']:.2f}m** | {tip}")
        
        with col2:
            st.subheader("📄 PDF Raporu")
            if st.button("Haritalı PDF Oluştur"):
                pdf_bytes = create_pro_pdf(targets, mode, fig)
                st.download_button(label="📥 PDF'i İndir", data=pdf_bytes, 
                                   file_name=f"C3_Analiz_Raporu.pdf", mime="application/pdf")

        # 3D GÖRÜNÜM
        st.divider()
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
