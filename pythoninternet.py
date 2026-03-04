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

# --- FİZİKSEL SABİTLER ---
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
MAX_DEPTH = 10.0
GRID_RES = 180

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="C3 ANALİZ PRO", layout="wide")
st.markdown("<h2 style='text-align: center;'>🛡️ C3 Manyetik Gradiometre Gerçek Zamanlı Analiz</h2>", unsafe_allow_html=True)

# --- SIDEBAR KONTROLLERİ ---
st.sidebar.header("🕹️ Kontrol Paneli")
gain = st.sidebar.slider("Kazanç (Gain)", 1.0, 1000.0, 100.0)
esik = st.sidebar.slider("Eşik (Filter)", 0, 500, 20)
blur = st.sidebar.slider("Yumuşatma", 0.0, 4.0, 0.8)
med_size = st.sidebar.slider("Nokta Filtre", 0, 9, 3, step=2)
mode = st.sidebar.selectbox("Analiz Modu", ("Raw", "Analytic", "Gradient"))

# --- YARDIMCI FONKSİYONLAR ---
def calculate_depth(peak_nt, shape_factor=2.8):
    grad = abs(peak_nt) / SENSOR_MESAFESI
    if grad < 0.005: return 0.0
    depth = YUKSEKLIK_SABITI + (shape_factor * abs(peak_nt) / (grad + 0.05))
    return min(depth, MAX_DEPTH)

def detect_targets(zi, std_noise, xi, yi):
    threshold = std_noise * 3.0
    binary = np.abs(zi) > (threshold * (gain / 50))
    labeled, num_features = label(binary)
    targets = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        if np.sum(mask) < 3: continue
        coords = np.argwhere(mask)
        y_idx, x_idx = coords.mean(axis=0).astype(int)
        real_x = xi[min(x_idx, len(xi)-1)]
        real_y = yi[min(y_idx, len(yi)-1)]
        val = zi[mask].mean()
        targets.append({'id': i, 'amp': val, 'x': real_x, 'y': real_y})
    return sorted(targets, key=lambda x: abs(x['amp']), reverse=True)[:5]

# --- VERİ İŞLEME VE GÖRSELLEŞTİRME ---
uploaded_file = st.file_uploader("CSV Dosyasını Buraya Bırakın", type=["csv"])

if uploaded_file:
    try:
        # Veri Temizleme
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower().str.replace('ı', 'i').str.replace('ş', 's')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        
        df['raw_diff'] = df['s1_z'] - df['s2_z']
        df['clean_diff'] = df['raw_diff'] - df['raw_diff'].median()
        
        for r in df['satir'].unique():
            mask = df['satir'] == r
            if sum(mask) > 5:
                df.loc[mask, 'clean_diff'] = detrend(df.loc[mask, 'clean_diff'], type='linear')
        
        std_noise = df['clean_diff'].std()

        # Gridleme ve Analiz
        xi = np.linspace(df['satir'].min(), df['satir'].max(), GRID_RES)
        yi = np.linspace(df['sutun'].min(), df['sutun'].max(), GRID_RES)
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        zi = griddata((df['satir'], df['sutun']), df['clean_diff'] * gain, (grid_x, grid_y), method='linear', fill_value=0)
        
        if med_size > 0:
            m_size = int(med_size)
            if m_size % 2 == 0: m_size += 1
            zi = median_filter(zi, size=m_size)
        
        zi = np.where(np.abs(zi) < esik, 0, zi)
        if blur > 0: zi = gaussian_filter(zi, sigma=blur)
        
        if mode == 'Analytic':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2 + zi**2)
        elif mode == 'Gradient':
            dx, dy = sobel(zi, axis=1), sobel(zi, axis=0)
            zi = np.sqrt(dx**2 + dy**2)

        targets = detect_targets(zi, std_noise, xi, yi)

        # --- EKRAN DÜZENİ (TEK SAYFA) ---
        col_map, col_list = st.columns([2, 1])

        with col_map:
            st.subheader(f"📍 2D Isı Haritası ({mode})")
            fig2d, ax2d = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            
            z_min, z_max = zi.min(), zi.max()
            if mode == 'Raw' and z_min < 0 < z_max:
                norm = TwoSlopeNorm(vmin=z_min, vcenter=0, vmax=z_max)
            else:
                norm = Normalize(vmin=z_min, vmax=z_max)
            
            im = ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], 
                            origin='lower', cmap='turbo', norm=norm, aspect='auto')
            plt.colorbar(im, ax=ax2d)
            
            for t in targets:
                ax2d.plot(t['x'], t['y'], 'kx', markersize=12, markeredgewidth=2)
                ax2d.text(t['x'], t['y'], f" H{t['id']}", color='white', fontweight='bold')
            
            st.pyplot(fig2d)

        with col_list:
            st.subheader("🎯 Hedef Analizi")
            target_rows = []
            for t in targets:
                d = calculate_depth(t['amp']/gain)
                target_rows.append({"ID": f"H{t['id']}", "Konum": f"{t['x']:.1f}, {t['y']:.1f}", "Derinlik": f"{d:.2f}m"})
            st.table(pd.DataFrame(target_rows))
            
            # Küçük bir histogram
            fig_h, ax_h = plt.subplots(figsize=(5, 3))
            ax_h.hist(zi.flatten(), bins=20, color='cyan', alpha=0.5)
            ax_h.set_title("Sinyal Dağılımı", fontsize=8)
            st.pyplot(fig_h)

        st.divider()

        # --- ALT KISIM: DEV 3D ---
        st.subheader("🧊 3D İnteraktif Topografya")
        st.info("💡 Mouse ile sağa-sola döndürebilir, tekerlek ile yakınlaşabilirsin.")
        # 
        fig3d = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Turbo')])
        fig3d.update_layout(
            scene=dict(
                xaxis_title='Satır (Yan)',
                yaxis_title='Sütun (İleri)',
                zaxis_title='Şiddet',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=800
        )
        st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Hata oluştu: {e}")
