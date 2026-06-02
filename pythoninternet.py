import streamlit as st
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from scipy.optimize import curve_fit
import json
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  SABİTLER
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_MESAFESI  = 0.80
YUKSEKLIK_SABITI = 0.05
MAX_DEPTH        = 5.0
DEFAULT_GRID_RES = 100
ADIM_MESAFESI    = 0.50

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA', '#0066FF', '#00CCFF', '#00CC44',
    '#FFFF00', '#FF6600', '#CC0000'], N=512)


# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSIYONLAR
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_col(s: str) -> str:
    """Sütun adını ASCII'ye normalize et"""
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s.strip().lower().replace(' ', '_')


def obje_tahmini(tip, r2_m, r2_b, fwhm, derinlik, val, snr, mod):
    """Tüm kanıtları birleştirerek olası obje tipini tahmin eder"""
    tahminler = []

    d  = derinlik or 0.0
    fw = fwhm     or 0.5
    am = abs(val)

    if mod in ('Gradient', 'Analitik'):
        tahminler.append(("Kenar/Sinir anomalisi — detay modda incele", 0.50, '#FFA500'))
        if not tahminler:
            return [("Tanımlanamadı", 0.0, '#555555')]
        tahminler.sort(key=lambda x: x[1], reverse=True)
        return tahminler[:3]

    # Metal grubu
    if tip == 'metal' and r2_m >= 0.35:
        if fw < 0.25 and d < 0.5:
            tahminler.append(("Küçük metal obje (sikke/parça)", r2_m * 0.90 + (1 - d) * 0.10, '#FF6600'))
        elif fw < 0.7 and 0.2 < d < 1.2:
            tahminler.append(("Metal küp / sandık / kap", r2_m * 0.85, '#FF4500'))
        elif fw >= 0.7 and d < 1.5:
            tahminler.append(("Metal boru / ray / levha", r2_m * 0.80, '#FF2200'))
        elif d >= 1.5:
            tahminler.append(("Derin metal yapı", r2_m * 0.70, '#FF0000'))
        tahminler.append(("Metal obje", r2_m * 0.60, '#FF8C00'))

    # Boşluk / seramik grubu
    if tip == 'bosluk' and r2_b >= 0.35:
        if fw < 0.5 and d < 0.8:
            tahminler.append(("Çömlek / pişmiş toprak kap", r2_b * 0.85, '#00CFFF'))
            tahminler.append(("Küçük boşluk / hava cebi", r2_b * 0.70, '#00AACC'))
        elif fw < 1.0 and 0.3 < d < 1.5:
            tahminler.append(("Çömlek küp / seramik obje", r2_b * 0.80, '#00CFFF'))
            tahminler.append(("Küçük yapı boşluğu", r2_b * 0.65, '#0088AA'))
        elif fw >= 1.0 or d >= 1.5:
            tahminler.append(("Tünel / oda / mezar odası", r2_b * 0.75, '#0066FF'))
            tahminler.append(("Büyük boşluk yapısı", r2_b * 0.60, '#0044CC'))
        tahminler.append(("Boşluk / içi dolu olmayan yapı", r2_b * 0.50, '#88AAFF'))

    if not tahminler:
        return [("Tanımlanamadı", 0.0, '#555555')]

    tahminler.sort(key=lambda x: x[1], reverse=True)
    return tahminler[:3]


# ─────────────────────────────────────────────────────────────────────────────
#  ANA SINIF
# ─────────────────────────────────────────────────────────────────────────────
class C3Analiz:
    def __init__(self, df, adim_m=None):
        self.df = df
        self.adim_m = adim_m if adim_m else ADIM_MESAFESI
        self.grid_res = DEFAULT_GRID_RES
        self.zi = None
        self.zi_raw = None
        self.grid_x = None
        self.grid_y = None
        self.xi_arr = None
        self.yi_arr = None
        self.targets = []
        self._veri_hazirla()

    def _veri_hazirla(self):
        """Veri hazırlama"""
        df = self.df.copy()
        df.columns = [_normalize_col(c) for c in df.columns]

        df['tfa1'] = np.sqrt(df['s1_x']**2 + df['s1_y']**2 + df['s1_z']**2)
        df['tfa2'] = np.sqrt(df['s2_x']**2 + df['s2_y']**2 + df['s2_z']**2)

        for col in ['tfa1', 'tfa2', 's1_z', 's2_z']:
            lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
            df[col] = df[col].clip(lo, hi)

        df['tfa_diff'] = df['tfa1'] - df['tfa2']
        df['z_diff'] = df['s1_z'] - df['s2_z']

        for col in ['tfa_diff', 'z_diff']:
            df[col] -= df[col].median()
            for r in df['satir'].unique():
                m = df['satir'] == r
                if m.sum() > 4:
                    df.loc[m, col] = detrend(df.loc[m, col], type='linear')
            df[col] -= df[col].median()

        df['sutun_m'] = (df['sutun'] - df['sutun'].min()) * self.adim_m
        df['satir_m'] = (df['satir'] - df['satir'].min()) * self.adim_m

        self.df = df
        self.n_satir = int(df['satir'].nunique())
        self.n_sutun = int(df['sutun'].nunique())
        self.sutun_min = int(df['sutun'].min())
        self.satir_min = int(df['satir'].min())
        self.sutun_max = int(df['sutun'].max())
        self.satir_max = int(df['satir'].max())
        self.gurultu_std = df['tfa_diff'].std()
        self.grid_res = max(20, min(DEFAULT_GRID_RES,
                                    max(self.n_satir, self.n_sutun) * 4))

    def _grid_olustur(self, veri_col):
        """Grid oluştur"""
        xi = np.linspace(self.df['sutun_m'].min(), self.df['sutun_m'].max(), self.grid_res)
        yi = np.linspace(self.df['satir_m'].min(), self.df['satir_m'].max(), self.grid_res)
        grid_X, grid_Y = np.meshgrid(xi, yi)

        zi = griddata(
            (self.df['sutun_m'], self.df['satir_m']),
            self.df[veri_col],
            (grid_X, grid_Y),
            method='linear', fill_value=0)
        return xi, yi, grid_X, grid_Y, zi

    def _fft_filtre(self, zi, mod='derin'):
        """FFT filtre"""
        F = np.fft.fftshift(np.fft.fft2(zi))
        rows, cols = zi.shape
        u = np.fft.fftshift(np.fft.fftfreq(cols))
        v = np.fft.fftshift(np.fft.fftfreq(rows))
        UU, VV = np.meshgrid(u, v)
        R = np.sqrt(UU**2 + VV**2)

        def han(r, fc, bw=0.04):
            m = np.ones_like(r)
            low = fc - bw
            high = fc + bw
            transit = (r > low) & (r < high)
            m[transit] = 0.5 * (1 + np.cos(np.pi * (r[transit] - low) / bw))
            m[r >= high] = 0.0
            return m

        if mod == 'derin':
            filtre = han(R, 0.08)
        else:
            filtre = 1.0 - han(R, 0.10)

        return np.real(np.fft.ifft2(np.fft.ifftshift(F * filtre)))

    def filtrele_ve_analiz(self, vcol, gain, esik, blur, noise, sigma, mod):
        """Filtreleme ve analiz"""
        xi, yi, grid_X, grid_Y, zi_raw = self._grid_olustur(vcol)
        self.xi_arr = xi
        self.yi_arr = yi
        self.grid_x = grid_X
        self.grid_y = grid_Y
        self.zi_raw = zi_raw

        zi = zi_raw * gain
        med = int(noise)
        if med > 1:
            size = med if med % 2 else med + 1
            zi = median_filter(zi, size=size)

        if blur > 0:
            zi = gaussian_filter(zi, sigma=blur)

        oto = self.gurultu_std * sigma * gain
        esik_final = max(oto, esik)
        zi = np.where(np.abs(zi) < esik_final, 0, zi)

        if mod == 'Gradient':
            zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
        elif mod == 'Analitik':
            zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
        elif mod == 'FFT Derin':
            zi = self._fft_filtre(zi, 'derin')
        elif mod == 'FFT Sığ':
            zi = self._fft_filtre(zi, 'sig')

        self.zi = zi

        # Hedef tespiti
        hedef_esik = esik_final * 0.60
        min_esik = self.gurultu_std * 0.8 * gain
        hedef_esik = max(hedef_esik, min_esik)

        binary = np.abs(zi) > hedef_esik
        labeled, num = label(binary)
        rows, cols = zi.shape
        targets = []

        for i in range(1, num + 1):
            mask = labeled == i
            if mask.sum() < 2:
                continue
            coords = np.argwhere(mask)
            peak = np.argmax(np.abs(zi[mask]))
            py, px = coords[peak]
            py = min(py, rows - 1)
            px = min(px, cols - 1)
            targets.append({
                'id': i,
                'x': xi[px],
                'y': yi[py],
                'amp': zi[py, px],
            })

        self.targets = sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:8]
        return zi, xi, yi, esik_final

    def _derinlik_pro(self, profil, eks):
        """Derinlik hesaplama"""
        try:
            if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01:
                return self._derinlik(np.max(np.abs(profil))), None, "Profil yetersiz"

            adim = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
            sonuclar = []
            yontemler = []
            abs_p = np.abs(profil)
            tv = abs_p.max()
            ti = int(np.argmax(abs_p))
            snr = tv / (self.gurultu_std + 1e-9)

            if tv > 1e-6:
                y2 = tv * 0.5
                sol = np.where(abs_p[:ti] < y2)[0]
                sag = np.where(abs_p[ti:] < y2)[0]
                if len(sol) > 0 and len(sag) > 0:
                    x12 = (ti - sol[-1] + sag[0]) * adim / 2.0
                    dp = max(0.0, (0.6 * 0.5 + 0.4 * 0.7) * x12 - YUKSEKLIK_SABITI)
                    if 0.01 < dp < MAX_DEPTH:
                        w = min(0.5, 0.15 + 0.35 * min(snr / 10.0, 1.0))
                        sonuclar.append((dp, w))
                        yontemler.append(f"P½={dp:.2f}m")

            if not sonuclar:
                return self._derinlik(tv), None, "Yeterli profil yok"

            tw = sum(w for _, w in sonuclar)
            df = sum(d * w for d, w in sonuclar) / tw
            df = max(0.0, min(df, MAX_DEPTH))

            return df, int(min(snr / 20.0, 1.0) * 100), " | ".join(yontemler)
        except Exception:
            peak_val = np.max(np.abs(profil)) if len(profil) > 0 else 0.0
            return self._derinlik(peak_val), None, "Hata"

    def _derinlik(self, peak_nt):
        """Basit derinlik hesaplama"""
        if abs(peak_nt) < 1e-9:
            return 0.0
        w_half = SENSOR_MESAFESI * 0.5
        d = max(0.0, w_half - YUKSEKLIK_SABITI)
        return min(d, MAX_DEPTH)

    def _dipol_fit(self, profil, eks):
        """Dipol fit metal vs boşluk"""
        try:
            if len(profil) < 5:
                return None, None, None, None, None, None, "Profil yetersiz"

            x = eks - np.mean(eks)
            amp = float(profil[np.argmax(np.abs(profil))])
            if abs(amp) < 1e-9:
                return None, None, None, None, None, None, "Sinyal yok"

            def metal_m(x, M, z, x0):
                xc = x - x0
                z2 = max(abs(z), 0.01)**2
                return M * (2 * z2 - xc**2) / (xc**2 + z2)**2.5

            def bosluk_m(x, K, z, x0):
                xc = x - x0
                z2 = max(abs(z), 0.01)**2
                return -abs(K) * abs(z) / (xc**2 + z2)**1.5

            def r2(g, t):
                return max(0.0, 1.0 - np.sum((g - t)**2) /
                           (np.sum((g - np.mean(g))**2) + 1e-12))

            r2m, zm, Mm, fitm = 0.0, None, None, None
            try:
                po, _ = curve_fit(metal_m, x, profil,
                                  p0=[amp * 0.1, 0.3, 0.0],
                                  bounds=([-abs(amp) * 10, 0.01, -2.0],
                                          [abs(amp) * 10, MAX_DEPTH, 2.0]),
                                  maxfev=2000, ftol=1e-6)
                Mm, zfm, _ = po
                fitm = metal_m(x, *po)
                r2m = r2(profil, fitm)
                zm = max(0.0, abs(zfm) - YUKSEKLIK_SABITI)
            except Exception:
                pass

            r2b, zb, fitb = 0.0, None, None
            try:
                po, _ = curve_fit(bosluk_m, x, profil,
                                  p0=[abs(amp) * 0.5, 0.3, 0.0],
                                  bounds=([0.0, 0.01, -2.0],
                                          [abs(amp) * 20, MAX_DEPTH, 2.0]),
                                  maxfev=2000, ftol=1e-6)
                fitb = bosluk_m(x, *po)
                r2b = r2(profil, fitb)
                zb = max(0.0, abs(po[1]) - YUKSEKLIK_SABITI)
            except Exception:
                pass

            if Mm is not None and Mm < 0 and r2m > 0.60:
                if fitb is None:
                    fitb, r2b, zb = fitm, r2m, zm
                r2m, zm, Mm = 0.0, None, None

            metal_ok = r2m >= 0.35 and Mm is not None and Mm > 0
            bosluk_ok = r2b >= 0.35

            if not metal_ok and not bosluk_ok:
                bf = fitm if fitm is not None else fitb
                return (zm or zb or 0.0), max(r2m, r2b), r2m, r2b, bf, 'belirsiz', "Gürültü/zemin"

            tip = ('metal' if (metal_ok and (not bosluk_ok or r2m >= r2b))
                   else 'bosluk')

            if tip == 'metal':
                zs, r2s, fs = zm, r2m, fitm
            else:
                zs = zb if zb is not None else zm
                r2s = r2b if bosluk_ok else r2m
                fs = fitb if fitb is not None else fitm

            return (zs or 0.0), r2s, r2m, r2b, fs, tip, "Uygun fit"
        except Exception:
            return None, None, None, None, None, None, "Hesaplanamadı"


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="C3 MANYETİK ANALİZ", layout="wide", initial_sidebar_state="collapsed")
    
    # CSS Styling
    st.markdown("""
    <style>
        body { background-color: #0a0a0f; }
        .main { background-color: #0a0a0f; }
        .stTabs [data-baseweb="tabs"] { background-color: #0a0a0f; }
        h1, h2, h3 { color: #00FF9D !important; }
        .metric-box { background-color: #1a1a2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00FF9D; }
        .target-card { background-color: #1a1a2e; padding: 12px; border-radius: 6px; margin: 8px 0; border-left: 4px solid #FFD700; }
        .info-panel { background-color: #0d0d0d; padding: 15px; border-radius: 8px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧲 C3 MANYETİK GRADİOMETRE ANALİZİ")

    # Sidebar - Ayarlar
    with st.sidebar:
        st.header("⚙️ Tarama Ayarları")
        
        uploaded_file = st.file_uploader("CSV Dosyası Seç", type="csv")
        
        if uploaded_file is None:
            st.warning("Lütfen bir CSV dosyası yükleyin")
            return

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Hata: {e}")
            return

        st.subheader("Adım Mesafesi")
        adim_secim = st.radio("Seç:", ["20 cm", "25 cm", "50 cm", "Manuel"], key="adim_sec")
        
        if adim_secim == "Manuel":
            adim_cm = st.number_input("Adım mesafesi (cm):", min_value=1, max_value=500, value=50)
        else:
            adim_cm = int(adim_secim.split()[0])
        
        adim_m = adim_cm / 100.0

        st.subheader("Filtre Parametreleri")
        gain = st.slider("Kazanç", min_value=1, max_value=1000, value=100, step=10)
        esik = st.slider("Eşik (nT)", min_value=0, max_value=500, value=25, step=5)
        blur = st.slider("Yumuşatma", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        noise = st.slider("Gürültü Filtres (Medyan)", min_value=1, max_value=9, value=3, step=2)
        sigma = st.slider("Sigma (σ)", min_value=1, max_value=3, value=3, step=1)
        
        st.subheader("Analiz Modu")
        mod = st.radio("Modu Seç:", 
                       ["TFA", "Sadece Z", "Gradient", "Analitik", "FFT Derin", "FFT Sığ"],
                       key="mod_sec")
        
        vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'

    # Ana analiz
    try:
        analiz = C3Analiz(df, adim_m=adim_m)
        zi, xi, yi, esik_final = analiz.filtrele_ve_analiz(vcol, gain, esik, blur, noise, sigma, mod)

        # Ana görüntü
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Harita Görüntüsü")
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0a0f')
            ax.set_facecolor('#0d0d0d')
            
            zmin, zmax = zi.min(), zi.max()
            if zmin < 0 < zmax:
                nz = zi[zi != 0]
                ph = np.percentile(np.abs(nz), 98) if len(nz) > 0 else max(abs(zmin), abs(zmax))
                norm = TwoSlopeNorm(vmin=-max(ph, 0.001), vcenter=0, vmax=max(ph, 0.001))
            else:
                norm = Normalize(vmin=zmin, vmax=zmax)

            im = ax.imshow(zi,
                          extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                          origin='lower', cmap=C3_CMAP, norm=norm,
                          aspect='equal', interpolation='bilinear')

            # Hedefleri işaretle
            for t in analiz.targets:
                ax.plot(t['x'], t['y'], 'w+', ms=12, mew=2)
                ax.text(t['x'] + 0.02, t['y'] + 0.02, f"H{t['id']}",
                       color='white', fontsize=9, weight='bold')

            x0 = analiz.df['sutun_m'].min()
            y0 = analiz.df['satir_m'].min()
            ax.plot(x0, y0, '*', color='yellow', ms=12, zorder=5)

            ax.set_xlabel("Sütun (m)", color='#aaa', fontsize=10)
            ax.set_ylabel("Satır (m)", color='#aaa', fontsize=10)
            ax.set_title(f"C3 — {mod} | {len(analiz.targets)} Hedef", 
                        color='#00FF9D' if analiz.targets else '#FF4444', fontsize=12, weight='bold')
            ax.tick_params(colors='#555', labelsize=8)
            
            for sp in ax.spines.values():
                sp.set_edgecolor('#222')

            plt.colorbar(im, ax=ax, label='nT', pad=0.02)
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Hedefler")
            
            if analiz.targets:
                hedef_secim = st.radio(
                    "Hedef Seç:",
                    [f"H{t['id']} - {t['amp']:.1f}nT" for t in analiz.targets],
                    key="hedef_sec"
                )
                
                hedef_idx = int(hedef_secim.split()[0][1:]) - 1
                secili_hedef = analiz.targets[hedef_idx]
            else:
                st.warning("Anomali tespit edilmedi")
                secili_hedef = None

        # Detay bilgisi
        if secili_hedef:
            st.markdown("---")
            st.subheader(f"📍 Hedef H{secili_hedef['id']} Detayları")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Konum**")
                st.info(f"X: {secili_hedef['x']:.3f} m\nY: {secili_hedef['y']:.3f} m")

            with col_b:
                st.markdown("**Şiddet**")
                st.info(f"{secili_hedef['amp']:.2f} nT")

            with col_c:
                st.markdown("**Hızlı Stat**")
                st.info(f"Kazanç: {gain}\nEşik: {esik} nT")

            # Profiller
            try:
                ci = np.argmin(np.abs(xi - secili_hedef['x']))
                ri = np.argmin(np.abs(yi - secili_hedef['y']))

                xp = analiz.zi_raw[ri, :] * gain
                yp = analiz.zi_raw[:, ci] * gain
                xn = analiz.zi_raw[ri, :]

                depth, guven, ystr = analiz._derinlik_pro(xn, xi)
                z_d, r2s, r2m, r2b, fp, tip, dyorum = analiz._dipol_fit(xn, xi)

                # X ve Y profil grafikleri
                fig_prof, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0a0a0f')

                for ax in [ax_x, ax_y]:
                    ax.set_facecolor('#0d0d0d')
                    ax.tick_params(colors='#555', labelsize=8)
                    for sp in ax.spines.values():
                        sp.set_edgecolor('#222')

                # X Profili
                ax_x.plot(xi, xp, color='#FFD700', lw=2, label='Veri')
                if fp is not None:
                    fr = '#00FF9D' if tip == 'metal' else '#00CFFF'
                    ax_x.plot(xi, fp * gain, color=fr, lw=1.5, ls='--', alpha=0.8, label=f'{tip.upper()} Fit')
                ax_x.axhline(0, color='#444', lw=0.5)
                ax_x.axvline(secili_hedef['x'], color='#FF4500', lw=1, ls='--', alpha=0.7)
                ax_x.set_title("X Kesiti (Sütun)", color='#aaa', fontsize=10)
                ax_x.set_ylabel("nT", color='#aaa', fontsize=9)
                ax_x.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333')
                ax_x.grid(True, alpha=0.2, color='#333')

                # Y Profili
                ax_y.plot(yi, yp, color='#00CFFF', lw=2, label='Veri')
                ax_y.axhline(0, color='#444', lw=0.5)
                ax_y.axvline(secili_hedef['y'], color='#FF4500', lw=1, ls='--', alpha=0.7)
                ax_y.set_title("Y Kesiti (Satır)", color='#aaa', fontsize=10)
                ax_y.set_ylabel("nT", color='#aaa', fontsize=9)
                ax_y.set_xlabel("Mesafe (m)", color='#aaa', fontsize=9)
                ax_y.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333')
                ax_y.grid(True, alpha=0.2, color='#333')

                plt.tight_layout()
                st.pyplot(fig_prof, use_container_width=True)

                # Analiz sonuçları
                st.markdown("---")
                st.subheader("🔬 Analiz Sonuçları")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.markdown("**Derinlik & Güven**")
                    depth_val = float(depth) if depth is not None else 0.0
                    st.success(f"Derinlik: ~{depth_val:.2f} m\nGüven: %{guven or '?'}")

                with result_col2:
                    st.markdown("**Dipol Analiz**")
                    if tip:
                        tip_renk = "🔴 Metal" if tip == 'metal' else "🔵 Boşluk" if tip == 'bosluk' else "⚪ Belirsiz"
                        st.info(f"{tip_renk}\nR²: {r2s:.2f}\n{dyorum}")

                with result_col3:
                    st.markdown("**Uyum Oranları**")
                    r2m_v = r2m if r2m is not None else 0.0
                    r2b_v = r2b if r2b is not None else 0.0
                    st.metric("Metal R²", f"{r2m_v:.2f}")
                    st.metric("Boşluk R²", f"{r2b_v:.2f}")

                # Obje Tahmini
                tahminler = obje_tahmini(tip, r2m_v, r2b_v, None, depth_val,
                                        secili_hedef['amp'], 
                                        abs(secili_hedef['amp']) / (analiz.gurultu_std * gain + 1e-9),
                                        mod)

                st.markdown("---")
                st.subheader("🏛️ Olası Objeler")

                for isim, puan, _ in tahminler:
                    puan_yuzde = int(puan * 100)
                    st.progress(puan, text=f"{isim} - %{puan_yuzde}")

                # Rapor indir
                st.markdown("---")
                rapor_data = {
                    'tarih': datetime.now().isoformat(),
                    'hedef_id': secili_hedef['id'],
                    'konum_x': float(secili_hedef['x']),
                    'konum_y': float(secili_hedef['y']),
                    'siddet_nT': float(secili_hedef['amp']),
                    'derinlik_m': depth_val,
                    'guven_pct': guven,
                    'dipol_tipi': tip,
                    'r2_metal': float(r2m) if r2m is not None else None,
                    'r2_bosluk': float(r2b) if r2b is not None else None,
                }

                rapor_json = json.dumps(rapor_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 Rapor İndir (JSON)",
                    data=rapor_json,
                    file_name=f"hedef_H{secili_hedef['id']}_rapor.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Detay analizi hatası: {e}")

    except Exception as e:
        st.error(f"Analiz hatası: {e}")


if __name__ == "__main__":
    main()
