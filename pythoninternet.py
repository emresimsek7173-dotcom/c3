import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from scipy.optimize import curve_fit
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  SABİTLER
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_MESAFESI  = 0.80   # iki sensör arası mesafe (m)
YUKSEKLIK_SABITI = 0.05   # sensörün yerden yüksekliği (m)
MAX_DEPTH        = 5.0    # maksimum derinlik (m)
DEFAULT_GRID_RES = 100    # orijinal kodundaki grid çözünürlüğü

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA', '#0066FF', '#00CCFF', '#00CC44',
    '#FFFF00', '#FF6600', '#CC0000'], N=512)

# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI — Türkçe karakter normalizasyonu (Orijinal koddan)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_col(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return (s.strip().lower().replace(' ', '_'))

# ─────────────────────────────────────────────────────────────────────────────
#  MATEMATİKSEL MOTORLAR (Orijinal c3q_analiz.py'den Birebir Kopyalandı)
# ─────────────────────────────────────────────────────────────────────────────
def _fft_filtre(zi, mod='derin'):
    F = np.fft.fftshift(np.fft.fft2(zi))
    rows, cols = zi.shape
    u = np.fft.fftshift(np.fft.fftfreq(cols))
    v = np.fft.fftshift(np.fft.fftfreq(rows))
    UU, VV = np.meshgrid(u, v)
    R = np.sqrt(UU**2 + VV**2)
    def han(r, fc, bw=0.04):
        m = np.ones_like(r)
        low, high = fc - bw, fc + bw
        transit = (r > low) & (r < high)
        m[transit] = 0.5 * (1 + np.cos(np.pi * (r[transit] - low) / bw))
        m[r >= high] = 0.0
        return m
    filtre = han(R, 0.08) if mod == 'derin' else (1.0 - han(R, 0.10))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * filtre)))

def _faz_kaymasi(x_prof):
    try:
        neg = x_prof < 0
        if len(neg) < 2 or neg.sum() < 2: return 0.0, "Negatif bolge yok"
        neg_idx = np.where(neg)[0]
        cukur = int(np.mean(neg_idx))
        ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tepe = int(np.argmax(ana))
        mesafe = abs(tepe - cukur)
        ortusme = max(0.0, 1.0 - mesafe / (len(x_prof) * 0.3))
        return ortusme, "Manyetik olmayan/bosluk" if ortusme > 0.80 else "Karisik sinyal" if ortusme > 0.50 else "Demir dipol"
    except: return 0.0, "Hesaplanamadi"

def _tepe_sivrilik(x_prof, xi):
    try:
        ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tv = ana.max()
        if tv < 1e-6: return None, "Tepe yok"
        idx = np.where(ana >= tv * 0.5)[0]
        if len(idx) < 2: return None, "Tepe dar"
        adim = (xi[-1] - xi[0]) / max(len(xi) - 1, 1)
        fwhm = (idx[-1] - idx[0]) * adim
        return fwhm, "Sivri → kucuk yogun" if fwhm < 0.3 else "Orta → hacimli kutle" if fwhm < 0.8 else "Yayvan → buyuk yapi"
    except: return None, "Hesaplanamadi"

def _derinlik_pro(profil, eks, gurultu_std):
    try:
        if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01: return 0.1, None, "Profil yetersiz"
        adim = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
        sonuclar, yontemler = [], []
        abs_p = np.abs(profil)
        tv = abs_p.max(); ti = int(np.argmax(abs_p)); snr = tv / (gurultu_std + 1e-9)
        if tv > 1e-6:
            y2 = tv * 0.5; sol = np.where(abs_p[:ti] < y2)[0]; sag = np.where(abs_p[ti:] < y2)[0]
            if len(sol) > 0 and len(sag) > 0:
                x12 = (ti - sol[-1] + sag[0]) * adim / 2.0
                dp = max(0.0, 0.65 * x12 - YUKSEKLIK_SABITI)
                if 0.01 < dp < MAX_DEPTH: sonuclar.append((dp, min(0.5, 0.15 + 0.35 * min(snr / 10.0, 1.0)))); yontemler.append(f"P½={dp:.2f}m")
        grad = np.gradient(profil, adim); ana = np.sqrt(grad**2 + profil**2); at = ana.max()
        if at > 1e-6:
            idx = np.where(ana >= at * 0.5)[0]
            if len(idx) >= 2:
                fwhm = (idx[-1] - idx[0]) * adim; fd = np.sqrt(max(fwhm**2 - (SENSOR_MESAFESI * 0.5)**2, fwhm**2 * 0.1))
                df2 = max(0.0, fd / 2.0 - YUKSEKLIK_SABITI)
                if 0.01 < df2 < MAX_DEPTH: sonuclar.append((df2, min(0.40, 0.10 + 0.30 * (len(idx) / len(profil))))); yontemler.append(f"FWHM={df2:.2f}m")
        if not sonuclar: return 0.2, None, "Yeterli profil yok"
        tw = sum(w for _, w in sonuclar); df = max(0.0, min(sum(d * w for d, w in sonuclar) / tw, MAX_DEPTH))
        guven = int(((1.0 - min(np.std([d for d,_ in sonuclar])/(np.mean([d for d,_ in sonuclar])+0.01), 1.0))*0.7 + min(snr/15.0, 1.0)*0.3)*100) if len(sonuclar)>=2 else max(20, int(min(snr/20.0, 1.0)*50))
        return df, guven, " | ".join(yontemler)
    except: return 0.2, None, "Hata"

def _dipol_fit(profil, eks, gurultu_std):
    try:
        if len(profil) < 5: return None, None, None, None, None, None, "Profil yetersiz"
        x = eks - np.mean(eks); amp = float(profil[np.argmax(np.abs(profil))])
        if abs(amp) < 1e-9: return None, None, None, None, None, None, "Sinyal yok"
        def metal_m(x, M, z, x0): xc = x - x0; z2 = max(abs(z), 0.01)**2; return M * (2 * z2 - xc**2) / (xc**2 + z2)**2.5
        def bosluk_m(x, K, z, x0): xc = x - x0; z2 = max(abs(z), 0.01)**2; return -abs(K) * abs(z) / (xc**2 + z2)**1.5
        def r2(g, t): return max(0.0, 1.0 - np.sum((g - t)**2) / (np.sum((g - np.mean(g))**2) + 1e-12))
        r2m, zm, Mm, fitm = 0.0, None, None, None
        try:
            po, _ = curve_fit(metal_m, x, profil, p0=[amp*0.1, 0.3, 0.0], bounds=([-abs(amp)*10, 0.01, -2.0], [abs(amp)*10, MAX_DEPTH, 2.0]), maxfev=2000, ftol=1e-6)
            Mm, zfm, _ = po; fitm = metal_m(x, *po); r2m = r2(profil, fitm); zm = max(0.0, abs(zfm) - YUKSEKLIK_SABITI)
        except: pass
        r2b, zb, fitb = 0.0, None, None
        try:
            po, _ = curve_fit(bosluk_m, x, profil, p0=[abs(amp)*0.5, 0.3, 0.0], bounds=([0.0, 0.01, -2.0], [abs(amp)*20, MAX_DEPTH, 2.0]), maxfev=2000, ftol=1e-6)
            fitb = bosluk_m(x, *po); r2b = r2(profil, fitb); zb = max(0.0, abs(po[1]) - YUKSEKLIK_SABITI)
        except: pass
        if Mm is not None and Mm < 0 and r2m > 0.60:
            if fitb is None: fitb, r2b, zb = fitm, r2m, zm
            r2m, zm, Mm = 0.0, None, None
        metal_ok, bosluk_ok = (r2m >= 0.35 and Mm is not None and Mm > 0), (r2b >= 0.35)
        if not metal_ok and not bosluk_ok: return (zm or zb or 0.0), max(r2m, r2b), r2m, r2b, (fitm if fitm is not None else fitb), 'belirsiz', "Gurultu/zemin"
        tip = 'metal' if (metal_ok and (not bosluk_ok or r2m >= r2b)) else 'bosluk'
        if tip == 'metal': return zm, r2m, r2m, r2b, fitm, 'metal', ("Guclu metal" if r2m >= 0.85 else "Orta metal")
        else: return zb, r2b, r2m, r2b, fitb, 'bosluk', ("Guclu bosluk" if r2b >= 0.75 else "Orta bosluk")
    except: return None, None, None, None, None, None, "Hesaplanamadi"

def _teshis(x_prof, val, tip, r2_m, r2_b, ortusme, fwhm, esik, mod):
    if abs(val) < esik: return "TEMIZ/SINYAL YOK", "#FFFFFF", "Anomali yok."
    if mod == 'Analitik': return "ENERJI MERKEZI", "#FF00FF", "Hedef odak noktasi."
    if mod == 'Gradient': return "KENAR/SINIR", "#FFA500", "Anomali siniri."
    vmax, vmin = float(np.max(x_prof)), float(np.min(x_prof)); vr = max(vmax - vmin, 1e-5)
    mo = bo = blo = 0.0
    if vmax > vr * 0.15 and vmin < -vr * 0.15:
        if abs(vmin) > vmax: mo += 0.25
        else: bo += 0.15; mo += 0.10
    elif vmin < -vr * 0.15: bo += 0.25
    elif vmax > vr * 0.15: mo += 0.20; blo += 0.05
    r2m_v, r2b_v = (r2_m or 0.0), (r2_b or 0.0)
    mo += 0.40 * r2m_v; bo += 0.40 * r2b_v
    if ortusme is not None:
        if ortusme > 0.80: bo += 0.20
        elif ortusme > 0.50: blo += 0.20
        else: mo += 0.20
    total = mo + bo + blo + 1e-9; pm, pb = mo/total, bo/total
    if abs(pm - pb) < 0.15 and max(pm, pb) > 0.30: return "KARMA YAPI", "#FFD700", f"M:%{int(pm*100)} B:%{int(pb*100)}"
    if pm > pb and pm > 0.4: return "METAL KUTLE", "#FF4500", f"Metal ihtimali %{int(pm*100)}"
    if pb > pm and pb > 0.4: return "BOSLUK/TUNEL", "#00FFFF", f"Bosluk ihtimali %{int(pb*100)}"
    return "BELIRSIZ", "#888888", "Yetersiz sinyal."

def obje_tahmini(tip, r2_m, r2_b, fwhm, derinlik, val, snr, mod):
    tahminler = []
    d = derinlik or 0.0; fw = fwhm or 0.5
    if tip == 'metal' and r2_m >= 0.35:
        if fw < 0.25 and d < 0.5: tahminler.append(("Kucuk Metal Obje (Sikke/Parca)", r2_m*0.9, '#FF6600'))
        elif fw < 0.7: tahminler.append(("Metal Kup / Sandik / Obje", r2_m*0.85, '#FF4500'))
        else: tahminler.append(("Metal Boru / Yapay Yapi", r2_m*0.8, '#FF2200'))
    elif tip == 'bosluk' and r2_b >= 0.35:
        if fw < 1.0: tahminler.append(("Seramik Kup / Kucuk Bosluk", r2_b*0.8, '#00CFFF'))
        else: tahminler.append(("Oda Mezar / Tunel / Mahzen", r2_b*0.85, '#0066FF'))
    else: tahminler.append(("Dogal Mineral / Kaya / Zemin", 0.4, '#888888'))
    return tahminler[:2]

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="C3 Kesintisiz Analiz Motoru", layout="wide")
st.markdown("<h2 style='color:#00FF9D; text-align:center;'>C3 GRADIOMETRE STREAMLIT PANELI</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    uploaded_file = st.file_uploader("C3 CSV Dosyasını Seçin", type=["csv"])
    adim_mesafesi = st.number_input("Adım Mesafesi (cm)", min_value=1.0, max_value=200.0, value=50.0) / 100.0
    mod = st.radio("Görüntüleme Modu", ["TFA Farkı", "Sadece Z", "Gradient", "Analitik", "FFT Derin", "FFT Sig"])
    
    st.subheader("🎛️ Filtre Kontrolleri")
    gain = st.slider("Gain (Kazanç)", 0.1, 50.0, 1.0, 0.1)
    noise_filter = st.slider("Median Filtre", 1, 9, 1, 2)
    blur = st.slider("Gaussian Blur", 0.0, 3.0, 0.0, 0.1)
    sigma_esik = st.slider("Oto Eşik (Sigma)", 0.0, 4.0, 1.0, 0.1)

if uploaded_file is not None:
    # 1. VERİ TEMİZLEME VE NORMALİZASYON (Orijinal Kod Sıralamasıyla)
    df = pd.read_csv(uploaded_file)
    df.columns = [_normalize_col(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    # TFA ve Fark Hesapları
    df['tfa1'] = np.sqrt(df['s1_x']**2 + df['s1_y']**2 + df['s1_z']**2)
    df['tfa2'] = np.sqrt(df['s2_x']**2 + df['s2_y']**2 + df['s2_z']**2)
    
    # %2 ve %98 Kırpma (Orijinal filtre mantığı)
    for col in ['tfa1', 'tfa2', 's1_z', 's2_z']:
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo, hi)

    df['tfa_diff'] = df['tfa1'] - df['tfa2']
    df['z_diff']   = df['s1_z'] - df['s2_z']

    # Çizgi Trend Temizliği (Detrend)
    for col in ['tfa_diff', 'z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir'] == r
            if m.sum() > 4:
                df.loc[m, col] = detrend(df.loc[m, col], type='linear')
        df[col] -= df[col].median()

    gurultu_std = df['tfa_diff'].std()
    vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'

    # Metreye Çevirme Kontrolü
    df['sutun_m'] = (df['sutun'] - df['sutun'].min()) * adim_mesafesi
    df['satir_m'] = (df['satir'] - df['satir'].min()) * adim_mesafesi

    n_satir, n_sutun = df['satir'].nunique(), df['sutun'].nunique()
    sutun_min, sutun_max = df['sutun'].min(), df['sutun'].max()
    satir_min, satir_max = df['satir'].min(), df['satir'].max()

    # 2. VERİLERİ ASLA SIKIŞTIRMAYAN ORİJİNAL GRIDDATA INTERPOLASYON MOTORU
    xi = np.linspace(df['sutun_m'].min(), df['sutun_m'].max(), DEFAULT_GRID_RES)
    yi = np.linspace(df['satir_m'].min(), df['satir_m'].max(), DEFAULT_GRID_RES)
    grid_X, grid_Y = np.meshgrid(xi, yi)

    # 1000 satır verinin tamamını grid yapısına eksiksiz dağıtan asıl motor:
    zi_raw = griddata((df['sutun_m'], df['satir_m']), df[vcol], (grid_X, grid_Y), method='linear', fill_value=0)

    # Filtrelerin Uygulanması
    zi = zi_raw * gain
    if noise_filter > 1: zi = median_filter(zi, size=noise_filter)
    if blur > 0: zi = gaussian_filter(zi, sigma=blur)
    
    esik = gurultu_std * sigma_esik * gain
    zi = np.where(np.abs(zi) < esik, 0, zi)

    if mod == 'Gradient':    zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
    elif mod == 'Analitik':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
    elif mod == 'FFT Derin': zi = _fft_filtre(zi, 'derin')
    elif mod == 'FFT Sig':   zi = _fft_filtre(zi, 'sig')

    # Hedef Odak Noktalarının Yakalanması
    hedef_esik = max((esik * 0.60), (gurultu_std * 0.8 * gain))
    binary = np.abs(zi) > hedef_esik
    labeled, num = label(binary)
    rows, cols = zi.shape
    targets = []
    for i in range(1, num + 1):
        mask = labeled == i
        if mask.sum() < 2: continue
        coords = np.argwhere(mask)
        peak = np.argmax(np.abs(zi[mask]))
        py, px = coords[peak]
        py, px = min(py, rows - 1), min(px, cols - 1)
        targets.append({'id': i, 'x': xi[px], 'y': yi[py], 'amp': zi[py, px], 'px': px, 'py': py})
    targets = sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:8]

    # Ekran Düzeni (2 Kolon)
    col1, col2 = st.columns([1.8, 1.2])

    with col1:
        st.subheader("🗺️ 2D Isı Haritası (Orijinal Eksen Oranları)")
        fig, ax2d = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#0a0a0f')
        ax2d.set_facecolor('#0a0a0f')

        zmin, zmax = zi.min(), zi.max()
        if zmin < 0 < zmax:
            nz = zi[zi != 0]
            ph = np.percentile(np.abs(nz), 98) if len(nz) > 0 else max(abs(zmin), abs(zmax))
            norm = TwoSlopeNorm(vmin=-max(ph, 0.001), vcenter=0, vmax=max(ph, 0.001))
        else:
            norm = Normalize(vmin=zmin, vmax=zmax)

        # Orijinal `extent` sınırlandırmasıyla çizim
        im = ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap=C3_CMAP, norm=norm, aspect='equal', interpolation='bilinear')
        
        for t in targets:
            ax2d.plot(t['x'], t['y'], 'w+', ms=12, mew=2)
            ax2d.text(t['x'] + .02, t['y'] + .02, f"H{t['id']}", color='white', fontsize=9, weight='bold')

        # Başlangıç Noktası Yıldızı (*)
        x0, y0 = df['sutun_m'].min(), df['satir_m'].min()
        ax2d.plot(x0, y0, '*', color='yellow', ms=12, zorder=5)

        # Gerçek Eksen Etiketleri (R0-RMax, S0-SMax)
        sutun_sayilari = list(range(sutun_min, sutun_max + 1))
        sutun_metreleri = [(s - sutun_min) * adim_mesafesi for s in sutun_sayilari]
        satir_sayilari = list(range(satir_min, satir_max + 1))
        satir_metreleri = [(s - satir_min) * adim_mesafesi for s in satir_sayilari]

        ax2d.set_xticks(sutun_metreleri)
        ax2d.set_xticklabels([f"S{s}" for s in sutun_sayilari], fontsize=7, color='#aaa')
        ax2d.set_yticks(satir_metreleri)
        ax2d.set_yticklabels([f"R{s}" for s in satir_sayilari], fontsize=7, color='#aaa')
        
        ax2d.set_xlabel(f"SÜTUN ({n_sutun} sütun × {adim_mesafesi*100:.0f}cm = {(n_sutun-1)*adim_mesafesi:.1f}m)", fontsize=8, color='#666')
        ax2d.set_ylabel(f"SATIR ({n_satir} satır × {adim_mesafesi*100:.0f}cm = {(n_satir-1)*adim_mesafesi:.1f}m)", fontsize=8, color='#666')
        
        title_color = '#00FF9D' if targets else '#FF4444'
        ax2d.set_title(f"C3 — {mod} | {len(targets)} Hedef Tespiti", color=title_color, fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("🎯 Nokta Detay ve Teşhis Paneli")
        if targets:
            selected_target_id = st.selectbox("İncelemek İstediğiniz Hedefi Seçin", [f"Hedef H{t['id']}" for t in targets])
            sel_target = next(t for t in targets if f"Hedef H{t['id']}" == selected_target_id)
            
            ri, ci = sel_target['py'], sel_target['px']
            xp = zi_raw[ri, :] * gain
            val = float(zi[ri, ci]) if abs(float(zi[ri, ci])) > 0 else float(xp[ci])

            # Profil Analizleri
            depth, guven, ystr = _derinlik_pro(xp, xi, gurultu_std)
            ortusme, faz_yorumu = _faz_kaymasi(xp)
            fwhm, form_yorumu = _tepe_sivrilik(xp, xi)
            
            zs, r2s, r2m, r2b, fit_profil, fit_tip, fit_yorum = _dipol_fit(xp, xi, gurultu_std)
            teshis_adi, teshis_renk, teshis_detay = _teshis(xp, val, fit_tip, r2m, r2b, ortusme, fwhm, esik, mod)
            olasi_objeler = obje_tahmini(fit_tip, r2m, r2b, fwhm, depth, val, (abs(val)/(gurultu_std+1e-9)), mod)

            # Bilgi Kartları
            st.markdown(f"""
            <div style='padding:15px; background-color:#14141f; border-left: 5px solid {teshis_renk}; border-radius:4px; margin-bottom:15px;'>
                <h4 style='margin:0; color:{teshis_renk};'>{teshis_adi}</h4>
                <p style='margin:5px 0 0 0; color:#aaa; font-size:14px;'>{teshis_detay}</p>
            </div>
            """, unsafe_allow_html=True)

            st.metric(label="Tahmini Derinlik", value=f"{depth:.2f} m", delta=f"Güven: %{guven or 50}")
            st.text(f"Yöntem: {ystr}")

            st.markdown("### 🔍 Model Verileri")
            st.write(f"**Sinyal Genliği:** {val:.1f} nT")
            st.write(f"**Dipol Uyumu (R²):** %{r2s*100:.1f} ({fit_yorum})")
            st.write(f"**Karakteristik:** {faz_yorumu} | {form_yorumu}")

            st.markdown("### 🏺 Obje Tahminleri")
            for obj, puan, r_renk in olasi_objeler:
                st.markdown(f"🔹 <span style='color:{r_renk}; font-weight:bold;'>{obj}</span> — Olasılık: **%{puan*100:.1f}**", unsafe_allow_html=True)
            
            # Profil Grafiği
            fig_p, ax_p = plt.subplots(figsize=(5, 3))
            fig_p.patch.set_facecolor('#0a0a0f')
            ax_p.set_facecolor('#0a0a0f')
            ax_p.plot(xi, xp, color='#00FF9D', label='Sinyal Kesiti', lw=1.5)
            if fit_profil is not None and len(fit_profil) == len(xi):
                ax_p.plot(xi, fit_profil, '--', color='#FF4500', label='Dipol Fit', lw=1)
            ax_p.axhline(0, color='#333', lw=0.8)
            ax_p.legend(facecolor='#14141f', edgecolor='none', labelcolor='white', fontsize=8)
            ax_p.tick_params(colors='#888', labelsize=8)
            st.pyplot(fig_p)
        else:
            st.info("Bu filtre değerleriyle belirgin bir odak saptanamadı.")
else:
    st.warning("Lütfen analiz için sol panelden CSV dosyanızı yükleyin.")
