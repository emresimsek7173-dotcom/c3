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
DEFAULT_GRID_RES = 100    # varsayılan grid çözünürlüğü

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA', '#0066FF', '#00CCFF', '#00CC44',
    '#FFFF00', '#FF6600', '#CC0000'], N=512)

# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI — Türkçe karakter normalizasyonu
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_col(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return (s.strip().lower().replace(' ', '_'))

# ─────────────────────────────────────────────────────────────────────────────
#  OBJE TAHMİN MOTORU (Birebir Korundu)
# ─────────────────────────────────────────────────────────────────────────────
def obje_tahmini(tip, r2_m, r2_b, fwhm, derinlik, val, snr, mod):
    tahminler = []
    d  = derinlik or 0.0
    fw = fwhm     or 0.5
    am = abs(val)

    if mod in ('Gradient', 'Analitik'):
        tahminler.append(("Kenar/Sinir anomalisi — detay modda incele", 0.50, '#FFA500'))
        return tahminler[:3]

    if tip == 'metal' and r2_m >= 0.35:
        if fw < 0.25 and d < 0.5:
            tahminler.append(("Küçük metal obje (sikke/parça)", r2_m * 0.90 + (1 - d) * 0.10, '#FF6600'))
        elif fw < 0.7 and 0.2 < d < 1.2:
            tahminler.append(("Metal küp / sandık / kap",       r2_m * 0.85,                  '#FF4500'))
        elif fw >= 0.7 and d < 1.5:
            tahminler.append(("Metal boru / ray / levha",        r2_m * 0.80,                  '#FF2200'))
        elif d >= 1.5:
            tahminler.append(("Derin metal yapı",                r2_m * 0.70,                  '#FF0000'))
        tahminler.append(("Metal obje",                          r2_m * 0.60,                  '#FF8C00'))

    if tip == 'bosluk' and r2_b >= 0.35:
        if fw < 0.5 and d < 0.8:
            tahminler.append(("Çömlek / pişmiş toprak kap",     r2_b * 0.85, '#00CFFF'))
            tahminler.append(("Küçük boşluk / hava cebi",       r2_b * 0.70, '#00AACC'))
        elif fw < 1.0 and 0.3 < d < 1.5:
            tahminler.append(("Çömlek küp / seramik obje",      r2_b * 0.80, '#00CFFF'))
            tahminler.append(("Küçük yapı boşluğu",             r2_b * 0.65, '#0088AA'))
        elif fw >= 1.0 or d >= 1.5:
            tahminler.append(("Tünel / oda / mezar odası",      r2_b * 0.75, '#0066FF'))
            tahminler.append(("Büyük boşluk yapısı",            r2_b * 0.60, '#0044CC'))
        tahminler.append(("Boşluk / içi dolu olmayan yapı",     r2_b * 0.50, '#88AAFF'))

    if tip == 'belirsiz' or (r2_m > 0.30 and r2_b > 0.30 and abs(r2_m - r2_b) < 0.20):
        if fw < 0.6 and d < 1.0:
            tahminler.append(("Çömlek (metal içerikli?)",        0.50, '#FFD700'))
            tahminler.append(("Pişmiş kil / karma malzeme",      0.45, '#FFA500'))
        else:
            tahminler.append(("Karma yapı / belirsiz",           0.30, '#888888'))

    if snr < 2.0 or am < 0.5:
        tahminler.append(("Zemin gürültüsü / mineral iz",        0.40, '#666666'))
        tahminler.append(("Manyetik kaya / doğal anomali",       0.35, '#556655'))

    if not tahminler:
        return [("Tanımlanamadı", 0.0, '#555555')]

    tahminler.sort(key=lambda x: x[1], reverse=True)
    return tahminler[:3]

# ─────────────────────────────────────────────────────────────────────────────
#  YEDEK VE GELİŞMİŞ MATEMATİKSEL MOTORLAR (Birebir Korundu)
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
        if len(neg) < 2 or neg.sum() < 2: return 0.0, "Negatif bölge yok"
        neg_idx = np.where(neg)[0]
        cukur = int(np.mean(neg_idx))
        ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tepe = int(np.argmax(ana))
        mesafe = abs(tepe - cukur)
        ortusme = max(0.0, 1.0 - mesafe / (len(x_prof) * 0.3))
        yorum = "Manyetik olmayan/bosluk" if ortusme > 0.80 else "Karisik sinyal" if ortusme > 0.50 else "Demir dipol"
        return ortusme, yorum
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
        form = "Sivri → kucuk yogun" if fwhm < 0.3 else "Orta → hacimli kutle" if fwhm < 0.8 else "Yayvan → buyuk yapi"
        return fwhm, form
    except: return None, "Hesaplanamadi"

def _derinlik_yedek(peak_nt):
    if abs(peak_nt) < 1e-9: return 0.0
    w_half = SENSOR_MESAFESI * 0.5
    return min(max(0.0, w_half - YUKSEKLIK_SABITI), MAX_DEPTH)

def _derinlik_pro(profil, eks, gurultu_std):
    try:
        if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01:
            return _derinlik_yedek(np.max(np.abs(profil))), None, "Profil yetersiz"
        adim = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
        sonuclar, yontemler = [], []
        abs_p = np.abs(profil)
        tv = abs_p.max()
        ti = int(np.argmax(abs_p))
        snr = tv / (gurultu_std + 1e-9)

        if tv > 1e-6:
            y2 = tv * 0.5
            sol = np.where(abs_p[:ti] < y2)[0]
            sag = np.where(abs_p[ti:] < y2)[0]
            if len(sol) > 0 and len(sag) > 0:
                x12 = (ti - sol[-1] + sag[0]) * adim / 2.0
                dp = max(0.0, 0.65 * x12 - YUKSEKLIK_SABITI)
                if 0.01 < dp < MAX_DEPTH:
                    sonuclar.append((dp, min(0.5, 0.15 + 0.35 * min(snr / 10.0, 1.0))))
                    yontemler.append(f"P½={dp:.2f}m")

        grad = np.gradient(profil, adim)
        ana = np.sqrt(grad**2 + profil**2)
        at = ana.max()
        if at > 1e-6:
            idx = np.where(ana >= at * 0.5)[0]
            if len(idx) >= 2:
                fwhm = (idx[-1] - idx[0]) * adim
                fd = np.sqrt(max(fwhm**2 - (SENSOR_MESAFESI * 0.5)**2, fwhm**2 * 0.1))
                df2 = max(0.0, fd / 2.0 - YUKSEKLIK_SABITI)
                if 0.01 < df2 < MAX_DEPTH:
                    sonuclar.append((df2, min(0.40, 0.10 + 0.30 * (len(idx) / len(profil)))))
                    yontemler.append(f"FWHM={df2:.2f}m")

        if tv > 1e-6 and len(grad) > 4:
            lo, hi = max(0, ti - len(profil) // 6), min(len(profil), ti + len(profil) // 6 + 1)
            Bw, dw = profil[lo:hi], grad[lo:hi]
            mk = np.abs(dw) > tv * 0.05
            if mk.sum() >= 3:
                oran = np.abs(Bw[mk]) / (np.abs(dw[mk]) + 1e-9)
                de = max(0.0, np.median(oran) * 1.5 - YUKSEKLIK_SABITI)
                if 0.01 < de < MAX_DEPTH:
                    sonuclar.append((de, min(0.25, 0.05 + 0.20 * (1.0 - min(np.std(oran) / (np.mean(oran) + 0.01), 1.0)))))
                    yontemler.append(f"Euler={de:.2f}m")

        if not sonuclar: return _derinlik_yedek(tv), None, "Yeterli profil yok"
        tw = sum(w for _, w in sonuclar)
        df = max(0.0, min(sum(d * w for d, w in sonuclar) / tw, MAX_DEPTH))
        guven = int(( (1.0 - min(np.std([d for d,_ in sonuclar])/(np.mean([d for d,_ in sonuclar])+0.01), 1.0))*0.7 + min(snr/15.0, 1.0)*0.3 )*100) if len(sonuclar)>=2 else max(20, int(min(snr/20.0, 1.0)*50))
        return df, guven, " | ".join(yontemler)
    except: return _derinlik_yedek(np.max(np.abs(profil)) if len(profil)>0 else 0.0), None, "Hata"

def _dipol_fit(profil, eks, gurultu_std):
    try:
        if len(profil) < 5: return None, None, None, None, None, None, "Profil yetersiz"
        x = eks - np.mean(eks)
        amp = float(profil[np.argmax(np.abs(profil))])
        if abs(amp) < 1e-9: return None, None, None, None, None, None, "Sinyal yok"

        def metal_m(x, M, z, x0):
            xc = x - x0; z2 = max(abs(z), 0.01)**2
            return M * (2 * z2 - xc**2) / (xc**2 + z2)**2.5
        def bosluk_m(x, K, z, x0):
            xc = x - x0; z2 = max(abs(z), 0.01)**2
            return -abs(K) * abs(z) / (xc**2 + z2)**1.5
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
        if not metal_ok and not bosluk_ok:
            return (zm or zb or 0.0), max(r2m, r2b), r2m, r2b, (fitm if fitm is not None else fitb), 'belirsiz', "Gurultu/zemin"

        tip = 'metal' if (metal_ok and (not bosluk_ok or r2m >= r2b)) else 'bosluk'
        if tip == 'metal':
            return zm, r2m, r2m, r2b, fitm, 'metal', ("Guclu metal dipol" if r2m >= 0.85 else "Metal kutle orta guven" if r2m >= 0.60 else "Zayif metal sinyali")
        else:
            return zb, r2b, r2m, r2b, fitb, 'bosluk', ("Guclu bosluk/tunel" if r2b >= 0.75 else "Olasi bosluk orta guven" if r2b >= 0.50 else "Zayif bosluk sinyali")
    except: return None, None, None, None, None, None, "Hesaplanamadi"

def _teshis(x_prof, val, tip, r2_m, r2_b, ortusme, fwhm, esik, mod):
    if abs(val) < esik: return "TEMIZ/SINYAL YOK", "#FFFFFF", "Anomali yok."
    if mod == 'Analitik': return "ENERJI MERKEZI", "#FF00FF", "Hedefin odak noktasi."
    if mod == 'Gradient': return "KENAR/SINIR", "#FFA500", "Anomali siniri."

    vmax, vmin = float(np.max(x_prof)), float(np.min(x_prof))
    vr = max(vmax - vmin, 1e-5)
    if abs(x_prof[0]) > abs(val) * 0.88 or abs(x_prof[-1]) > abs(val) * 0.88:
        return "KENAR/DEGERLI?", "#FFA500", "Sinirda kesilmis — alani buyut!"

    mo = bo = blo = 0.0
    if vmax > vr * 0.15 and vmin < -vr * 0.15:
        if abs(vmin) > vmax: mo += 0.25
        else: bo += 0.15; mo += 0.10
    elif vmin < -vr * 0.15: bo += 0.25
    elif vmax > vr * 0.15: mo += 0.20; blo += 0.05
    else: blo += 0.25

    r2m_v, r2b_v = (r2_m or 0.0), (r2_b or 0.0)
    mo += 0.40 * r2m_v
    bo += 0.40 * r2b_v
    blo += 0.10 * max(0.0, 0.35 - max(r2m_v, r2b_v))

    if ortusme is not None:
        if ortusme > 0.80: bo += 0.20
        elif ortusme > 0.50: blo += 0.20
        else: mo += 0.20

    if fwhm is not None:
        if fwhm < 0.3: mo += 0.15
        elif fwhm < 0.8: mo += 0.08; blo += 0.07
        else: bo += 0.10; blo += 0.05

    total = mo + bo + blo + 1e-9
    pm, pb, pbl = mo/total, bo/total, blo/total
    if abs(pm - pb) < 0.15 and max(pm, pb) > 0.30:
        return "KARMA/CELISKILI", "#FFD700", f"Metal%{int(pm*100)} Bosluk%{int(pb*100)} — coklu tarama"

    kaz = max([('metal', pm), ('bosluk', pb), ('belirsiz', pbl)], key=lambda x: x[1])
    if kaz[0] == 'metal':
        return ("METAL KUTLE", "#FF4500", f"Guclu metal (%{int(pm*100)})") if pm > 0.70 else ("MUHTEMEL METAL", "#FF8C00", f"Metal ihtimali %{int(pm*100)}")
    elif kaz[0] == 'bosluk':
        return ("BOSLUK/TUNEL", "#00FFFF", f"Guclu bosluk (%{int(pb*100)})") if pb > 0.70 else ("MUHTEMEL BOSLUK", "#88AAFF", f"Bosluk ihtimali %{int(pb*100)}")
    return "BELIRSIZ", "#888888", "Yetersiz sinyal — sigma/esik dus"

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="C3 Gradiometre Analiz Motoru", layout="wide")
st.markdown("<h2 style='color:#00FF9D; text-align:center;'>C3 GRADIOMETRE ANALIZ MOTORU</h2>", unsafe_allow_html=True)

# Sol Kontrol Paneli
with st.sidebar:
    st.header("⚙️ Kontrol Parametreleri")
    uploaded_file = st.file_uploader("CSV Dosyası Yükleyin", type=["csv"])
    
    adim_mesafesi = st.number_input("Adım Mesafesi (cm)", min_value=1.0, max_value=200.0, value=50.0) / 100.0
    mod = st.radio("Görüntüleme Modu", ["TFA Farkı", "Sadece Z", "Gradient", "Analitik", "FFT Derin", "FFT Sig"])
    
    st.subheader("🎛️ Filtre Ayarları")
    gain = st.slider("Sinyal Kazancı (Gain)", 0.1, 50.0, 1.0, 0.1)
    noise_filter = st.slider("Median Filtre (Gürültü Bastırma)", 1, 9, 1, 2)
    blur = st.slider("Gaussian Blur (Pürüzsüzleştirme)", 0.0, 3.0, 0.0, 0.1)
    sigma_esik = st.slider("Oto Eşik Çarpanı (Sigma)", 0.0, 4.0, 1.0, 0.1)
    manuel_esik = st.number_input("Manuel Eşik (nT)", min_value=0.0, value=0.0)

if uploaded_file is not None:
    # Veri Ön İşleme
    df = pd.read_csv(uploaded_file)
    df.columns = [_normalize_col(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    df['tfa1'] = np.sqrt(df['s1_x']**2 + df['s1_y']**2 + df['s1_z']**2)
    df['tfa2'] = np.sqrt(df['s2_x']**2 + df['s2_y']**2 + df['s2_z']**2)

    for col in ['tfa1', 'tfa2', 's1_z', 's2_z']:
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo, hi)

    df['tfa_diff'] = df['tfa1'] - df['tfa2']
    df['z_diff']   = df['s1_z'] - df['s2_z']

    gurultu_std = df['tfa_diff'].std()

    for col in ['tfa_diff', 'z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir'] == r
            if m.sum() > 4:
                df.loc[m, col] = detrend(df.loc[m, col], type='linear')
        df[col] -= df[col].median()

    df['sutun_m'] = (df['sutun'] - df['sutun'].min()) * adim_mesafesi
    df['satir_m'] = (df['satir'] - df['satir'].min()) * adim_mesafesi

    n_satir, n_sutun = df['satir'].nunique(), df['sutun'].nunique()
    sutun_min, sutun_max = df['sutun'].min(), df['sutun'].max()
    satir_min, satir_max = df['satir'].min(), df['satir'].max()

    grid_res = max(20, min(DEFAULT_GRID_RES, max(n_satir, n_sutun) * 4))

    # Grid Oluşturma
    vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'
    xi = np.linspace(df['sutun_m'].min(), df['sutun_m'].max(), grid_res)
    yi = np.linspace(df['satir_m'].min(), df['satir_m'].max(), grid_res)
    grid_X, grid_Y = np.meshgrid(xi, yi)

    zi_raw = griddata((df['sutun_m'], df['satir_m']), df[vcol], (grid_X, grid_Y), method='linear', fill_value=0)

    # Filtreleme Uygulaması
    zi = zi_raw * gain
    if noise_filter > 1:
        zi = median_filter(zi, size=noise_filter)
    if blur > 0:
        zi = gaussian_filter(zi, sigma=blur)

    esik = max((gurultu_std * sigma_esik * gain), manuel_esik)
    zi = np.where(np.abs(zi) < esik, 0, zi)

    if mod == 'Gradient':    zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
    elif mod == 'Analitik':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
    elif mod == 'FFT Derin': zi = _fft_filtre(zi, 'derin')
    elif mod == 'FFT Sig':   zi = _fft_filtre(zi, 'sig')

    # Hedef Tespiti
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
        st.subheader("🗺️ 2D Isı Haritası")
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

        im = ax2d.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin='lower', cmap=C3_CMAP, norm=norm, aspect='equal', interpolation='bilinear')
        
        for t in targets:
            ax2d.plot(t['x'], t['y'], 'w+', ms=12, mew=2)
            ax2d.text(t['x'] + .02, t['y'] + .02, f"H{t['id']}", color='white', fontsize=9, weight='bold')

        # Başlangıç İşareti (*)
        x0, y0 = df['sutun_m'].min(), df['satir_m'].min()
        ax2d.plot(x0, y0, '*', color='yellow', ms=12, zorder=5)

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
        ax2d.set_title(f"C3 — {mod} | {len(targets)} Hedep Tespiti", color=title_color, fontsize=10)
        st.pyplot(fig)

    with col2:
        st.subheader("🎯 Hedef Analiz ve Teşhis Paneli")
        if targets:
            selected_target_id = st.selectbox("Detaylı İncelemek İstediğiniz Hedefi Seçin", [f"Hedef H{t['id']}" for t in targets])
            sel_target = next(t for t in targets if f"Hedef H{t['id']}" == selected_target_id)
            
            # Seçilen hedefin koordinatlarından kesit çıkarma
            ri, ci = sel_target['py'], sel_target['px']
            xp = zi_raw[ri, :] * gain
            yp = zi_raw[:, ci] * gain
            val = float(zi[ri, ci]) if abs(float(zi[ri, ci])) > 0 else float(xp[ci])

            # Profil Analitik Hesaplamaları
            xn = zi_raw[ri, :]
            depth, guven, ystr = _derinlik_pro(xn, xi, gurultu_std)
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
            st.text(f"Yöntem İzleri: {ystr}")

            st.markdown("### 🔍 Model Eşleştirme Verileri")
            st.write(f"**Sinyal Genliği:** {val:.1f} nT")
            st.write(f"**Dipol Uyumu (R²):** %{r2s*100:.1f} ({fit_yorum})")
            st.write(f"**Faz / Tepe Yapısı:** {faz_yorumu} | {form_yorumu}")

            st.markdown("### 🏺 Yapay Zeka Obje Tahminleri")
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
            st.info("Filtre değerlerine göre haritada belirgin bir anomali odağı bulunamadı. Eşik çarpanını veya kazancı değiştirmeyi deneyin.")
else:
    st.warning("Lütfen analiz başlatmak için sol panelden gradiometre verilerini içeren bir CSV dosyası yükleyin.")
