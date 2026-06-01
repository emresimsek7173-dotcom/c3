"""
C3 MANYETİK GRADİOMETRE — Streamlit Arayüzü
Tüm gelişmiş özellikler: Dipol Fit, FFT, Obje Tahmini, Derinlik Pro, Faz Kayması, Rapor
"""

import unicodedata
import io
import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
SENSOR_MESAFESI  = 0.80
YUKSEKLIK_SABITI = 0.05
MAX_DEPTH        = 5.0
DEFAULT_GRID_RES = 120
ADIM_MESAFESI    = 0.50

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA', '#0066FF', '#00CCFF', '#00CC44',
    '#FFFF00', '#FF6600', '#CC0000'], N=512)

# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI — Türkçe karakter normalizasyonu
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_col(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return s.strip().lower().replace(' ', '_')

# ─────────────────────────────────────────────────────────────────────────────
#  OBJE TAHMİN MOTORU
# ─────────────────────────────────────────────────────────────────────────────
def obje_tahmini(tip, r2_m, r2_b, fwhm, derinlik, val, snr, mod):
    tahminler = []
    d  = derinlik or 0.0
    fw = fwhm or 0.5
    am = abs(val)

    if mod in ('Gradient', 'Analitik'):
        tahminler.append(("Kenar/Sinir anomalisi — detay modda incele", 0.50, '#FFA500'))
        return tahminler[:3]

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

    if tip == 'belirsiz' or (r2_m > 0.30 and r2_b > 0.30 and abs(r2_m - r2_b) < 0.20):
        if fw < 0.6 and d < 1.0:
            tahminler.append(("Çömlek (metal içerikli?)", 0.50, '#FFD700'))
            tahminler.append(("Pişmiş kil / karma malzeme", 0.45, '#FFA500'))
        else:
            tahminler.append(("Karma yapı / belirsiz", 0.30, '#888888'))

    if snr < 2.0 or am < 0.5:
        tahminler.append(("Zemin gürültüsü / mineral iz", 0.40, '#666666'))
        tahminler.append(("Manyetik kaya / doğal anomali", 0.35, '#556655'))

    if not tahminler:
        return [("Tanımlanamadı", 0.0, '#555555')]

    tahminler.sort(key=lambda x: x[1], reverse=True)
    return tahminler[:3]

# ─────────────────────────────────────────────────────────────────────────────
#  VERİ İŞLEME FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────
def veri_yukle(uploaded_file, adim_m):
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

    for col in ['tfa_diff', 'z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir'] == r
            if m.sum() > 4:
                df.loc[m, col] = detrend(df.loc[m, col], type='linear')
        df[col] -= df[col].median()

    df['sutun_m'] = (df['sutun'] - df['sutun'].min()) * adim_m
    df['satir_m'] = (df['satir'] - df['satir'].min()) * adim_m

    meta = {
        'n_satir':    int(df['satir'].nunique()),
        'n_sutun':    int(df['sutun'].nunique()),
        'sutun_min':  int(df['sutun'].min()),
        'satir_min':  int(df['satir'].min()),
        'sutun_max':  int(df['sutun'].max()),
        'satir_max':  int(df['satir'].max()),
        'gurultu_std': df['tfa_diff'].std(),
        'adim_m':     adim_m,
    }
    grid_res = max(20, min(DEFAULT_GRID_RES, max(meta['n_satir'], meta['n_sutun']) * 4))
    meta['grid_res'] = grid_res
    return df, meta


def grid_olustur(df, veri_col, grid_res):
    xi = np.linspace(df['sutun_m'].min(), df['sutun_m'].max(), grid_res)
    yi = np.linspace(df['satir_m'].min(), df['satir_m'].max(), grid_res)
    grid_X, grid_Y = np.meshgrid(xi, yi)
    zi = griddata(
        (df['sutun_m'], df['satir_m']),
        df[veri_col],
        (grid_X, grid_Y),
        method='linear', fill_value=0)
    return xi, yi, grid_X, grid_Y, zi


def fft_filtre(zi, mod='derin'):
    F    = np.fft.fftshift(np.fft.fft2(zi))
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

    filtre = han(R, 0.08) if mod == 'derin' else 1.0 - han(R, 0.10)
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * filtre)))


def filtrele(zi, gain, med_size, blur, sigma_esik, esik_manual, gurultu_std, mod):
    zi = zi * gain
    if med_size > 1:
        size = med_size if med_size % 2 else med_size + 1
        zi = median_filter(zi, size=size)
    if blur > 0:
        zi = gaussian_filter(zi, sigma=blur)
    oto  = gurultu_std * sigma_esik * gain
    esik = max(oto, esik_manual)
    zi   = np.where(np.abs(zi) < esik, 0, zi)
    if   mod == 'Gradient':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
    elif mod == 'Analitik':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
    elif mod == 'FFT Derin': zi = fft_filtre(zi, 'derin')
    elif mod == 'FFT Sig':   zi = fft_filtre(zi, 'sig')
    return zi, esik


def hedef_tespit(zi, xi, yi, filtre_esik, gurultu_std, gain):
    hedef_esik = max(filtre_esik * 0.60, gurultu_std * 0.8 * gain)
    binary = np.abs(zi) > hedef_esik
    labeled, num = label(binary)
    rows, cols = zi.shape
    targets = []
    for i in range(1, num + 1):
        mask = labeled == i
        if mask.sum() < 2:
            continue
        coords = np.argwhere(mask)
        peak   = np.argmax(np.abs(zi[mask]))
        py, px = coords[peak]
        py, px = min(py, rows - 1), min(px, cols - 1)
        targets.append({'id': i, 'x': xi[px], 'y': yi[py], 'amp': zi[py, px]})
    return sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:8]


def faz_kaymasi(x_prof, y_prof):
    try:
        neg = x_prof < 0
        if neg.sum() < 2:
            return 0.0, "Negatif bölge yok"
        neg_idx = np.where(neg)[0]
        cukur   = int(np.mean(neg_idx))
        ana     = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tepe    = int(np.argmax(ana))
        mesafe  = abs(tepe - cukur)
        ortusme = max(0.0, 1.0 - mesafe / (len(x_prof) * 0.3))
        yorum = ("Manyetik olmayan / boşluk" if ortusme > 0.80 else
                 "Karışık sinyal"             if ortusme > 0.50 else
                 "Demir dipol")
        return ortusme, yorum
    except Exception:
        return 0.0, "Hesaplanamadı"


def tepe_sivrilik(x_prof, xi):
    try:
        ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tv  = ana.max()
        if tv < 1e-6:
            return None, "Tepe yok"
        idx = np.where(ana >= tv * 0.5)[0]
        if len(idx) < 2:
            return None, "Tepe dar"
        adim = (xi[-1] - xi[0]) / max(len(xi) - 1, 1)
        fwhm = (idx[-1] - idx[0]) * adim
        form = ("Sivri → küçük yoğun" if fwhm < 0.3 else
                "Orta → hacimli kütle" if fwhm < 0.8 else
                "Yayvan → büyük yapı")
        return fwhm, form
    except Exception:
        return None, "Hesaplanamadı"


def derinlik_simple(peak_nt):
    if abs(peak_nt) < 1e-9:
        return 0.0
    w_half = SENSOR_MESAFESI * 0.5
    return min(max(0.0, w_half - YUKSEKLIK_SABITI), MAX_DEPTH)


def derinlik_pro(profil, eks, gurultu_std):
    try:
        if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01:
            return derinlik_simple(np.max(np.abs(profil))), None, "Profil yetersiz"
        adim     = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
        sonuclar = []
        yontemler= []
        abs_p    = np.abs(profil)
        tv       = abs_p.max()
        ti       = int(np.argmax(abs_p))
        snr      = tv / (gurultu_std + 1e-9)

        # P½ yöntemi
        idx_half = np.where(abs_p >= tv * 0.5)[0]
        if len(idx_half) >= 2:
            w_half = (idx_half[-1] - idx_half[0]) * adim
            if w_half > 1e-6:
                dp = max(0.0, w_half * 0.5 - YUKSEKLIK_SABITI)
                dp = min(dp, MAX_DEPTH)
                w  = min(snr / 10.0, 1.0) * 0.4 + 0.2
                sonuclar.append((dp, w))
                yontemler.append(f"P½={dp:.2f}m")

        # Eğim yöntemi
        try:
            grad = np.gradient(abs_p, adim)
            mg   = np.max(np.abs(grad))
            if mg > 1e-9 and tv > 1e-9:
                dg  = max(0.0, min(tv / (2 * mg + 1e-9) - YUKSEKLIK_SABITI, MAX_DEPTH))
                w   = min(snr / 12.0, 1.0) * 0.35 + 0.15
                sonuclar.append((dg, w))
                yontemler.append(f"Egim={dg:.2f}m")
        except Exception:
            pass

        # Euler dekonvolüsyon
        try:
            x   = eks - eks[np.argmax(abs_p)]
            g   = np.gradient(abs_p, adim)
            m   = abs_p > tv * 0.15
            if m.sum() >= 4:
                A = np.column_stack([x[m], np.ones(m.sum())])
                b = -g[m] * x[m] + abs_p[m]
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                de = max(0.0, min(abs(sol[0]) - YUKSEKLIK_SABITI, MAX_DEPTH))
                w  = min(snr / 15.0, 1.0) * 0.25 + 0.10
                sonuclar.append((de, w))
                yontemler.append(f"Euler={de:.2f}m")
        except Exception:
            pass

        if not sonuclar:
            return derinlik_simple(tv), None, "Yeterli profil yok"

        tw  = sum(w for _, w in sonuclar)
        df_ = sum(d * w for d, w in sonuclar) / tw
        df_ = max(0.0, min(df_, MAX_DEPTH))

        if len(sonuclar) >= 2:
            vals  = [d for d, _ in sonuclar]
            tut   = 1.0 - min(np.std(vals) / (np.mean(vals) + 0.01), 1.0)
            snrs  = min(snr / 15.0, 1.0)
            guven = int((tut * 0.7 + snrs * 0.3) * 100)
        else:
            guven = max(20, int(min(snr / 20.0, 1.0) * 50))

        return df_, guven, " | ".join(yontemler)
    except Exception:
        peak_val = np.max(np.abs(profil)) if len(profil) > 0 else 0.0
        return derinlik_simple(peak_val), None, "Hata"


def dipol_fit(profil, eks):
    try:
        if len(profil) < 5:
            return None, None, None, None, None, None, "Profil yetersiz"
        x   = eks - np.mean(eks)
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
            return max(0.0, 1.0 - np.sum((g - t)**2) / (np.sum((g - np.mean(g))**2) + 1e-12))

        r2m, zm, Mm, fitm = 0.0, None, None, None
        try:
            po, _ = curve_fit(metal_m, x, profil,
                              p0=[amp * 0.1, 0.3, 0.0],
                              bounds=([-abs(amp) * 10, 0.01, -2.0],
                                      [ abs(amp) * 10, MAX_DEPTH,  2.0]),
                              maxfev=2000, ftol=1e-6)
            Mm, zfm, _ = po
            fitm = metal_m(x, *po)
            r2m  = r2(profil, fitm)
            zm   = max(0.0, abs(zfm) - YUKSEKLIK_SABITI)
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
            r2b  = r2(profil, fitb)
            zb   = max(0.0, abs(po[1]) - YUKSEKLIK_SABITI)
        except Exception:
            pass

        if Mm is not None and Mm < 0 and r2m > 0.60:
            if fitb is None:
                fitb, r2b, zb = fitm, r2m, zm
            r2m, zm, Mm = 0.0, None, None

        metal_ok  = r2m >= 0.35 and Mm is not None and Mm > 0
        bosluk_ok = r2b >= 0.35

        if not metal_ok and not bosluk_ok:
            bf = fitm if fitm is not None else fitb
            return (zm or zb or 0.0), max(r2m, r2b), r2m, r2b, bf, 'belirsiz', "Gürültü/zemin"

        tip = ('metal' if (metal_ok and (not bosluk_ok or r2m >= r2b)) else 'bosluk')

        if tip == 'metal':
            zs, r2s, fs = zm, r2m, fitm
            yorum = ("Güçlü metal dipol"     if r2s >= 0.85 else
                     "Metal kütle orta güven" if r2s >= 0.60 else
                     "Zayıf metal sinyali")
        else:
            zs  = zb if zb is not None else zm
            r2s = r2b if bosluk_ok else r2m
            fs  = fitb if fitb is not None else fitm
            yorum = ("Güçlü boşluk/tünel"     if r2s >= 0.75 else
                     "Olası boşluk orta güven" if r2s >= 0.50 else
                     "Zayıf boşluk sinyali")

        return (zs or 0.0), r2s, r2m, r2b, fs, tip, yorum
    except Exception:
        return None, None, None, None, None, None, "Hesaplanamadı"


def teshis(x_prof, val, tip, r2_m, r2_b, ortusme, fwhm, esik_manual, mod):
    if abs(val) < esik_manual:
        return "TEMİZ / SİNYAL YOK", "#FFFFFF", "Anomali yok."
    if mod == 'Analitik':
        return "ENERJİ MERKEZİ", "#FF00FF", "Hedefin odak noktası."
    if mod == 'Gradient':
        return "KENAR / SINIR", "#FFA500", "Anomali sınırı."

    vmax = float(np.max(x_prof))
    vmin = float(np.min(x_prof))
    vr   = max(vmax - vmin, 1e-5)

    if abs(x_prof[0]) > abs(val) * 0.88 or abs(x_prof[-1]) > abs(val) * 0.88:
        return "KENAR / DEĞERLİ?", "#FFA500", "Sınırda kesilmiş — alanı büyüt!"

    mo = bo = blo = 0.0
    hp = vmax >  vr * 0.15
    ht = vmin < -vr * 0.15
    if hp and ht:
        if abs(vmin) > vmax: mo  += 0.25
        else:                bo  += 0.15; mo += 0.10
    elif ht:  bo  += 0.25
    elif hp:  mo  += 0.20; blo += 0.05
    else:     blo += 0.25

    r2m_v = r2_m if r2_m is not None else 0.0
    r2b_v = r2_b if r2_b is not None else 0.0
    mo  += 0.40 * r2m_v
    bo  += 0.40 * r2b_v
    blo += 0.10 * max(0.0, 0.35 - max(r2m_v, r2b_v))

    if ortusme is not None:
        if   ortusme > 0.80: bo  += 0.20
        elif ortusme > 0.50: blo += 0.20
        else:                mo  += 0.20

    if fwhm is not None:
        if   fwhm < 0.3: mo  += 0.15
        elif fwhm < 0.8: mo  += 0.08; blo += 0.07
        else:            bo  += 0.10; blo += 0.05

    total = mo + bo + blo + 1e-9
    pm = mo / total; pb = bo / total; pbl = blo / total

    celisik = abs(pm - pb) < 0.15 and max(pm, pb) > 0.30
    if celisik:
        return ("KARMA / ÇELİŞKİLİ", "#FFD700",
                f"Metal%{int(pm*100)} Boşluk%{int(pb*100)} — çoklu tarama")

    kaz = max([('metal', pm), ('bosluk', pb), ('belirsiz', pbl)], key=lambda x: x[1])

    if kaz[0] == 'metal':
        if pm > 0.70: return ("METAL KÜTLE",   "#FF4500", f"Güçlü metal (%{int(pm*100)})")
        else:         return ("MUHTEMEL METAL", "#FF8C00", f"Metal ihtimali %{int(pm*100)}")
    elif kaz[0] == 'bosluk':
        if pb > 0.70: return ("BOŞLUK / TÜNEL",   "#00FFFF", f"Güçlü boşluk (%{int(pb*100)})")
        else:         return ("MUHTEMEL BOŞLUK", "#88AAFF", f"Boşluk ihtimali %{int(pb*100)}")
    else:
        return ("BELİRSİZ", "#888888", "Yetersiz sinyal — sigma/eşik düş")

# ─────────────────────────────────────────────────────────────────────────────
#  GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────────
def harita_ciz(zi, xi, yi, targets, meta, mod, click_x, click_y):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0d0d0d')

    zmin, zmax = zi.min(), zi.max()
    if zmin < 0 < zmax:
        nz  = zi[zi != 0]
        ph  = np.percentile(np.abs(nz), 98) if len(nz) > 0 else max(abs(zmin), abs(zmax))
        sym = max(ph, 0.001)
        norm = TwoSlopeNorm(vmin=-sym, vcenter=0, vmax=sym)
    else:
        norm = Normalize(vmin=zmin, vmax=zmax)

    im = ax.imshow(zi,
                   extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                   origin='lower', cmap=C3_CMAP, norm=norm,
                   aspect='equal', interpolation='bilinear')
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('nT', color='#aaa', fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

    for t in targets:
        ax.plot(t['x'], t['y'], 'w+', ms=14, mew=2.5)
        ax.text(t['x'] + 0.02, t['y'] + 0.02, f"H{t['id']}",
                color='white', fontsize=9, weight='bold')

    x0 = meta['adim_m'] * 0
    y0 = meta['adim_m'] * 0
    ax.plot(x0, y0, '*', color='yellow', ms=14, zorder=5)
    ax.text(x0, y0, ' ★BAŞLANGIÇ', color='yellow', fontsize=8, weight='bold')

    if click_x is not None and click_y is not None:
        ax.axvline(click_x, color='#FF4500', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(click_y, color='#FF4500', lw=0.8, ls='--', alpha=0.7)
        ax.plot(click_x, click_y, 'o', color='#FF4500', ms=8, zorder=10)

    adim = meta['adim_m']
    sutun_sayilari  = list(range(meta['sutun_min'], meta['sutun_max'] + 1))
    sutun_metreleri = [(s - meta['sutun_min']) * adim for s in sutun_sayilari]
    satir_sayilari  = list(range(meta['satir_min'], meta['satir_max'] + 1))
    satir_metreleri = [(s - meta['satir_min']) * adim for s in satir_sayilari]

    ax.set_xticks(sutun_metreleri)
    ax.set_xticklabels([f"S{s}" for s in sutun_sayilari], fontsize=7, color='#aaa')
    ax.set_yticks(satir_metreleri)
    ax.set_yticklabels([f"R{s}" for s in satir_sayilari], fontsize=7, color='#aaa')
    ax.set_xlabel(f"SÜTUN  ({meta['n_sutun']} × {adim*100:.0f}cm = {(meta['n_sutun']-1)*adim:.1f}m)",
                  fontsize=8, color='#aaa')
    ax.set_ylabel(f"SATIR  ({meta['n_satir']} × {adim*100:.0f}cm = {(meta['n_satir']-1)*adim:.1f}m)",
                  fontsize=8, color='#aaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.tick_params(colors='#555')

    br = '#00FF9D' if targets else '#FF4444'
    bl = f"C3 — {mod} | " + (f"{len(targets)} Hedef" if targets else "⚠ ANOMALİ YOK")
    ax.set_title(bl, color=br, fontsize=11, pad=8)

    plt.tight_layout()
    return fig


def profil_ciz(zi, zi_raw, xi, yi, xv, yv, gain, gurultu_std):
    ci = np.argmin(np.abs(xi - xv))
    ri = np.argmin(np.abs(yi - yv))

    xp  = zi_raw[ri, :] * gain
    yp  = zi_raw[:, ci] * gain
    val = float(zi[ri, ci])
    if abs(val) < abs(xp[ci]) * 0.1:
        val = float(xp[ci])
    xn = zi_raw[ri, :]

    # Hesaplamalar
    depth, guven, ystr          = derinlik_pro(xn, xi, gurultu_std)
    ortusme, faz_y              = faz_kaymasi(xp, yp)
    fwhm, siv_y                 = tepe_sivrilik(xp, xi)
    z_d, r2s, r2m, r2b, fp, tip, dyorum = dipol_fit(xn, xi)
    snr = abs(val) / (gurultu_std * gain + 1e-9)

    # Derinliği dipol ile güçlendir
    depth_val = float(depth) if depth is not None else 0.0
    if z_d is not None and r2s is not None and tip == 'metal' and r2s >= 0.60:
        wd = r2s * 0.5
        if depth is not None and guven is not None:
            depth_val = (depth_val * 0.6 + z_d * wd) / (0.6 + wd)
            guven = min(100, int(guven * 0.7 + r2s * 30))
            ystr = (ystr or "") + f" | Dipol={z_d:.2f}m"
        else:
            depth_val, guven = z_d, int(r2s * 80)
            ystr = f"Dipol={z_d:.2f}m"
    elif z_d is not None and r2s is not None and tip == 'bosluk' and r2s >= 0.50:
        ystr = (ystr or "") + f" | Boşluk={z_d:.2f}m"

    # X profil figürü
    plt.style.use('dark_background')
    fig_xp, ax_xp = plt.subplots(figsize=(6, 2.2))
    fig_xp.patch.set_facecolor('#0a0a0f')
    ax_xp.set_facecolor('#0d0d0d')
    ax_xp.plot(xi, xp, color='#FFD700', lw=1.4, label='Veri')
    if fp is not None and r2s is not None:
        tip_renk = ('#00FF9D' if (tip == 'metal' and r2s >= 0.80) else
                    '#00CFFF' if (tip == 'bosluk' and r2s >= 0.75) else
                    '#888888')
        ax_xp.plot(xi, fp * gain, color=tip_renk, lw=1.0, ls='--', alpha=0.85,
                   label=f"{tip.upper() if tip else '?'} R²={r2s:.2f}")
        ax_xp.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#aaa')
    ax_xp.axhline(0, color='#444', lw=0.5)
    ax_xp.axvline(xv, color='#FF4500', lw=0.8, ls='--')
    ax_xp.set_title("X Kesiti  (kesik = dipol fit)", fontsize=8, color='#aaa')
    ax_xp.tick_params(colors='#555', labelsize=7)
    for sp in ax_xp.spines.values(): sp.set_edgecolor('#222')
    plt.tight_layout()

    # Y profil figürü
    fig_yp, ax_yp = plt.subplots(figsize=(6, 2.2))
    fig_yp.patch.set_facecolor('#0a0a0f')
    ax_yp.set_facecolor('#0d0d0d')
    ax_yp.plot(yi, yp, color='#00CFFF', lw=1.4)
    ax_yp.axhline(0, color='#444', lw=0.5)
    ax_yp.axvline(yv, color='#FF4500', lw=0.8, ls='--')
    ax_yp.set_title("Y Kesiti", fontsize=8, color='#aaa')
    ax_yp.tick_params(colors='#555', labelsize=7)
    for sp in ax_yp.spines.values(): sp.set_edgecolor('#222')
    plt.tight_layout()

    return fig_xp, fig_yp, val, depth_val, guven, ystr, ortusme, faz_y, fwhm, siv_y, r2m, r2b, tip, dyorum, snr


def goster_3d(zi, grid_X, grid_Y, xi, yi, targets, gain, gurultu_std):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.set_facecolor('#0a0a0f')
    for p in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#222')
    ax3.tick_params(colors='#777', labelsize=7)

    surf = ax3.plot_surface(grid_X, grid_Y, zi, cmap=C3_CMAP, edgecolor='none', alpha=0.95)
    cb = fig.colorbar(surf, ax=ax3, shrink=0.4, aspect=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')
    cb.set_label('nT', color='#aaa', fontsize=8)
    ax3.set_xlabel('X (m) — Sütun', color='#aaa', fontsize=9, labelpad=8)
    ax3.set_ylabel('Y (m) — Satır', color='#aaa', fontsize=9, labelpad=8)
    ax3.set_zlabel('nT', color='#aaa', fontsize=9, labelpad=8)

    zr  = zi.max() - zi.min()
    ph  = zr * 0.12
    for t in targets:
        try:
            ri2 = np.argmin(np.abs(yi - t['y']))
            ci2 = np.argmin(np.abs(xi - t['x']))
            pt  = zi[ri2, :] / (gain or 1.0)
            _, _, _, _, _, tip2, _ = dipol_fit(pt, xi)
            pr  = {'metal': '#FF4444', 'bosluk': '#00CFFF', 'belirsiz': '#FFD700'}.get(tip2, '#FFD700')
            zt  = zi[ri2, ci2]
            ax3.scatter([t['x']], [t['y']], [zt + ph], color=pr, s=120, marker='^', zorder=10)
            ax3.plot([t['x'], t['x']], [t['y'], t['y']], [zt, zt + ph], color=pr, lw=1.2, alpha=0.8)
            ax3.text(t['x'], t['y'], zt + ph * 1.2, f"H{t['id']}",
                     color='white', fontsize=8, weight='bold', ha='center')
        except Exception:
            pass

    ax3.set_title("C3 3D  (Kırmızı=Metal · Mavi=Boşluk · Sarı=Belirsiz)",
                  color='#00FF9D', pad=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  RAPOR ÜRETİCİ
# ─────────────────────────────────────────────────────────────────────────────
def rapor_uret(meta, mod, targets, ai_sonuc, xi, yi, zi, gain, gurultu_std, dosya_adi):
    now   = datetime.now()
    hlist = []
    lines = []
    lines.append("C3 MANYETİK GRADİOMETRE RAPORU")
    lines.append("=" * 44)
    lines.append(f"Tarih   : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dosya   : {dosya_adi}")
    lines.append(f"Adım    : {meta['adim_m']*100:.0f}cm")
    lines.append(f"Mod     : {mod}")
    lines.append(f"Sensör  : {SENSOR_MESAFESI}m aralık, {YUKSEKLIK_SABITI}m yükseklik")
    lines.append("")
    lines.append("HEDEFLER:")

    for t in targets:
        try:
            ri2 = np.argmin(np.abs(yi - t['y']))
            pt  = zi[ri2, :] / (gain or 1.0)
            d2, gv2, _  = derinlik_pro(pt, xi, gurultu_std)
            _, r2s2, r2m2, r2b2, _, tip2, _ = dipol_fit(pt, xi)
            tah = obje_tahmini(tip2, r2m2 or 0, r2b2 or 0, None, d2,
                               t['amp'],
                               abs(t['amp']) / (gurultu_std * gain + 1e-9), mod)
        except Exception:
            d2 = derinlik_simple(t['amp'] / (gain or 1.0))
            gv2 = r2m2 = r2b2 = tip2 = None
            tah = [("?", 0, "")]

        d2v = float(d2) if d2 is not None else 0.0
        t_sut = round(t['x'] / meta['adim_m']) + meta['sutun_min']
        t_sat = round(t['y'] / meta['adim_m']) + meta['satir_min']
        r2m_str = f"{r2m2:.2f}" if r2m2 is not None else "?"
        r2b_str = f"{r2b2:.2f}" if r2b2 is not None else "?"
        lines.append(f"  H{t['id']}: Sutun{t_sut}/Satir{t_sat} "
                     f"({t['x']:.2f}m,{t['y']:.2f}m) "
                     f"{t['amp']:.1f}nT ~{d2v:.2f}m [{tip2}] "
                     f"R2M={r2m_str} R2B={r2b_str}")
        lines.append(f"  Tahmin: {tah[0][0]}")

        hlist.append({
            'id': t['id'], 'x': round(t['x'], 3), 'y': round(t['y'], 3),
            'amp_nT': round(float(t['amp']), 2), 'derinlik_m': round(d2v, 3),
            'guven_pct': gv2, 'dipol_tip': tip2,
            'r2_metal': round(float(r2m2), 3) if r2m2 else None,
            'r2_bosluk': round(float(r2b2), 3) if r2b2 else None,
            'tahmin_1': tah[0][0],
        })

    if ai_sonuc:
        a = ai_sonuc
        lines.append("")
        lines.append("SEÇİLİ NOKTA:")
        lines.append(f"  Konum: {a.get('x',0):.2f},{a.get('y',0):.2f}  "
                     f"Şiddet: {a.get('val',0):.1f}nT")
        lines.append(f"  Derinlik: ~{a.get('depth',0):.2f}m  Güven: %{a.get('guven','?')}")
        lines.append(f"  Metal R²: {a.get('r2m','?')}  Boşluk R²: {a.get('r2b','?')}")
        lines.append(f"  Teşhis: {a.get('durum','')}  {a.get('aciklama','')}")
        tah0 = a.get('tahminler', [['']])
        lines.append(f"  Tahmin: {tah0[0][0] if tah0 else '?'}")

    lines.append("")
    lines.append("--- Rapor Sonu ---")
    txt = "\n".join(lines)

    # CSV
    csv_buf = io.StringIO()
    if hlist:
        writer = csv.DictWriter(csv_buf, fieldnames=hlist[0].keys())
        writer.writeheader()
        writer.writerows(hlist)

    # JSON
    def sf(v):
        return float(v) if isinstance(v, (np.floating, np.integer)) else v
    json_data = json.dumps({
        'tarih': now.isoformat(), 'dosya': dosya_adi,
        'adim_cm': meta['adim_m'] * 100, 'mod': mod,
        'hedefler': hlist,
        'secili': {k: sf(v) for k, v in ai_sonuc.items() if not isinstance(v, list)} if ai_sonuc else {},
    }, ensure_ascii=False, indent=2)

    return txt, csv_buf.getvalue(), json_data

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLİT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C3 Manyetik Analiz",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body, .stApp { background-color: #0a0a0f; color: #e0e0e0; }
[data-testid="stSidebar"] { background-color: #0d0d1a; }
h1, h2, h3 { color: #00FF9D; font-family: 'Courier New', monospace; }
.metric-card {
    background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
    padding: 12px 16px; margin: 4px 0;
    font-family: 'Courier New', monospace;
}
.badge-metal   { background:#FF450033; border:1px solid #FF4500; border-radius:6px; padding:4px 10px; color:#FF4500; font-weight:bold; }
.badge-bosluk  { background:#00FFFF22; border:1px solid #00FFFF; border-radius:6px; padding:4px 10px; color:#00FFFF; font-weight:bold; }
.badge-temiz   { background:#ffffff11; border:1px solid #888; border-radius:6px; padding:4px 10px; color:#888; font-weight:bold; }
.badge-belirsiz{ background:#FFD70033; border:1px solid #FFD700; border-radius:6px; padding:4px 10px; color:#FFD700; font-weight:bold; }
.stButton > button { background:#1a237e; color:white; border:none; font-family:monospace; }
.stButton > button:hover { background:#283593; }
div[data-testid="stMetricValue"] { color: #00FF9D; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🧲 C3 ANALİZ PRO")
    st.markdown("---")

    uploaded = st.file_uploader("CSV Dosyası Seç", type=['csv'])
    adim_cm  = st.selectbox("Adım Mesafesi (cm)", [20, 25, 50], index=2)
    adim_m   = adim_cm / 100.0

    st.markdown("---")
    st.markdown("### Görüntüleme Modu")
    mod = st.radio("Mod", ['TFA', 'Sadece Z', 'Gradient', 'Analitik', 'FFT Derin', 'FFT Sig'],
                   index=0)

    st.markdown("---")
    st.markdown("### Filtre Ayarları")
    gain      = st.slider("Kazanç",    1, 1000, 100, 5)
    esik      = st.slider("Eşik (nT)", 0, 500,   25, 1)
    blur      = st.slider("Yumuşat",   0.0, 5.0, 0.5, 0.1)
    noise     = st.slider("Parazit",   1, 9, 3, 2)
    sigma_esik= st.slider("σ Eşik",    1, 3, 3, 1)

    st.markdown("---")
    st.markdown("### Nokta Seçimi (m)")
    click_x = st.number_input("X (Sütun m)", value=0.0, step=0.1, format="%.2f")
    click_y = st.number_input("Y (Satır m)",  value=0.0, step=0.1, format="%.2f")

    st.markdown("---")
    show_3d = st.checkbox("3D Görselleştir", value=False)

# ── Ana içerik ──
st.markdown("# C3 MANYETİK GRADİOMETRE ANALİZ SİSTEMİ")

if uploaded is None:
    st.info("📂 Sol panelden CSV dosyası yükleyin. Sensör verileri: satir, sutun, s1_x/y/z, s2_x/y/z")
    st.markdown("""
    **Desteklenen sütunlar:** `satir`, `sutun`, `s1_x`, `s1_y`, `s1_z`, `s2_x`, `s2_y`, `s2_z`
    
    **Gelişmiş özellikler:**
    - Dipol Fit — Metal vs Boşluk ayrımı (R² skoru ile)
    - Çok yöntemli Derinlik tahmini (P½ + Eğim + Euler)
    - FFT Derin / FFT Sig filtreleri
    - Faz kayması analizi
    - Otomatik obje tahmini (Çömlek, Küp, Boru, Tünel…)
    - TXT + CSV + JSON rapor çıktısı
    """)
    st.stop()

# ── Veri yükle (cache) ──
@st.cache_data(show_spinner="Veri işleniyor…")
def yukle(file_bytes, file_name, adim_m):
    return veri_yukle(io.BytesIO(file_bytes), adim_m)

try:
    df, meta = yukle(uploaded.read(), uploaded.name, adim_m)
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

# ── Grid & Filtre ──
vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'
xi, yi, grid_X, grid_Y, zi_raw = grid_olustur(df, vcol, meta['grid_res'])
zi, esik_val = filtrele(zi_raw, gain, noise, blur, sigma_esik, esik, meta['gurultu_std'], mod)
targets = hedef_tespit(zi, xi, yi, esik_val, meta['gurultu_std'], gain)

# ── Profil hesapla ──
cx = float(np.clip(click_x, xi.min(), xi.max()))
cy = float(np.clip(click_y, yi.min(), yi.max()))

fig_xp, fig_yp, val, depth_val, guven, ystr, ortusme, faz_y, fwhm, siv_y, r2m, r2b, tip, dyorum, snr = \
    profil_ciz(zi, zi_raw, xi, yi, cx, cy, gain, meta['gurultu_std'])

durum, renk, aciklama = teshis(
    zi_raw[np.argmin(np.abs(yi - cy)), :] * gain,
    val, tip, r2m, r2b, ortusme, fwhm, esik, mod)

tahminler = obje_tahmini(tip, r2m or 0.0, r2b or 0.0, fwhm, depth_val, val, snr, mod)

ai_sonuc = {
    'durum': durum, 'aciklama': aciklama, 'renk': renk,
    'depth': depth_val, 'val': val, 'x': cx, 'y': cy,
    'guven': guven, 'yontem': ystr,
    'r2m': r2m, 'r2b': r2b, 'dipol_tip': tip, 'dipol_yorum': dyorum,
    'tahminler': tahminler,
}

# ── Üst metrikler ──
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Hedef Sayısı", len(targets), delta="anomali" if targets else None)
with c2:
    st.metric("Gürültü Std", f"{meta['gurultu_std']:.3f} nT")
with c3:
    st.metric("Oto Eşik", f"{meta['gurultu_std']*sigma_esik:.2f} nT")
with c4:
    st.metric("Grid", f"{meta['grid_res']}×{meta['grid_res']}")
with c5:
    st.metric("Alan", f"{(meta['n_sutun']-1)*adim_m:.1f}×{(meta['n_satir']-1)*adim_m:.1f}m")

st.markdown("---")

# ── Ana düzen ──
col_map, col_info = st.columns([3, 2])

with col_map:
    st.markdown("### 🗺 Anomali Haritası")
    fig_map = harita_ciz(zi, xi, yi, targets, meta, mod, cx, cy)
    st.pyplot(fig_map, use_container_width=True)
    plt.close(fig_map)

with col_info:
    st.markdown("### 🔬 Teşhis Paneli")

    # Durum badge
    badge_cls = ('badge-metal' if 'METAL' in durum else
                 'badge-bosluk' if 'BOŞLUK' in durum or 'TÜNEL' in durum else
                 'badge-temiz' if 'TEMİZ' in durum else 'badge-belirsiz')
    st.markdown(f'<span class="{badge_cls}">{durum}</span>', unsafe_allow_html=True)
    st.markdown(f"*{aciklama}*")

    st.markdown("---")
    st.markdown("**📍 Seçili Nokta**")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("X (Sütun)", f"{cx:.2f} m")
        st.metric("Şiddet", f"{val:.1f} nT")
        st.metric("SNR", f"{snr:.1f}×")
    with mc2:
        st.metric("Y (Satır)", f"{cy:.2f} m")
        st.metric("Derinlik ~", f"{depth_val:.2f} m")
        st.metric("Güven", f"%{guven if guven else '?'}")

    st.markdown("---")
    st.markdown("**🔩 Dipol Analizi**")
    d1, d2 = st.columns(2)
    with d1:
        st.metric("Metal R²", f"{r2m:.3f}" if r2m is not None else "—")
    with d2:
        st.metric("Boşluk R²", f"{r2b:.3f}" if r2b is not None else "—")
    if dyorum:
        st.caption(f"Dipol: {dyorum}")
    if ystr:
        st.caption(f"Yöntem: {ystr}")

    if fwhm is not None:
        st.markdown(f"**FWHM:** {fwhm:.3f} m — *{siv_y}*")
    if ortusme > 0:
        st.markdown(f"**Faz örtüşme:** {ortusme:.2f} — *{faz_y}*")

    st.markdown("---")
    st.markdown("**🏺 Obje Tahmini**")
    for isim, puan, renk2 in tahminler:
        bar = "█" * int(puan * 5) + "░" * (5 - int(puan * 5))
        st.markdown(
            f'<div class="metric-card" style="border-color:{renk2}33">'
            f'<span style="color:{renk2}">{bar} %{int(puan*100)}</span><br>'
            f'<span style="font-size:0.85em">{isim}</span></div>',
            unsafe_allow_html=True)

# ── Profil grafikleri ──
st.markdown("---")
st.markdown("### 📈 Kesit Profilleri")
pc1, pc2 = st.columns(2)
with pc1:
    st.pyplot(fig_xp, use_container_width=True)
    plt.close(fig_xp)
with pc2:
    st.pyplot(fig_yp, use_container_width=True)
    plt.close(fig_yp)

# ── Hedef tablosu ──
if targets:
    st.markdown("---")
    st.markdown("### 🎯 Tespit Edilen Hedefler")
    rows = []
    for t in targets:
        try:
            ri2 = np.argmin(np.abs(yi - t['y']))
            pt  = zi[ri2, :] / (gain or 1.0)
            d2, gv2, _ = derinlik_pro(pt, xi, meta['gurultu_std'])
            _, r2s2, r2m2, r2b2, _, tip2, _ = dipol_fit(pt, xi)
            tah = obje_tahmini(tip2, r2m2 or 0, r2b2 or 0, None, d2,
                               t['amp'], abs(t['amp']) / (meta['gurultu_std'] * gain + 1e-9), mod)
        except Exception:
            d2 = derinlik_simple(t['amp'] / (gain or 1.0))
            gv2 = r2m2 = r2b2 = tip2 = None
            tah = [("?", 0, "")]
        t_sut = round(t['x'] / adim_m) + meta['sutun_min']
        t_sat = round(t['y'] / adim_m) + meta['satir_min']
        rows.append({
            'ID': f"H{t['id']}",
            'Konum': f"S{t_sut}/R{t_sat}",
            'X (m)': round(t['x'], 2),
            'Y (m)': round(t['y'], 2),
            'Şiddet (nT)': round(float(t['amp']), 1),
            'Derinlik ~(m)': round(float(d2) if d2 else 0, 2),
            'Güven (%)': gv2 if gv2 else '?',
            'Tip': tip2 or '?',
            'R² Metal': round(float(r2m2), 2) if r2m2 else '—',
            'R² Boşluk': round(float(r2b2), 2) if r2b2 else '—',
            'Tahmin': tah[0][0] if tah else '?',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 column_config={'ID': st.column_config.TextColumn(width='small')})

# ── 3D ──
if show_3d:
    st.markdown("---")
    st.markdown("### 🌐 3D Görselleştirme")
    fig3d = goster_3d(zi, grid_X, grid_Y, xi, yi, targets, gain, meta['gurultu_std'])
    st.pyplot(fig3d, use_container_width=True)
    plt.close(fig3d)

# ── Rapor ──
st.markdown("---")
st.markdown("### 📄 Rapor")
rc1, rc2, rc3 = st.columns(3)
with rc1:
    if st.button("📋 Rapor Oluştur"):
        txt_r, csv_r, json_r = rapor_uret(
            meta, mod, targets, ai_sonuc, xi, yi, zi, gain, meta['gurultu_std'], uploaded.name)
        st.session_state['rapor_txt']  = txt_r
        st.session_state['rapor_csv']  = csv_r
        st.session_state['rapor_json'] = json_r
        st.success("Rapor hazır — indirin!")

if 'rapor_txt' in st.session_state:
    with rc2:
        st.download_button("⬇ TXT İndir", st.session_state['rapor_txt'],
                           file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime='text/plain')
    with rc3:
        st.download_button("⬇ JSON İndir", st.session_state['rapor_json'],
                           file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime='application/json')
    if st.session_state['rapor_csv']:
        st.download_button("⬇ CSV Hedefler", st.session_state['rapor_csv'],
                           file_name=f"C3_Hedefler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime='text/csv')
    with st.expander("Rapor Önizleme"):
        st.text(st.session_state['rapor_txt'])

st.markdown("---")
st.caption(f"C3 ANALİZ PRO — Streamlit | Grid:{meta['grid_res']}px | "
           f"Satır:{meta['n_satir']} × Sütun:{meta['n_sutun']} | "
           f"Adım:{adim_cm}cm")
