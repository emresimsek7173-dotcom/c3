import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from scipy.optimize import curve_fit
import json, csv
from pathlib import Path
from datetime import datetime
from io import StringIO, BytesIO
import streamlit as st

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
    fw = fwhm     or 0.5
    am = abs(val)

    if mod in ('Gradient', 'Analitik'):
        tahminler.append(("Kenar/Sınır anomalisi — detay modda incele", 0.50, '#FFA500'))
        tahminler.sort(key=lambda x: x[1], reverse=True)
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
#  ANALİZ SINIFI (UI bağımsız — saf hesaplama)
# ─────────────────────────────────────────────────────────────────────────────
class C3Analiz:
    def __init__(self, csv_bytes_or_path, adim_m=ADIM_MESAFESI):
        self.adim_m   = adim_m
        self.grid_res = DEFAULT_GRID_RES
        self.zi = self.zi_raw = self.grid_x = self.grid_y = None
        self.xi_arr = self.yi_arr = None
        self.targets  = []
        self.ai_sonuc = {}
        self._veri_yukle(csv_bytes_or_path)

    # ── Veri yükleme ──────────────────────────────────────────────────────────
    def _veri_yukle(self, src):
        if isinstance(src, bytes):
            df = pd.read_csv(BytesIO(src))
        elif isinstance(src, str):
            df = pd.read_csv(src)
        else:
            df = pd.read_csv(src)

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

        df['sutun_m'] = (df['sutun'] - df['sutun'].min()) * self.adim_m
        df['satir_m'] = (df['satir'] - df['satir'].min()) * self.adim_m

        self.df          = df
        self.n_satir     = int(df['satir'].nunique())
        self.n_sutun     = int(df['sutun'].nunique())
        self.sutun_min   = int(df['sutun'].min())
        self.satir_min   = int(df['satir'].min())
        self.sutun_max   = int(df['sutun'].max())
        self.satir_max   = int(df['satir'].max())
        self.gurultu_std = df['tfa_diff'].std()
        self.grid_res    = max(20, min(DEFAULT_GRID_RES, max(self.n_satir, self.n_sutun) * 4))

    # ── Grid ──────────────────────────────────────────────────────────────────
    def _grid_olustur(self, veri_col):
        xi = np.linspace(self.df['sutun_m'].min(), self.df['sutun_m'].max(), self.grid_res)
        yi = np.linspace(self.df['satir_m'].min(), self.df['satir_m'].max(), self.grid_res)
        grid_X, grid_Y = np.meshgrid(xi, yi)
        zi = griddata(
            (self.df['sutun_m'], self.df['satir_m']),
            self.df[veri_col],
            (grid_X, grid_Y),
            method='linear', fill_value=0)
        return xi, yi, grid_X, grid_Y, zi

    # ── FFT filtre ─────────────────────────────────────────────────────────────
    def _fft_filtre(self, zi, mod='derin'):
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

        filtre = han(R, 0.08) if mod == 'derin' else 1.0 - han(R, 0.10)
        return np.real(np.fft.ifft2(np.fft.ifftshift(F * filtre)))

    # ── Filtre ────────────────────────────────────────────────────────────────
    def _filtrele(self, zi, gain, noise_size, blur, sigma_mult, esik_val, mod):
        zi = zi * gain
        med = int(noise_size)
        if med > 1:
            size = med if med % 2 else med + 1
            zi = median_filter(zi, size=size)
        if blur > 0:
            zi = gaussian_filter(zi, sigma=blur)
        oto  = self.gurultu_std * sigma_mult * gain
        esik = max(oto, esik_val)
        zi   = np.where(np.abs(zi) < esik, 0, zi)
        if   mod == 'Gradient':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
        elif mod == 'Analitik':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
        elif mod == 'FFT Derin': zi = self._fft_filtre(zi, 'derin')
        elif mod == 'FFT Sig':   zi = self._fft_filtre(zi, 'sig')
        return zi, esik

    # ── Hedef tespiti ──────────────────────────────────────────────────────────
    def _hedef_tespit(self, zi, xi, yi, filtre_esik):
        hedef_esik = max(filtre_esik * 0.60, self.gurultu_std * 0.8)
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
            targets.append({'id': i, 'x': xi[px], 'y': yi[py], 'amp': zi[py, px]})
        return sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:8]

    # ── Faz kayması ───────────────────────────────────────────────────────────
    def _faz_kaymasi(self, x_prof, y_prof):
        try:
            neg = x_prof < 0
            if neg.sum() < 2:
                return 0.0, "Negatif bölge yok"
            neg_idx = np.where(neg)[0]
            cukur = int(np.mean(neg_idx))
            ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
            tepe = int(np.argmax(ana))
            mesafe = abs(tepe - cukur)
            ortusme = max(0.0, 1.0 - mesafe / (len(x_prof) * 0.3))
            yorum = ("Manyetik olmayan/boşluk" if ortusme > 0.80 else
                     "Karışık sinyal"           if ortusme > 0.50 else
                     "Demir dipol")
            return ortusme, yorum
        except Exception:
            return 0.0, "Hesaplanamadı"

    # ── Tepe sivrilik ─────────────────────────────────────────────────────────
    def _tepe_sivrilik(self, x_prof, xi):
        try:
            ana = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
            tv = ana.max()
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

    # ── Derinlik ─────────────────────────────────────────────────────────────
    def _derinlik(self, peak_nt):
        if abs(peak_nt) < 1e-9:
            return 0.0
        w_half = SENSOR_MESAFESI * 0.5
        return min(max(0.0, w_half - YUKSEKLIK_SABITI), MAX_DEPTH)

    # ── Derinlik pro ─────────────────────────────────────────────────────────
    def _derinlik_pro(self, profil, eks):
        try:
            if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01:
                return self._derinlik(np.max(np.abs(profil))), None, "Profil yetersiz"
            adim = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
            sonuclar, yontemler = [], []
            abs_p = np.abs(profil)
            tv = abs_p.max()
            ti = int(np.argmax(abs_p))
            snr = tv / (self.gurultu_std + 1e-9)

            # W/2 yöntemi
            try:
                half = np.where(abs_p >= tv * 0.5)[0]
                if len(half) >= 2:
                    w_half = (half[-1] - half[0]) * adim * 0.5
                    dw = max(0.0, w_half - YUKSEKLIK_SABITI)
                    w_snr = min(snr / 10.0, 1.0)
                    sonuclar.append((min(dw, MAX_DEPTH), w_snr))
                    yontemler.append(f"W/2={dw:.2f}m")
            except Exception:
                pass

            # Peters yöntemi
            try:
                quarter = np.where(abs_p >= tv * 0.25)[0]
                if len(quarter) >= 2:
                    lp = (quarter[-1] - quarter[0]) * adim
                    dp = lp / 1.6
                    dp = max(0.0, min(dp - YUKSEKLIK_SABITI, MAX_DEPTH))
                    p_snr = min(snr / 15.0, 1.0) * 0.8
                    sonuclar.append((dp, p_snr))
                    yontemler.append(f"Peters={dp:.2f}m")
            except Exception:
                pass

            # Euler yöntemi
            try:
                if len(profil) > 6:
                    grad = np.gradient(profil, adim)
                    si = max(1, len(profil) // 5)
                    ei = len(profil) - si
                    if ei > si + 2:
                        seg_g = grad[si:ei]
                        seg_p = profil[si:ei]
                        seg_x = eks[si:ei] - eks[ti]
                        denom = np.sum(seg_g**2)
                        if denom > 1e-12:
                            de_raw = abs(np.sum(seg_g * (seg_g * seg_x - seg_p)) / denom)
                            de = max(0.0, min(de_raw - YUKSEKLIK_SABITI, MAX_DEPTH))
                            e_snr = min(snr / 20.0, 1.0) * 0.6
                            sonuclar.append((de, e_snr))
                            yontemler.append(f"Euler={de:.2f}m")
            except Exception:
                pass

            if not sonuclar:
                return self._derinlik(tv), None, "Yeterli profil yok"

            tw = sum(w for _, w in sonuclar)
            df_val = sum(d * w for d, w in sonuclar) / tw
            df_val = max(0.0, min(df_val, MAX_DEPTH))

            if len(sonuclar) >= 2:
                vals = [d for d, _ in sonuclar]
                tut = 1.0 - min(np.std(vals) / (np.mean(vals) + 0.01), 1.0)
                snrs = min(snr / 15.0, 1.0)
                guven = int((tut * 0.7 + snrs * 0.3) * 100)
            else:
                guven = max(20, int(min(snr / 20.0, 1.0) * 50))

            return df_val, guven, " | ".join(yontemler)
        except Exception:
            peak_val = np.max(np.abs(profil)) if len(profil) > 0 else 0.0
            return self._derinlik(peak_val), None, "Hata"

    # ── Dipol fit ─────────────────────────────────────────────────────────────
    def _dipol_fit(self, profil, eks):
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
                                  bounds=([-abs(amp)*10, 0.01, -2.0],
                                          [ abs(amp)*10, MAX_DEPTH,  2.0]),
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
                                  p0=[abs(amp)*0.5, 0.3, 0.0],
                                  bounds=([0.0, 0.01, -2.0],
                                          [abs(amp)*20, MAX_DEPTH, 2.0]),
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
                yorum = ("Güçlü metal dipol"      if r2s >= 0.85 else
                         "Metal kütle orta güven"  if r2s >= 0.60 else
                         "Zayıf metal sinyali")
            else:
                zs  = zb if zb is not None else zm
                r2s = r2b if bosluk_ok else r2m
                fs  = fitb if fitb is not None else fitm
                yorum = ("Güçlü boşluk/tünel"      if r2s >= 0.75 else
                         "Olası boşluk orta güven"  if r2s >= 0.50 else
                         "Zayıf boşluk sinyali")

            return (zs or 0.0), r2s, r2m, r2b, fs, tip, yorum
        except Exception:
            return None, None, None, None, None, None, "Hesaplanamadı"

    # ── Teşhis ───────────────────────────────────────────────────────────────
    def _teshis(self, x_prof, val, tip, r2_m, r2_b, ortusme, fwhm, esik_val, mod):
        if abs(val) < esik_val:
            return "TEMİZ/SİNYAL YOK", "#FFFFFF", "Anomali yok."
        if mod == 'Analitik':
            return "ENERJİ MERKEZİ", "#FF00FF", "Hedefin odak noktası."
        if mod == 'Gradient':
            return "KENAR/SINIR", "#FFA500", "Anomali sınırı."

        vmax = float(np.max(x_prof))
        vmin = float(np.min(x_prof))
        vr   = max(vmax - vmin, 1e-5)

        if abs(x_prof[0]) > abs(val) * 0.88 or abs(x_prof[-1]) > abs(val) * 0.88:
            return "KENAR/DEĞERLİ?", "#FFA500", "Sınırda kesilmiş — alanı büyüt!"

        mo = bo = blo = 0.0
        hp = vmax > vr * 0.15
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
            return ("KARMA/ÇELİŞKİLİ", "#FFD700",
                    f"Metal%{int(pm*100)} Boşluk%{int(pb*100)} — çoklu tarama")

        kaz = max([('metal', pm), ('bosluk', pb), ('belirsiz', pbl)], key=lambda x: x[1])

        if kaz[0] == 'metal':
            if pm > 0.70: return ("METAL KÜTLE",    "#FF4500", f"Güçlü metal (%{int(pm*100)})")
            else:         return ("MUHTEMEL METAL",  "#FF8C00", f"Metal ihtimali %{int(pm*100)}")
        elif kaz[0] == 'bosluk':
            if pb > 0.70: return ("BOŞLUK/TÜNEL",   "#00FFFF", f"Güçlü boşluk (%{int(pb*100)})")
            else:         return ("MUHTEMEL BOŞLUK", "#88AAFF", f"Boşluk ihtimali %{int(pb*100)}")
        else:
            return ("BELİRSİZ", "#888888", "Yetersiz sinyal — sigma/eşik düş")

    # ── Tam analiz (parametreler dışarıdan) ──────────────────────────────────
    def analiz_et(self, gain, noise_size, blur, sigma_mult, esik_val, mod, sel_x=None, sel_y=None):
        vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'
        xi, yi, grid_X, grid_Y, zi_raw = self._grid_olustur(vcol)

        self.xi_arr = xi
        self.yi_arr = yi
        self.grid_x = grid_X
        self.grid_y = grid_Y
        self.zi_raw = zi_raw

        zi, esik = self._filtrele(zi_raw, gain, noise_size, blur, sigma_mult, esik_val, mod)
        self.zi = zi

        targets = self._hedef_tespit(zi, xi, yi, esik)
        self.targets = targets

        # Seçili nokta analizi
        ai_sonuc = {}
        if sel_x is not None and sel_y is not None:
            ci = np.argmin(np.abs(xi - sel_x))
            ri = np.argmin(np.abs(yi - sel_y))
            xp = zi_raw[ri, :] * gain
            yp = zi_raw[:, ci] * gain
            val = float(zi[ri, ci])
            if abs(val) < abs(xp[ci]) * 0.1:
                val = float(xp[ci])
            xn = zi_raw[ri, :]
            depth, guven, ystr = self._derinlik_pro(xn, xi)
            ortusme, faz_y     = self._faz_kaymasi(xp, yp)
            fwhm, siv_y        = self._tepe_sivrilik(xp, xi)
            z_d, r2s, r2m, r2b, fp, tip, dyorum = self._dipol_fit(xn, xi)
            snr = abs(val) / (self.gurultu_std * gain + 1e-9)
            durum, renk, aciklama = self._teshis(xp, val, tip, r2m, r2b, ortusme, fwhm, esik_val, mod)

            if z_d is not None and r2s is not None and tip == 'metal' and r2s >= 0.60:
                wd = r2s * 0.5
                if depth is not None and guven is not None:
                    depth = (depth * 0.6 + z_d * wd) / (0.6 + wd)
                    guven = min(100, int(guven * 0.7 + r2s * 30))
                    ystr  = (ystr or "") + f" | Dipol={z_d:.2f}m"
                else:
                    depth, guven = z_d, int(r2s * 80)
                    ystr = f"Dipol={z_d:.2f}m"
            elif z_d is not None and r2s is not None and tip == 'bosluk' and r2s >= 0.50:
                ystr = (ystr or "") + f" | Boşluk={z_d:.2f}m"

            depth_val = float(depth) if depth is not None else 0.0
            tahminler = obje_tahmini(tip, r2m or 0.0, r2b or 0.0, fwhm,
                                     depth_val, val, snr, mod)
            ai_sonuc = {
                'durum': durum, 'aciklama': aciklama, 'renk': renk,
                'depth': depth_val, 'val': val, 'x': sel_x, 'y': sel_y,
                'guven': guven, 'yontem': ystr,
                'r2m': r2m, 'r2b': r2b, 'dipol_tip': tip, 'dipol_yorum': dyorum,
                'tahminler': tahminler,
                'xp': xp, 'yp': yp, 'xi': xi, 'yi': yi, 'fp': fp, 'r2s': r2s,
            }
        self.ai_sonuc = ai_sonuc
        return zi, xi, yi, targets, esik, ai_sonuc

    # ── Rapor oluştur (string) ────────────────────────────────────────────────
    def rapor_olustur(self, mod, dosya_adi=""):
        gain = 100.0  # rapor için varsayılan
        now = datetime.now()
        lines = [
            "C3 MANYETİK GRADİOMETRE RAPORU",
            "=" * 44,
            f"Tarih   : {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dosya   : {dosya_adi}",
            f"Adım    : {self.adim_m*100:.0f}cm",
            f"Mod     : {mod}",
            f"Sensör  : {SENSOR_MESAFESI}m aralık, {YUKSEKLIK_SABITI}m yükseklik",
            "",
            "HEDEFLER:",
        ]
        xi = self.xi_arr
        yi = self.yi_arr
        hlist = []
        for t in (self.targets or []):
            try:
                ri2 = np.argmin(np.abs(yi - t['y']))
                pt  = self.zi[ri2, :] / (gain or 1.0)
                d2, gv2, _ = self._derinlik_pro(pt, xi)
                _, r2s2, r2m2, r2b2, _, tip2, _ = self._dipol_fit(pt, xi)
                tah = obje_tahmini(tip2, r2m2 or 0, r2b2 or 0, None, d2,
                                   t['amp'],
                                   abs(t['amp']) / (self.gurultu_std * gain + 1e-9),
                                   mod)
            except Exception:
                d2 = self._derinlik(t['amp'] / (gain or 1.0))
                gv2 = r2m2 = r2b2 = tip2 = None
                tah = [("?", 0, "")]

            d2v = float(d2) if d2 is not None else 0.0
            t_sut = round(t['x'] / self.adim_m) + self.sutun_min
            t_sat = round(t['y'] / self.adim_m) + self.satir_min
            r2m_s = f"{r2m2:.2f}" if r2m2 is not None else "?"
            r2b_s = f"{r2b2:.2f}" if r2b2 is not None else "?"
            lines.append(f"  H{t['id']}: Sütun{t_sut}/Satır{t_sat} "
                         f"({t['x']:.2f}m,{t['y']:.2f}m) "
                         f"{t['amp']:.1f}nT ~{d2v:.2f}m [{tip2}] "
                         f"R2M={r2m_s} R2B={r2b_s}")
            lines.append(f"  Tahmin: {tah[0][0]}")
            hlist.append({
                'id': t['id'], 'x': round(t['x'], 3), 'y': round(t['y'], 3),
                'amp_nT': round(float(t['amp']), 2), 'derinlik_m': round(d2v, 3),
                'guven_pct': gv2, 'dipol_tip': tip2,
                'r2_metal': round(float(r2m2), 3) if r2m2 is not None else None,
                'r2_bosluk': round(float(r2b2), 3) if r2b2 is not None else None,
                'tahmin_1': tah[0][0],
            })

        if self.ai_sonuc:
            a = self.ai_sonuc
            lines += [
                "",
                "SEÇİLİ NOKTA:",
                f"  Konum:{a.get('x',0):.2f},{a.get('y',0):.2f}  Şiddet:{a.get('val',0):.1f}nT",
                f"  Derinlik:~{a.get('depth',0):.2f}m  Güven:%{a.get('guven','?')}",
                f"  Metal R²:{a.get('r2m','?')}  Boşluk R²:{a.get('r2b','?')}",
                f"  Teşhis:{a.get('durum','')}  {a.get('aciklama','')}",
                f"  Tahmin:{a.get('tahminler',[['']])[0][0] if a.get('tahminler') else '?'}",
            ]
        lines.append("\n--- Rapor Sonu ---")
        txt = "\n".join(lines)

        # CSV içeriği
        csv_buf = StringIO()
        if hlist:
            writer = csv.DictWriter(csv_buf, fieldnames=hlist[0].keys())
            writer.writeheader()
            writer.writerows(hlist)

        # JSON içeriği
        def sf(v):
            return float(v) if isinstance(v, (np.floating, np.integer)) else v
        json_data = {
            'tarih': now.isoformat(),
            'dosya': dosya_adi,
            'adim_cm': self.adim_m * 100,
            'mod': mod,
            'hedefler': hlist,
            'secili': ({k: sf(v) for k, v in self.ai_sonuc.items()
                        if not isinstance(v, (list, np.ndarray))}
                       if self.ai_sonuc else {}),
        }
        return txt, csv_buf.getvalue(), json.dumps(json_data, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  GÖRSEL ÇIKTI — matplotlib figür döndür
# ─────────────────────────────────────────────────────────────────────────────
def harita_ciz(analiz: C3Analiz, zi, xi, yi, targets, mod, sel_x=None, sel_y=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0d0d0d')
    ax.tick_params(colors='#555', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#222')

    zmin, zmax = zi.min(), zi.max()
    if zmin < 0 < zmax:
        nz = zi[zi != 0]
        ph = np.percentile(np.abs(nz), 98) if len(nz) > 0 else max(abs(zmin), abs(zmax))
        norm = TwoSlopeNorm(vmin=-max(ph, 0.001), vcenter=0, vmax=max(ph, 0.001))
    else:
        norm = Normalize(vmin=zmin, vmax=zmax)

    im = ax.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                   origin='lower', cmap=C3_CMAP, norm=norm,
                   aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label='nT')

    for t in targets:
        ax.plot(t['x'], t['y'], 'w+', ms=14, mew=2.5)
        ax.text(t['x'] + .02, t['y'] + .02, f"H{t['id']}",
                color='white', fontsize=9, weight='bold')

    x0 = analiz.df['sutun_m'].min()
    y0 = analiz.df['satir_m'].min()
    ax.plot(x0, y0, '*', color='yellow', ms=13, zorder=5)
    ax.text(x0, y0, ' BAŞLANGIÇ', color='yellow', fontsize=8, weight='bold')

    if sel_x is not None and sel_y is not None:
        ax.axvline(sel_x, color='#FF4500', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(sel_y, color='#FF4500', lw=0.8, ls='--', alpha=0.7)
        ax.plot(sel_x, sel_y, 'o', color='#FF4500', ms=8, mew=2, mfc='none')

    sutun_sayilari  = list(range(analiz.sutun_min, analiz.sutun_max + 1))
    sutun_metreleri = [(s - analiz.sutun_min) * analiz.adim_m for s in sutun_sayilari]
    satir_sayilari  = list(range(analiz.satir_min, analiz.satir_max + 1))
    satir_metreleri = [(s - analiz.satir_min) * analiz.adim_m for s in satir_sayilari]

    ax.set_xticks(sutun_metreleri)
    ax.set_xticklabels([f"S{s}" for s in sutun_sayilari], fontsize=7, color='#aaa')
    ax.set_yticks(satir_metreleri)
    ax.set_yticklabels([f"R{s}" for s in satir_sayilari], fontsize=7, color='#aaa')
    ax.set_xlabel(f"SÜTUN  ({analiz.n_sutun} sütun × {analiz.adim_m*100:.0f}cm"
                  f" = {(analiz.n_sutun-1)*analiz.adim_m:.1f}m)", fontsize=8, color='#aaa')
    ax.set_ylabel(f"SATIR  ({analiz.n_satir} satır × {analiz.adim_m*100:.0f}cm"
                  f" = {(analiz.n_satir-1)*analiz.adim_m:.1f}m)", fontsize=8, color='#aaa')

    br = '#00FF9D' if targets else '#FF4444'
    bl = f"C3 — {mod} | " + (f"{len(targets)} Hedef" if targets else "ANOMALİ YOK")
    ax.set_title(bl, color=br, fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def profil_ciz(analiz: C3Analiz, ai_sonuc: dict):
    if not ai_sonuc:
        return None
    plt.style.use('dark_background')
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor('#0a0a0f')

    xi  = ai_sonuc['xi']
    yi  = ai_sonuc['yi']
    xp  = ai_sonuc['xp']
    yp  = ai_sonuc['yp']
    fp  = ai_sonuc.get('fp')
    r2s = ai_sonuc.get('r2s')
    tip = ai_sonuc.get('dipol_tip')
    gain = 1.0  # xp/yp zaten gain uygulanmış

    for ax in [ax_x, ax_y]:
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#555', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222')

    ax_x.plot(xi, xp, color='#FFD700', lw=1.3, label='Veri')
    if fp is not None and r2s is not None:
        fr = ('#00FF9D' if r2s >= 0.80 else '#FFD700' if r2s >= 0.50 else '#FF4444') \
            if tip == 'metal' else \
            ('#00CFFF' if r2s >= 0.75 else '#88AAFF' if r2s >= 0.50 else '#888888')
        ax_x.plot(xi, fp, color=fr, lw=1.0, ls='--', alpha=0.85,
                  label=f"{(tip or '').upper()} R²={r2s:.2f}")
        ax_x.legend(fontsize=7, loc='upper right', facecolor='#1a1a2e',
                    edgecolor='#333', labelcolor='#aaa')
    ax_x.axhline(0, color='#444', lw=.5)
    ax_x.axvline(ai_sonuc['x'], color='#FF4500', lw=.8, ls='--')
    ax_x.set_title("X Kesiti (dipol fit)", fontsize=8, color='#aaa')
    ax_x.set_xlabel("X (m)", fontsize=7, color='#aaa')

    ax_y.plot(yi, yp, color='#00CFFF', lw=1.3)
    ax_y.axhline(0, color='#444', lw=.5)
    ax_y.axvline(ai_sonuc['y'], color='#FF4500', lw=.8, ls='--')
    ax_y.set_title("Y Kesiti", fontsize=8, color='#aaa')
    ax_y.set_xlabel("Y (m)", fontsize=7, color='#aaa')

    fig.tight_layout(pad=1.0)
    return fig


def surface_3d(analiz: C3Analiz):
    if analiz.zi is None:
        return None
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.set_facecolor('#0a0a0f')
    for p in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        p.fill = False; p.set_edgecolor('#222')
    ax3.tick_params(colors='#777', labelsize=7)

    surf = ax3.plot_surface(analiz.grid_x, analiz.grid_y, analiz.zi,
                            cmap=C3_CMAP, edgecolor='none', alpha=0.95)
    cb = fig.colorbar(surf, ax=ax3, shrink=0.4, aspect=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')
    cb.set_label('nT', color='#aaa', fontsize=8)
    ax3.set_xlabel('X (m) — Sütun', color='#aaa', fontsize=9, labelpad=8)
    ax3.set_ylabel('Y (m) — Satır', color='#aaa', fontsize=9, labelpad=8)
    ax3.set_zlabel('nT', color='#aaa', fontsize=9, labelpad=8)

    xi = analiz.xi_arr
    yi = analiz.yi_arr
    zr = analiz.zi.max() - analiz.zi.min()
    ph = zr * 0.12

    for t in analiz.targets:
        try:
            ri2 = np.argmin(np.abs(yi - t['y']))
            ci2 = np.argmin(np.abs(xi - t['x']))
            pt = analiz.zi[ri2, :]
            _, _, _, _, _, tip2, _ = analiz._dipol_fit(pt, xi)
            pr = {'metal': '#FF4444', 'bosluk': '#00CFFF', 'belirsiz': '#FFD700'}.get(tip2, '#FFD700')
            zt = analiz.zi[ri2, ci2]
            ax3.scatter([t['x']], [t['y']], [zt + ph], color=pr, s=120, marker='^', zorder=10)
            ax3.plot([t['x'], t['x']], [t['y'], t['y']], [zt, zt + ph], color=pr, lw=1.2, alpha=0.8)
            ax3.text(t['x'], t['y'], zt + ph * 1.2, f"H{t['id']}",
                     color='white', fontsize=8, weight='bold', ha='center')
        except Exception:
            pass

    ax3.set_title("C3 3D  (Kırmızı=Metal · Mavi=Boşluk · Sarı=Belirsiz)",
                  color='#00FF9D', pad=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLİT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="C3 Manyetik Analiz",
        page_icon="🧲",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

    html, body, [class*="css"] { background-color: #0a0a0f; color: #c8c8d8; }
    .stApp { background: #0a0a0f; }

    h1, h2, h3 { font-family: 'Exo 2', sans-serif !important; letter-spacing: 0.04em; }
    code, pre, .mono { font-family: 'Share Tech Mono', monospace !important; }

    .block-container { padding-top: 1.5rem; }

    /* Header band */
    .c3-header {
        background: linear-gradient(90deg, #000820 0%, #001040 50%, #000820 100%);
        border: 1px solid #00FF9D33;
        border-radius: 6px;
        padding: 14px 22px;
        margin-bottom: 18px;
    }
    .c3-header h1 { color: #00FF9D; margin: 0; font-size: 1.6rem; font-weight: 800; }
    .c3-header span { color: #556688; font-size: 0.85rem; font-family: 'Share Tech Mono', monospace; }

    /* Metric cards */
    .metric-card {
        background: #0d1220;
        border: 1px solid #1a2a40;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-label { color: #556688; font-size: 0.72rem; font-family: 'Share Tech Mono', monospace; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { color: #00FF9D; font-size: 1.3rem; font-weight: 700; font-family: 'Exo 2', sans-serif; }

    /* Teşhis badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
        font-weight: bold;
        letter-spacing: 0.05em;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #080810; }
    section[data-testid="stSidebar"] .stSlider label { font-size: 0.78rem !important; color: #778899 !important; }

    /* Hedef listesi */
    .target-row {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        padding: 6px 10px;
        border-left: 3px solid #00FF9D44;
        background: #0d1220;
        margin-bottom: 4px;
        border-radius: 0 4px 4px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="c3-header">
      <h1>🧲 C3 MANYETİK GRADİOMETRE ANALİZ</h1>
      <span>İki sensörlü manyetik tarama · Derinlik tahmini · Dipol fit · Obje sınıflandırma</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar — Dosya & Parametreler ───────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Veri Yükle")
        uploaded = st.file_uploader("C3 CSV Dosyası", type=["csv"],
                                    help="Sütunlar: satir, sutun, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z")

        st.markdown("---")
        st.markdown("### ⚙️ Tarama Ayarları")
        adim_cm = st.radio("Adım mesafesi (cm)", [20, 25, 50], index=2, horizontal=True)
        adim_cm_man = st.number_input("veya manuel (cm)", min_value=1, max_value=200,
                                       value=adim_cm, step=1)
        adim_m = adim_cm_man / 100.0

        st.markdown("---")
        st.markdown("### 🎛️ Görüntüleme")
        mod = st.radio("Mod", ['TFA', 'Sadece Z', 'Gradient', 'Analitik', 'FFT Derin', 'FFT Sig'])

        gain    = st.slider("Kazanç",   1,   1000, 100, step=5)
        esik    = st.slider("Eşik",     0,    500,  25, step=5)
        blur    = st.slider("Yumuşat",  0.0,  5.0, 0.5, step=0.1)
        noise   = st.select_slider("Parazit (medyan)", options=[1, 3, 5, 7, 9], value=3)
        sigma   = st.select_slider("Sigma (σ)",       options=[1, 2, 3], value=3)

        st.markdown("---")
        st.markdown("### 🖱️ Nokta Seçimi")
        st.caption("Harita üzerindeki X/Y koordinatını girin (metre).")
        sel_x = st.number_input("Seçili X (m)", value=0.0, step=0.1, format="%.2f")
        sel_y = st.number_input("Seçili Y (m)", value=0.0, step=0.1, format="%.2f")

    # ── Ana içerik ────────────────────────────────────────────────────────────
    if uploaded is None:
        st.info("👈 Sol panelden CSV dosyanızı yükleyin. "
                "Dosyada şu sütunlar bulunmalıdır: **satir, sutun, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z**")

        # Örnek veri oluşturma rehberi
        with st.expander("📋 Örnek CSV Formatı"):
            st.code("""satir,sutun,s1_x,s1_y,s1_z,s2_x,s2_y,s2_z
1,1,12.3,-5.1,48200.5,11.8,-5.3,48195.2
1,2,13.1,-4.9,48210.3,12.5,-5.0,48205.1
...
""", language="csv")
        return

    # ── Analiz ────────────────────────────────────────────────────────────────
    csv_bytes = uploaded.read()
    cache_key = (hash(csv_bytes), adim_m)

    @st.cache_resource(show_spinner="Veri yükleniyor ve ön işlem yapılıyor…")
    def yukle(key, _bytes, _adim):
        return C3Analiz(_bytes, adim_m=_adim)

    try:
        analiz = yukle(cache_key, csv_bytes, adim_m)
    except Exception as e:
        st.error(f"❌ Veri yüklenemedi: {e}")
        st.stop()

    # Sınırlar
    x_max = float(analiz.df['sutun_m'].max())
    y_max = float(analiz.df['satir_m'].max())
    sel_x = float(np.clip(sel_x, 0.0, x_max))
    sel_y = float(np.clip(sel_y, 0.0, y_max))

    # Analiz hesapla
    with st.spinner("Analiz hesaplanıyor…"):
        zi, xi, yi, targets, esik_val, ai_sonuc = analiz.analiz_et(
            gain=gain, noise_size=noise, blur=blur,
            sigma_mult=sigma, esik_val=esik, mod=mod,
            sel_x=sel_x, sel_y=sel_y
        )

    # ── Özet metrikler ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Hedef Sayısı</div>
          <div class="metric-value">{'🔴 ' if targets else ''}{len(targets)}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Gürültü Std (nT)</div>
          <div class="metric-value">{analiz.gurultu_std:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Grid</div>
          <div class="metric-value">{analiz.n_sutun}×{analiz.n_satir}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        aktif_esik = max(analiz.gurultu_std * sigma * gain, esik)
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Aktif Eşik</div>
          <div class="metric-value">{aktif_esik:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── İki sütun: harita + bilgi ─────────────────────────────────────────────
    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.markdown("#### 🗺️ 2D Harita")
        fig2d = harita_ciz(analiz, zi, xi, yi, targets, mod, sel_x, sel_y)
        buf2d = BytesIO()
        fig2d.savefig(buf2d, format='png', dpi=120, bbox_inches='tight',
                      facecolor='#0a0a0f')
        plt.close(fig2d)
        st.image(buf2d, use_container_width=True)

        # Profil grafiği
        if ai_sonuc:
            st.markdown("#### 📈 Profil Kesitleri")
            fig_p = profil_ciz(analiz, ai_sonuc)
            if fig_p:
                buf_p = BytesIO()
                fig_p.savefig(buf_p, format='png', dpi=100, bbox_inches='tight',
                              facecolor='#0a0a0f')
                plt.close(fig_p)
                st.image(buf_p, use_container_width=True)

    with right_col:
        # Teşhis paneli
        if ai_sonuc:
            durum = ai_sonuc.get('durum', '')
            renk  = ai_sonuc.get('renk', '#888')
            acikl = ai_sonuc.get('aciklama', '')
            depth = ai_sonuc.get('depth', 0.0)
            guven = ai_sonuc.get('guven', '?')
            r2m   = ai_sonuc.get('r2m')
            r2b   = ai_sonuc.get('r2b')
            val   = ai_sonuc.get('val', 0.0)
            tip   = ai_sonuc.get('dipol_tip', '')
            ystr  = ai_sonuc.get('yontem', '')
            tahminler = ai_sonuc.get('tahminler', [])

            st.markdown(f"""
            <div style="border:1px solid {renk}44; border-radius:6px; padding:14px; background:#0d1220; margin-bottom:12px;">
              <div style="font-size:0.7rem; color:#556688; font-family:'Share Tech Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px;">TEŞHİS — ({sel_x:.2f}m, {sel_y:.2f}m)</div>
              <span class="badge" style="background:{renk}22; color:{renk}; border:1px solid {renk}66;">{durum}</span>
              <div style="margin-top:8px; color:#aaa; font-size:0.82rem;">{acikl}</div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Derinlik (m)", f"~{depth:.2f}")
                st.metric("Şiddet (nT)", f"{val:.1f}")
            with col_b:
                st.metric("Güven", f"%{guven}" if guven else "?")
                st.metric("Dipol Tipi", tip or "—")

            if r2m is not None or r2b is not None:
                c1, c2 = st.columns(2)
                c1.metric("Metal R²", f"{r2m:.2f}" if r2m is not None else "?")
                c2.metric("Boşluk R²", f"{r2b:.2f}" if r2b is not None else "?")

            if ystr:
                st.caption(f"🔬 Yöntem: `{ystr}`")

            # Obje tahminleri
            st.markdown("**🔍 Obje Tahminleri**")
            for isim, puan, rk in tahminler:
                bar = "█" * max(1, int(puan * 5)) + "░" * (5 - max(1, int(puan * 5)))
                st.markdown(
                    f'<div class="target-row" style="border-color:{rk}aa;">'
                    f'<span style="color:{rk}">{bar}</span> '
                    f'<span style="color:#aaa">%{int(puan*100)}</span> '
                    f'<span style="color:#ddd">{isim}</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Bir nokta seçmek için sol paneldeki X/Y değerlerini ayarlayın.")

        st.markdown("---")

        # Hedef listesi
        st.markdown("#### 🎯 Tespit Edilen Hedefler")
        if not targets:
            st.warning("Bu parametre ayarlarıyla anomali tespit edilmedi.")
        else:
            for t in targets:
                t_sut = round(t['x'] / analiz.adim_m) + analiz.sutun_min
                t_sat = round(t['y'] / analiz.adim_m) + analiz.satir_min
                st.markdown(
                    f'<div class="target-row">'
                    f'<strong style="color:#00FF9D">H{t["id"]}</strong> '
                    f'S{t_sut}/R{t_sat} — '
                    f'({t["x"]:.2f}m, {t["y"]:.2f}m) — '
                    f'<span style="color:#FFD700">{t["amp"]:.1f} nT</span></div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── 3D Görselleştirme ─────────────────────────────────────────────────────
    with st.expander("🌐 3D Yüzey Görünümü"):
        fig3d = surface_3d(analiz)
        if fig3d:
            buf3d = BytesIO()
            fig3d.savefig(buf3d, format='png', dpi=100, bbox_inches='tight',
                          facecolor='#0a0a0f')
            plt.close(fig3d)
            st.image(buf3d, use_container_width=True)

    # ── Ham Veri Önizleme ─────────────────────────────────────────────────────
    with st.expander("📊 Ham Veri Önizleme"):
        st.dataframe(analiz.df.head(100).style.background_gradient(cmap='Blues'),
                     use_container_width=True)

    # ── Rapor İndir ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Rapor İndir")
    rapor_txt, rapor_csv, rapor_json = analiz.rapor_olustur(mod, uploaded.name if uploaded else "")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📄 TXT Raporu İndir", data=rapor_txt,
                           file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain", use_container_width=True)
    with c2:
        st.download_button("📊 CSV Hedefler İndir", data=rapor_csv,
                           file_name=f"C3_Hedefler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", use_container_width=True)
    with c3:
        st.download_button("🔧 JSON Raporu İndir", data=rapor_json,
                           file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json", use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; margin-top:24px; color:#334455;
                font-family:'Share Tech Mono',monospace; font-size:0.72rem;">
      C3 Manyetik Gradyometre Analiz Sistemi · Streamlit Arayüzü
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
