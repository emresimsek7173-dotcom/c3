import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from scipy.optimize import curve_fit
import json, csv
from tkinter import filedialog, Tk, messagebox
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  SABİTLER
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_MESAFESI  = 0.80   # iki sensör arası mesafe (m)
YUKSEKLIK_SABITI = 0.05   # sensörün yerden yüksekliği (m)
MAX_DEPTH        = 5.0    # maksimum derinlik (m)
DEFAULT_GRID_RES = 100    # varsayılan grid çözünürlüğü
ADIM_MESAFESI    = 0.50   # varsayılan adım mesafesi (m)

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA', '#0066FF', '#00CCFF', '#00CC44',
    '#FFFF00', '#FF6600', '#CC0000'], N=512)


# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI — Türkçe karakter normalizasyonu
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_col(s: str) -> str:
    """Sütun adını ASCII'ye normalize et (Türkçe dahil tüm aksan işaretleri)."""
    # Unicode NFKD → aksan işaretlerini ayır → ASCII'ye dönüştür
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return (s.strip().lower()
            .replace('i', 'i')   # ı zaten yukarıda hallolur ama açık bırak
            .replace(' ', '_'))


# ─────────────────────────────────────────────────────────────────────────────
#  OBJE TAHMİN MOTORU
# ─────────────────────────────────────────────────────────────────────────────
def obje_tahmini(tip, r2_m, r2_b, fwhm, derinlik, val, snr, mod):
    """
    Tüm kanıtları birleştirerek olası obje tipini tahmin eder.
    Çömlek, küp, demir, boru, oda, kaya gibi somut tahminler üretir.
    Güven düzeyiyle birlikte döndürür.

    Parametreler
    ------------
    mod : str — Aktif görüntüleme modu; Gradient/Analitik için farklı yorumlar.
    """
    tahminler = []  # (obje_adi, puan, renk)

    d  = derinlik or 0.0
    fw = fwhm     or 0.5
    am = abs(val)

    # ── Gradient / Analitik modunda farklı yorum ─────────────────────────────
    if mod in ('Gradient', 'Analitik'):
        tahminler.append(("Kenar/Sinir anomalisi — detay modda incele", 0.50, '#FFA500'))
        if not tahminler:
            return [("Tanımlanamadı", 0.0, '#555555')]
        tahminler.sort(key=lambda x: x[1], reverse=True)
        return tahminler[:3]

    # ── Metal grubu ───────────────────────────────────────────────────────────
    if tip == 'metal' and r2_m >= 0.35:
        if fw < 0.25 and d < 0.5:
            tahminler.append(("Küçük metal obje (sikke/parça)", r2_m * 0.90 + (1 - d) * 0.10, '#FF6600'))
        elif fw < 0.7 and 0.2 < d < 1.2:
            tahminler.append(("Metal küp / sandık / kap",       r2_m * 0.85,                  '#FF4500'))
        elif fw >= 0.7 and d < 1.5:
            tahminler.append(("Metal boru / ray / levha",        r2_m * 0.80,                  '#FF2200'))
        elif d >= 1.5:
            tahminler.append(("Derin metal yapı",                r2_m * 0.70,                  '#FF0000'))
        # Genel metal (her durumda eklenir, sıralamada arkada kalır)
        tahminler.append(("Metal obje",                          r2_m * 0.60,                  '#FF8C00'))

    # ── Boşluk / seramik grubu ────────────────────────────────────────────────
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

    # ── Karma / çelişkili ─────────────────────────────────────────────────────
    if tip == 'belirsiz' or (r2_m > 0.30 and r2_b > 0.30 and abs(r2_m - r2_b) < 0.20):
        if fw < 0.6 and d < 1.0:
            tahminler.append(("Çömlek (metal içerikli?)",        0.50, '#FFD700'))
            tahminler.append(("Pişmiş kil / karma malzeme",      0.45, '#FFA500'))
        else:
            tahminler.append(("Karma yapı / belirsiz",           0.30, '#888888'))

    # ── Zemin anomalisi ───────────────────────────────────────────────────────
    if snr < 2.0 or am < 0.5:
        tahminler.append(("Zemin gürültüsü / mineral iz",        0.40, '#666666'))
        tahminler.append(("Manyetik kaya / doğal anomali",       0.35, '#556655'))

    if not tahminler:
        return [("Tanımlanamadı", 0.0, '#555555')]

    tahminler.sort(key=lambda x: x[1], reverse=True)
    return tahminler[:3]


# ─────────────────────────────────────────────────────────────────────────────
#  ANA SINIF
# ─────────────────────────────────────────────────────────────────────────────
class C3Analiz:
    def __init__(self, path, adim_m=None):
        self.path      = Path(path)
        self.adim_m    = adim_m if adim_m else ADIM_MESAFESI
        self.grid_res  = DEFAULT_GRID_RES   # FIX: global yerine örnek değişkeni
        self.zi        = None
        self.zi_raw    = None
        self.grid_x    = None  # shape: (n_sutun, n_satir) — meshgrid(sutun, satir)
        self.grid_y    = None
        self.xi_arr    = None  # sutun_m ekseni (X)
        self.yi_arr    = None  # satir_m ekseni (Y)
        self.targets   = []
        self.sel       = [0.0, 0.0]
        self.ai_sonuc  = {}
        self._veri_yukle()

    # ── Veri yükleme ──────────────────────────────────────────────────────────
    def _veri_yukle(self):
        try:
            df = pd.read_csv(self.path)
            # FIX: unicodedata tabanlı kapsamlı normalizasyon
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

            # FIX: self.grid_res — global değişken yok
            self.grid_res = max(20, min(DEFAULT_GRID_RES,
                                        max(self.n_satir, self.n_sutun) * 4))
        except Exception as e:
            messagebox.showerror("Hata", f"Veri yüklenemedi:\n{e}")
            raise

    # ── Grid ──────────────────────────────────────────────────────────────────
    def _grid_olustur(self, veri_col):
        """
        Tutarlı eksen tanımı:
          xi  = sutun_m ekseni  (X, soldan sağa)   shape: (grid_res,)
          yi  = satir_m ekseni  (Y, aşağıdan yukarı) shape: (grid_res,)
          meshgrid(xi, yi) → grid_X.shape = (grid_res, grid_res)
            grid_X[row, col] = xi[col]  ← sütun (X)
            grid_Y[row, col] = yi[row]  ← satır (Y)
          zi[row, col] = veri @ (xi[col], yi[row])
        """
        xi = np.linspace(self.df['sutun_m'].min(), self.df['sutun_m'].max(), self.grid_res)
        yi = np.linspace(self.df['satir_m'].min(), self.df['satir_m'].max(), self.grid_res)
        # FIX: meshgrid(xi, yi) — X=sütun, Y=satır; indexing='xy' (varsayılan)
        grid_X, grid_Y = np.meshgrid(xi, yi)   # her ikisi de (grid_res, grid_res)

        zi = griddata(
            (self.df['sutun_m'], self.df['satir_m']),   # (X, Y) noktaları
            self.df[veri_col],
            (grid_X, grid_Y),
            method='linear', fill_value=0)
        # zi[row, col] = veri @ (xi[col], yi[row]) — transpose GEREKMİYOR
        return xi, yi, grid_X, grid_Y, zi

    # ── FFT filtre ─────────────────────────────────────────────────────────────
    def _fft_filtre(self, zi, mod='derin'):
        """
        FIX: Hanning geçiş penceresi — bw normalizasyonu düzeltildi.
        han(r, fc) → r < fc-bw: 1.0,  geçiş: cos penceresi,  r > fc+bw: 0.0
        """
        F    = np.fft.fftshift(np.fft.fft2(zi))
        rows, cols = zi.shape
        u = np.fft.fftshift(np.fft.fftfreq(cols))
        v = np.fft.fftshift(np.fft.fftfreq(rows))
        UU, VV = np.meshgrid(u, v)
        R = np.sqrt(UU**2 + VV**2)

        def han(r, fc, bw=0.04):
            m = np.ones_like(r)
            low  = fc - bw
            high = fc + bw
            # FIX: geçiş bandında doğru normalizasyon (bw, 2*bw değil)
            transit = (r > low) & (r < high)
            m[transit] = 0.5 * (1 + np.cos(np.pi * (r[transit] - low) / bw))
            m[r >= high] = 0.0
            return m

        if mod == 'derin':
            filtre = han(R, 0.08)          # alçak geçiren — derin gömülü cisimler
        else:
            # FIX: yüksek geçiren — sığ yüzey gürültüsünü bastır
            filtre = 1.0 - han(R, 0.10)

        return np.real(np.fft.ifft2(np.fft.ifftshift(F * filtre)))

    # ── Filtre ────────────────────────────────────────────────────────────────
    def _filtrele(self, zi):
        """
        FIX: Uygulama sırası düzeltildi:
          1. Kazanç
          2. Median (gürültü bastırma)
          3. Gaussian blur  ← önce blur
          4. Eşik uygula    ← sonra eşik (halo artefaktı önlenir)
          5. Mod dönüşümü (Gradient, FFT …)
        """
        zi  = zi * self.s_gain.val
        med = int(self.s_noise.val)
        if med > 1:
            size = med if med % 2 else med + 1
            zi   = median_filter(zi, size=size)

        # FIX: blur ÖNCE
        if self.s_blur.val > 0:
            zi = gaussian_filter(zi, sigma=self.s_blur.val)

        # FIX: eşik SONRA
        oto  = self.gurultu_std * self.s_sigma.val * self.s_gain.val
        esik = max(oto, self.s_esik.val)
        zi   = np.where(np.abs(zi) < esik, 0, zi)

        mod = self.r_mode.value_selected
        if   mod == 'Gradient':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2)
        elif mod == 'Analitik':  zi = np.sqrt(sobel(zi, 1)**2 + sobel(zi, 0)**2 + zi**2)
        elif mod == 'FFT Derin': zi = self._fft_filtre(zi, 'derin')
        elif mod == 'FFT Sig':   zi = self._fft_filtre(zi, 'sig')
        return zi, esik

    # ── Hedef tespiti ──────────────────────────────────────────────────────────
    def _hedef_tespit(self, zi, xi, yi, filtre_esik):
        """
        FIX: GRID_RES yerine zi.shape kullanılıyor (indeks taşması giderildi).
        FIX: Koordinat ataması xi/yi eksen tanımıyla tutarlı hale getirildi:
          xi = sutun_m (X),  yi = satir_m (Y)
          zi[row, col] → row=Y indeksi, col=X indeksi
        """
        hedef_esik = filtre_esik * 0.60
        min_esik   = self.gurultu_std * 0.8 * self.s_gain.val
        hedef_esik = max(hedef_esik, min_esik)

        binary      = np.abs(zi) > hedef_esik
        labeled, num = label(binary)
        rows, cols  = zi.shape   # FIX: shape'ten al
        targets     = []

        for i in range(1, num + 1):
            mask = labeled == i
            if mask.sum() < 2:
                continue
            coords = np.argwhere(mask)          # (row, col)
            peak   = np.argmax(np.abs(zi[mask]))
            py, px = coords[peak]               # py=row(Y), px=col(X)
            py = min(py, rows - 1)              # FIX: zi.shape sınırı
            px = min(px, cols - 1)
            targets.append({
                'id':  i,
                'x':   xi[px],   # FIX: xi=sutun_m=X, indeks=px(col)
                'y':   yi[py],   # FIX: yi=satir_m=Y, indeks=py(row)
                'amp': zi[py, px],
            })

        return sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:8]

    # ── Faz kayması ───────────────────────────────────────────────────────────
    def _faz_kaymasi(self, x_prof, y_prof):
        try:
            neg = x_prof < 0
            if neg.sum() < 2:
                return 0.0, "Negatif bölge yok"
            neg_idx  = np.where(neg)[0]
            cukur    = int(np.mean(neg_idx))
            ana      = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
            tepe     = int(np.argmax(ana))
            mesafe   = abs(tepe - cukur)
            ortusme  = max(0.0, 1.0 - mesafe / (len(x_prof) * 0.3))
            yorum = ("Manyetik olmayan/bosluk" if ortusme > 0.80 else
                     "Karisik sinyal"           if ortusme > 0.50 else
                     "Demir dipol")
            return ortusme, yorum
        except Exception:
            return 0.0, "Hesaplanamadi"

    # ── Tepe sivrilik ─────────────────────────────────────────────────────────
    def _tepe_sivrilik(self, x_prof, xi):
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
            form = ("Sivri → kucuk yogun" if fwhm < 0.3 else
                    "Orta → hacimli kutle" if fwhm < 0.8 else
                    "Yayvan → buyuk yapi")
            return fwhm, form
        except Exception:
            return None, "Hesaplanamadi"

    # ── Derinlik — yedek ─────────────────────────────────────────────────────
    def _derinlik(self, peak_nt):
        """
        FIX: W/2 yöntemi — manyetik anomalinin yarı-genişliğinden derinlik tahmini.
        Formül: d ≈ 0.5 × W½ − h  (W½ = tepe genişliği yarı maksimumda, h = yükseklik)
        Burada peak_nt skaler olduğundan W½ ≈ SENSOR_MESAFESI varsayılır (yedek).
        """
        if abs(peak_nt) < 1e-9:
            return 0.0
        # W/2 yaklaşımı: iki sensör arası mesafe = anomali genişliği tahmini
        w_half = SENSOR_MESAFESI * 0.5
        d = max(0.0, w_half - YUKSEKLIK_SABITI)
        return min(d, MAX_DEPTH)

    # ── Derinlik — profesyonel dinamik ───────────────────────────────────────
    def _derinlik_pro(self, profil, eks):
        try:
            if len(profil) < 4 or (eks[-1] - eks[0]) < 0.01:
                return self._derinlik(np.max(np.abs(profil))), None, "Profil yetersiz"
            adim     = (eks[-1] - eks[0]) / max(len(eks) - 1, 1)
            sonuclar = []
            yontemler= []
            abs_p    = np.abs(profil)
            tv       = abs_p.max()
            ti       = int(np.argmax(abs_p))
            snr      = tv / (self.gurultu_std + 1e-9)

            # P½ (yarı-maksimum genişliği)
            if tv > 1e-6:
                y2  = tv * 0.5
                sol = np.where(abs_p[:ti] < y2)[0]
                sag = np.where(abs_p[ti:] < y2)[0]
                if len(sol) > 0 and len(sag) > 0:
                    x12 = (ti - sol[-1] + sag[0]) * adim / 2.0
                    dp  = max(0.0, (0.6 * 0.5 + 0.4 * 0.7) * x12 - YUKSEKLIK_SABITI)
                    if 0.01 < dp < MAX_DEPTH:
                        w = min(0.5, 0.15 + 0.35 * min(snr / 10.0, 1.0))
                        sonuclar.append((dp, w))
                        yontemler.append(f"P½={dp:.2f}m")

            # FWHM analitik sinyal
            grad = np.gradient(profil, adim)
            ana  = np.sqrt(grad**2 + profil**2)
            at   = ana.max()
            if at > 1e-6:
                idx = np.where(ana >= at * 0.5)[0]
                if len(idx) >= 2:
                    fwhm = (idx[-1] - idx[0]) * adim
                    fd   = np.sqrt(max(fwhm**2 - (SENSOR_MESAFESI * 0.5)**2,
                                       fwhm**2 * 0.1))
                    df2  = max(0.0, fd / 2.0 - YUKSEKLIK_SABITI)
                    if 0.01 < df2 < MAX_DEPTH:
                        w = min(0.40, 0.10 + 0.30 * (len(idx) / len(profil)))
                        sonuclar.append((df2, w))
                        yontemler.append(f"FWHM={df2:.2f}m")

            # Euler dekonvolüsyon (basit)
            if tv > 1e-6 and len(grad) > 4:
                lo = max(0, ti - len(profil) // 6)
                hi = min(len(profil), ti + len(profil) // 6 + 1)
                Bw = profil[lo:hi]; dw = grad[lo:hi]
                mk = np.abs(dw) > tv * 0.05
                if mk.sum() >= 3:
                    oran = np.abs(Bw[mk]) / (np.abs(dw[mk]) + 1e-9)
                    de   = max(0.0, np.median(oran) * 1.5 - YUKSEKLIK_SABITI)
                    if 0.01 < de < MAX_DEPTH:
                        gt = 1.0 - min(np.std(oran) / (np.mean(oran) + 0.01), 1.0)
                        w  = min(0.25, 0.05 + 0.20 * gt)
                        sonuclar.append((de, w))
                        yontemler.append(f"Euler={de:.2f}m")

            if not sonuclar:
                return self._derinlik(tv), None, "Yeterli profil yok"

            tw    = sum(w for _, w in sonuclar)
            df    = sum(d * w for d, w in sonuclar) / tw
            df    = max(0.0, min(df, MAX_DEPTH))

            if len(sonuclar) >= 2:
                vals  = [d for d, _ in sonuclar]
                tut   = 1.0 - min(np.std(vals) / (np.mean(vals) + 0.01), 1.0)
                snrs  = min(snr / 15.0, 1.0)
                guven = int((tut * 0.7 + snrs * 0.3) * 100)
            else:
                guven = max(20, int(min(snr / 20.0, 1.0) * 50))

            return df, guven, " | ".join(yontemler)
        except Exception:
            peak_val = np.max(np.abs(profil)) if len(profil) > 0 else 0.0
            return self._derinlik(peak_val), None, "Hata"

    # ── Dipol fit — metal vs boşluk ───────────────────────────────────────────
    def _dipol_fit(self, profil, eks):
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
                return max(0.0, 1.0 - np.sum((g - t)**2) /
                           (np.sum((g - np.mean(g))**2) + 1e-12))

            # Metal fit
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

            # Boşluk fit
            r2b, zb, fitb = 0.0, None, None
            try:
                po, _ = curve_fit(bosluk_m, x, profil,
                                  p0=[abs(amp) * 0.5, 0.3, 0.0],
                                  bounds=([0.0,         0.01, -2.0],
                                          [abs(amp) * 20, MAX_DEPTH,  2.0]),
                                  maxfev=2000, ftol=1e-6)
                fitb = bosluk_m(x, *po)
                r2b  = r2(profil, fitb)
                zb   = max(0.0, abs(po[1]) - YUKSEKLIK_SABITI)
            except Exception:
                pass

            # M < 0 → ters dipol = boşluk davranışı
            if Mm is not None and Mm < 0 and r2m > 0.60:
                if fitb is None:
                    fitb, r2b, zb = fitm, r2m, zm
                r2m, zm, Mm = 0.0, None, None

            metal_ok  = r2m >= 0.35 and Mm is not None and Mm > 0
            bosluk_ok = r2b >= 0.35

            if not metal_ok and not bosluk_ok:
                bf = fitm if fitm is not None else fitb
                return (zm or zb or 0.0), max(r2m, r2b), r2m, r2b, bf, \
                       'belirsiz', "Gurultu/zemin"

            tip = ('metal' if (metal_ok and (not bosluk_ok or r2m >= r2b))
                   else 'bosluk')

            if tip == 'metal':
                zs, r2s, fs = zm, r2m, fitm
                yorum = ("Guclu metal dipol"      if r2s >= 0.85 else
                         "Metal kutle orta guven"  if r2s >= 0.60 else
                         "Zayif metal sinyali")
            else:
                zs  = zb if zb is not None else zm
                r2s = r2b if bosluk_ok else r2m
                fs  = fitb if fitb is not None else fitm
                yorum = ("Guclu bosluk/tunel"      if r2s >= 0.75 else
                         "Olasi bosluk orta guven"  if r2s >= 0.50 else
                         "Zayif bosluk sinyali")

            return (zs or 0.0), r2s, r2m, r2b, fs, tip, yorum
        except Exception:
            return None, None, None, None, None, None, "Hesaplanamadi"

    # ── Entegre teşhis ────────────────────────────────────────────────────────
    def _teshis(self, x_prof, val, tip, r2_m, r2_b, ortusme, fwhm):
        esik = self.s_esik.val
        mod  = self.r_mode.value_selected
        if abs(val) < esik:
            return "TEMIZ/SINYAL YOK", "#FFFFFF", "Anomali yok."
        if mod == 'Analitik':
            return "ENERJI MERKEZI", "#FF00FF", "Hedefin odak noktasi."
        if mod == 'Gradient':
            return "KENAR/SINIR", "#FFA500", "Anomali siniri."

        vmax = float(np.max(x_prof))
        vmin = float(np.min(x_prof))
        vr   = max(vmax - vmin, 1e-5)

        if abs(x_prof[0]) > abs(val) * 0.88 or abs(x_prof[-1]) > abs(val) * 0.88:
            return "KENAR/DEGERLI?", "#FFA500", "Sinirda kesilmis — alani buyut!"

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
            return ("KARMA/CELISKILI", "#FFD700",
                    f"Metal%{int(pm*100)} Bosluk%{int(pb*100)} — coklu tarama")

        kaz = max([('metal', pm), ('bosluk', pb), ('belirsiz', pbl)],
                  key=lambda x: x[1])

        if kaz[0] == 'metal':
            if pm > 0.70: return ("METAL KUTLE",    "#FF4500", f"Guclu metal (%{int(pm*100)})")
            else:         return ("MUHTEMEL METAL",  "#FF8C00", f"Metal ihtimali %{int(pm*100)}")
        elif kaz[0] == 'bosluk':
            if pb > 0.70: return ("BOSLUK/TUNEL",   "#00FFFF", f"Guclu bosluk (%{int(pb*100)})")
            else:         return ("MUHTEMEL BOSLUK", "#88AAFF", f"Bosluk ihtimali %{int(pb*100)}")
        else:
            return ("BELIRSIZ", "#888888", "Yetersiz sinyal — sigma/esik dus")

    # ── Ana güncelleme ────────────────────────────────────────────────────────
    def _guncelle(self, _=None):
        self.ax2d.clear()
        mod  = self.r_mode.value_selected
        vcol = 'z_diff' if mod == 'Sadece Z' else 'tfa_diff'

        # FIX: _grid_olustur artık xi, yi, grid_X, grid_Y, zi döndürüyor
        xi, yi, grid_X, grid_Y, zi_raw = self._grid_olustur(vcol)
        self.xi_arr  = xi      # sutun_m (X ekseni)
        self.yi_arr  = yi      # satir_m (Y ekseni)
        self.grid_x  = grid_X  # shape (grid_res, grid_res)
        self.grid_y  = grid_Y

        zi, esik      = self._filtrele(zi_raw)
        self.zi       = zi
        self.zi_raw   = zi_raw
        self.targets  = self._hedef_tespit(zi, xi, yi, esik)

        zmin, zmax = zi.min(), zi.max()
        if zmin < 0 < zmax:
            nz = zi[zi != 0]
            ph = np.percentile(np.abs(nz), 98) if len(nz) > 0 else max(abs(zmin), abs(zmax))
            norm = TwoSlopeNorm(vmin=-max(ph, 0.001), vcenter=0, vmax=max(ph, 0.001))
        else:
            norm = Normalize(vmin=zmin, vmax=zmax)

        # imshow: extent=[xmin, xmax, ymin, ymax], origin='lower'
        self.ax2d.imshow(zi,
                         extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                         origin='lower', cmap=C3_CMAP, norm=norm,
                         aspect='equal', interpolation='bilinear')

        for t in self.targets:
            self.ax2d.plot(t['x'], t['y'], 'w+', ms=12, mew=2)
            self.ax2d.text(t['x'] + .02, t['y'] + .02, f"H{t['id']}",
                           color='white', fontsize=9, weight='bold')

        x0 = self.df['sutun_m'].min()
        y0 = self.df['satir_m'].min()
        self.ax2d.plot(x0, y0, '*', color='yellow', ms=12, zorder=5)
        self.ax2d.text(x0, y0, ' BASLANGIC', color='yellow', fontsize=8, weight='bold')

        sutun_sayilari  = list(range(self.sutun_min, self.sutun_max + 1))
        sutun_metreleri = [(s - self.sutun_min) * self.adim_m for s in sutun_sayilari]
        satir_sayilari  = list(range(self.satir_min, self.satir_max + 1))
        satir_metreleri = [(s - self.satir_min) * self.adim_m for s in satir_sayilari]

        self.ax2d.set_xticks(sutun_metreleri)
        self.ax2d.set_xticklabels([f"S{s}" for s in sutun_sayilari], fontsize=7, color='#aaa')
        self.ax2d.set_yticks(satir_metreleri)
        self.ax2d.set_yticklabels([f"R{s}" for s in satir_sayilari], fontsize=7, color='#aaa')
        self.ax2d.set_xlabel(
            f"SUTUN  ({self.n_sutun} sutun × {self.adim_m*100:.0f}cm"
            f" = {(self.n_sutun-1)*self.adim_m:.1f}m)", fontsize=8, color='#aaa')
        self.ax2d.set_ylabel(
            f"SATIR  ({self.n_satir} satir × {self.adim_m*100:.0f}cm"
            f" = {(self.n_satir-1)*self.adim_m:.1f}m)", fontsize=8, color='#aaa')

        br = '#00FF9D' if self.targets else '#FF4444'
        bl = f"C3 — {mod} | " + (f"{len(self.targets)} Hedef" if self.targets else "ANOMALI YOK")
        self.ax2d.set_title(bl, color=br, fontsize=10, pad=6)

        self._profil_ciz(self.sel[0], self.sel[1])
        self.fig.canvas.draw_idle()

    # FIX: is not None kullanılıyor (xdata == 0.0 durumu artık çalışır)
    def _tikla(self, event):
        if (event.inaxes == self.ax2d and
                event.xdata is not None and event.ydata is not None):
            self.sel = [event.xdata, event.ydata]
            self._profil_ciz(event.xdata, event.ydata)
            self.fig.canvas.draw_idle()

    # ── Profil çiz + info ─────────────────────────────────────────────────────
    def _profil_ciz(self, xv, yv):
        if self.zi is None:
            return

        xi  = self.xi_arr   # sutun_m (X), shape (grid_res,)
        yi  = self.yi_arr   # satir_m (Y), shape (grid_res,)

        # FIX: xi/yi eksenlerine göre doğru satır/sütun indeksi
        ci = np.argmin(np.abs(xi - xv))   # sütun indeksi (X)
        ri = np.argmin(np.abs(yi - yv))   # satır indeksi (Y)

        gain = self.s_gain.val or 1.0
        # zi[row, col] → satır boyunca (sabit row): X profili
        #              → sütun boyunca (sabit col): Y profili
        xp  = self.zi_raw[ri, :] * gain   # X profili (ham × kazanç)
        yp  = self.zi_raw[:, ci] * gain   # Y profili (ham × kazanç)
        val = float(self.zi[ri, ci])
        if abs(val) < abs(xp[ci]) * 0.1:
            val = float(xp[ci])

        xn = self.zi_raw[ri, :]            # normalize profil (dipol/derinlik için)

        depth, guven, ystr          = self._derinlik_pro(xn, xi)
        ortusme, faz_y              = self._faz_kaymasi(xp, yp)
        fwhm, siv_y                 = self._tepe_sivrilik(xp, xi)
        z_d, r2s, r2m, r2b, fp, tip, dyorum = self._dipol_fit(xn, xi)
        snr = abs(val) / (self.gurultu_std * self.s_gain.val + 1e-9)

        durum, renk, aciklama = self._teshis(xp, val, tip, r2m, r2b, ortusme, fwhm)

        # Derinliği dipol ile güçlendir (metal)
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
            ystr = (ystr or "") + f" | Bosluk={z_d:.2f}m"

        # FIX: depth None olabilir — güvenli dönüşüm
        depth_val = float(depth) if depth is not None else 0.0

        tahminler = obje_tahmini(tip, r2m or 0.0, r2b or 0.0, fwhm,
                                 depth_val, val, snr, self.r_mode.value_selected)

        self.ai_sonuc = {
            'durum': durum, 'aciklama': aciklama, 'renk': renk,
            'depth': depth_val, 'val': val, 'x': xv, 'y': yv,
            'guven': guven, 'yontem': ystr,
            'r2m': r2m, 'r2b': r2b, 'dipol_tip': tip, 'dipol_yorum': dyorum,
            'tahminler': tahminler,
        }

        # ── X profil ──────────────────────────────────────────────────────────
        self.ax_xp.clear(); self._ax_karalat(self.ax_xp)
        self.ax_xp.plot(xi, xp, color='#FFD700', lw=1.2, label='Veri')
        if fp is not None and r2s is not None:
            if   tip == 'metal':  fr = ('#00FF9D' if r2s >= 0.80 else
                                        '#FFD700' if r2s >= 0.50 else '#FF4444')
            elif tip == 'bosluk': fr = ('#00CFFF' if r2s >= 0.75 else
                                        '#88AAFF' if r2s >= 0.50 else '#888888')
            else:                 fr = '#888888'
            fl = f"{tip.upper()} R²={r2s:.2f}"
            self.ax_xp.plot(xi, fp * self.s_gain.val, color=fr,
                            lw=1.0, ls='--', alpha=0.85, label=fl)
            self.ax_xp.legend(fontsize=6, loc='upper right',
                              facecolor='#1a1a2e', edgecolor='#333', labelcolor='#aaa')
        self.ax_xp.axhline(0, color='#444', lw=.5)
        self.ax_xp.axvline(xv, color='#FF4500', lw=.8, ls='--')
        self.ax_xp.set_title("X Kesiti  (kesik=dipol fit)", fontsize=7, color='#aaa')

        # ── Y profil ──────────────────────────────────────────────────────────
        self.ax_yp.clear(); self._ax_karalat(self.ax_yp)
        self.ax_yp.plot(yi, yp, color='#00CFFF', lw=1.2)
        self.ax_yp.axhline(0, color='#444', lw=.5)
        self.ax_yp.axvline(yv, color='#FF4500', lw=.8, ls='--')
        self.ax_yp.set_title("Y Kesiti", fontsize=7, color='#aaa')

        # ── Info paneli ───────────────────────────────────────────────────────
        self.ax_info.clear(); self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0d0d0d')

        def bar(v, n=5):
            d = max(0, min(n, int(v * n)))
            return '█' * d + '░' * (n - d)

        gr = '#00FF9D' if self.targets else '#FF4444'
        self.ax_info.text(.02, .98,
            f"── GURULTU ──\nStd:{self.gurultu_std:.3f}nT  "
            f"Sigma:{int(self.s_sigma.val)}σ  Esik:{self.gurultu_std*self.s_sigma.val:.2f}nT\n"
            f"{'ANOMALI VAR' if self.targets else 'ANOMALI YOK'}",
            color=gr, fontsize=8, family='monospace', va='top')

        self.ax_info.text(.02, .82, "TESHIS:", color='#666', fontsize=7, va='top')
        self.ax_info.text(.02, .77, durum, color=renk, fontsize=10, weight='bold', va='top')
        self.ax_info.text(.02, .71, aciklama, color='#aaa', fontsize=7, va='top', style='italic')

        r2m_v   = r2m if r2m is not None else 0.0
        r2b_v   = r2b if r2b is not None else 0.0
        zstr    = f"{z_d:.2f}" if z_d is not None else "?"
        dp_renk = ('#00FF9D' if tip == 'metal' else
                   '#00CFFF' if tip == 'bosluk' else '#888888')
        rm_renk = ('#00FF9D' if r2m_v >= 0.70 else
                   '#FFD700' if r2m_v >= 0.40 else '#FF4444')
        rb_renk = ('#00CFFF' if r2b_v >= 0.70 else
                   '#88AAFF' if r2b_v >= 0.40 else '#888888')
        rm_str = f"Metal  R2: {bar(r2m_v)} {r2m_v:.2f}" + (" ◄" if tip == 'metal'  else "")
        rb_str = f"Bosluk R2: {bar(r2b_v)} {r2b_v:.2f}" + (" ◄" if tip == 'bosluk' else "")

        self.ax_info.text(.02, .64, "── DIPOL FIT ──",
            color='#aaa', fontsize=7, family='monospace', va='top')
        self.ax_info.text(.02, .60, rm_str,
            color=rm_renk, fontsize=7, family='monospace', va='top',
            weight='bold' if tip == 'metal' else 'normal')
        self.ax_info.text(.02, .56, rb_str,
            color=rb_renk, fontsize=7, family='monospace', va='top',
            weight='bold' if tip == 'bosluk' else 'normal')
        self.ax_info.text(.02, .52,
            f"Karar: {(tip or '?').upper()}  ~{zstr}m",
            color=dp_renk, fontsize=8, family='monospace', va='top', weight='bold')
        self.ax_info.text(.02, .48, (dyorum or '')[:28],
            color='#888', fontsize=7, family='monospace', va='top', style='italic')

        fr2 = ("#00FF9D" if ortusme > 0.80 else
               "#FFD700" if ortusme > 0.50 else "#FF4444")
        sr2 = ("#FF00FF" if (fwhm and fwhm < 0.3) else
               "#FFA500" if (fwhm and fwhm < 0.8) else "#00CFFF")
        self.ax_info.text(.02, .43,
            f"── FAZ KAYMASI ──\n"
            f"Ortusme:%{int(ortusme*100)} {bar(ortusme)}\n"
            f"{faz_y[:22]}",
            color=fr2, fontsize=8, family='monospace', va='top')
        self.ax_info.text(.02, .33,
            f"── TEPE ──\nFWHM:{fwhm:.2f}m\n" if fwhm else "── TEPE ──\nFWHM:?\n",
            color=sr2, fontsize=8, family='monospace', va='top')
        if fwhm:
            self.ax_info.text(.02, .25, siv_y[:24], color=sr2, fontsize=7, va='top')

        # FIX: depth_val (None değil) kullanılıyor
        gbar = bar((guven or 0) / 100)
        gr2  = ('#00FF9D' if (guven or 0) >= 70 else
                '#FFD700' if (guven or 0) >= 40 else '#FF4444')
        sel_sutun = round(xv / self.adim_m) + self.sutun_min
        sel_satir = round(yv / self.adim_m) + self.satir_min
        self.ax_info.text(.52, .98,
            f"── SECILI NOKTA ──\n"
            f"Sutun:{sel_sutun}  Satir:{sel_satir}\n"
            f"({xv:.2f}m, {yv:.2f}m)\n"
            f"Siddet : {val:.2f} nT\n"
            f"Derinlik: ~{depth_val:.2f}m\n"   # FIX: depth_val
            f"Guven  : {gbar} %{guven or '?'}\n"
            f"SNR    : {snr:.1f}\n"
            f"{(ystr or '')[:26]}",
            color=gr2, fontsize=8, family='monospace', va='top')

        self.ax_info.text(.52, .62, "── OBJE TAHMINI ──",
                          color='#FFD700', fontsize=8, family='monospace', va='top')
        for i, (isim, puan, renk2) in enumerate(tahminler):
            pb = bar(puan, 4)
            self.ax_info.text(.52, .56 - i * 0.09,
                f"{pb} %{int(puan*100)}\n{isim[:28]}",
                color=renk2, fontsize=7, family='monospace', va='top')

        max_amp = max((abs(t['amp']) for t in self.targets), default=1.0)
        htxt = "── HEDEFLER ──\n"
        for t in self.targets:
            try:
                # FIX: yi ekseninde t['y'] (satir_m) ara
                ri2 = np.argmin(np.abs(yi - t['y']))
                pt  = self.zi[ri2, :] / (self.s_gain.val or 1.0)
                d2, gv2, _ = self._derinlik_pro(pt, xi)
                _, _, _, _, _, tip2, _ = self._dipol_fit(pt, xi)
                th = {'metal': 'M', 'bosluk': 'B', 'belirsiz': '?'}.get(tip2, '?')
            except Exception:
                d2  = self._derinlik(t['amp'] / (self.s_gain.val or 1.0))
                gv2 = None; th = '?'
            gvs   = f"%{gv2}" if gv2 else "?"
            ab    = bar(abs(t['amp']) / max_amp, 6)
            t_sut = round(t['x'] / self.adim_m) + self.sutun_min
            t_sat = round(t['y'] / self.adim_m) + self.satir_min
            d2v   = float(d2) if d2 is not None else 0.0
            htxt += f"H{t['id']}[{th}] S{t_sut}/R{t_sat} ~{d2v:.2f}m[{gvs}]\n{ab}\n"
        self.ax_info.text(.52, .27, htxt, color='#00FF9D', fontsize=7,
                          family='monospace', va='top')

    # ── 3D görselleştirme ─────────────────────────────────────────────────────
    def _show_3d(self, _):
        if self.zi is None:
            return
        f3  = plt.figure(figsize=(10, 7)); f3.patch.set_facecolor('#0a0a0f')
        ax3 = f3.add_subplot(111, projection='3d'); ax3.set_facecolor('#0a0a0f')
        for p in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            p.fill = False; p.set_edgecolor('#222')
        ax3.tick_params(colors='#777', labelsize=7)

        # FIX: grid_X=sütun(X), grid_Y=satır(Y) — eksenler artık doğru
        surf = ax3.plot_surface(self.grid_x, self.grid_y, self.zi,
                                cmap=C3_CMAP, edgecolor='none', alpha=0.95)
        cb = f3.colorbar(surf, ax=ax3, shrink=0.4, aspect=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')
        cb.set_label('nT', color='#aaa', fontsize=8)
        ax3.set_xlabel('X (m) — Sütun', color='#aaa', fontsize=9, labelpad=8)
        ax3.set_ylabel('Y (m) — Satır', color='#aaa', fontsize=9, labelpad=8)
        ax3.set_zlabel('nT',             color='#aaa', fontsize=9, labelpad=8)

        xi  = self.xi_arr
        yi  = self.yi_arr
        zr  = self.zi.max() - self.zi.min()
        ph  = zr * 0.12

        for t in self.targets:
            try:
                # FIX: yi (satir) ekseninde t['y'] ara
                ri2 = np.argmin(np.abs(yi - t['y']))
                ci2 = np.argmin(np.abs(xi - t['x']))
                pt  = self.zi[ri2, :] / (self.s_gain.val or 1.0)
                _, _, _, _, _, tip2, _ = self._dipol_fit(pt, xi)
                pr  = {'metal': '#FF4444', 'bosluk': '#00CFFF',
                       'belirsiz': '#FFD700'}.get(tip2, '#FFD700')
                zt  = self.zi[ri2, ci2]
                ax3.scatter([t['x']], [t['y']], [zt + ph],
                            color=pr, s=120, marker='^', zorder=10)
                ax3.plot([t['x'], t['x']], [t['y'], t['y']], [zt, zt + ph],
                         color=pr, lw=1.2, alpha=0.8)
                ax3.text(t['x'], t['y'], zt + ph * 1.2, f"H{t['id']}",
                         color='white', fontsize=8, weight='bold', ha='center')
            except Exception:
                pass

        ax3.set_title("C3 3D  (Kirmizi=Metal · Mavi=Bosluk · Sari=Belirsiz)",
                      color='#00FF9D', pad=10)
        plt.show()

    # ── Rapor ─────────────────────────────────────────────────────────────────
    def _rapor_kaydet(self, _):
        # FIX: se/sue None kontrolü — grid oluşturulmamışsa erken çık
        if self.zi is None or self.xi_arr is None or self.yi_arr is None:
            messagebox.showwarning("Uyari", "Önce veri güncelleyin (görüntü oluşturulmalı).")
            return

        now   = datetime.now()
        base  = self.path.parent / f"C3_Rapor_{now.strftime('%Y%m%d_%H%M%S')}"
        xi    = self.xi_arr     # sutun_m (X)
        yi    = self.yi_arr     # satir_m (Y)
        hlist = []

        with open(str(base) + ".txt", 'w', encoding='utf-8') as f:
            f.write(f"C3 MANYETİK GRADİOMETRE RAPORU\n{'='*44}\n")
            f.write(f"Tarih   : {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dosya   : {self.path.name}\n")
            f.write(f"Adim    : {self.adim_m*100:.0f}cm\n")
            f.write(f"Mod     : {self.r_mode.value_selected}\n")
            f.write(f"Sensor  : {SENSOR_MESAFESI}m aralik, {YUKSEKLIK_SABITI}m yukseklik\n\n")
            f.write("HEDEFLER:\n")

            for t in self.targets:
                try:
                    ri2 = np.argmin(np.abs(yi - t['y']))   # FIX: yi (satir)
                    pt  = self.zi[ri2, :] / (self.s_gain.val or 1.0)
                    d2, gv2, _              = self._derinlik_pro(pt, xi)
                    _, r2s2, r2m2, r2b2, _, tip2, _ = self._dipol_fit(pt, xi)
                    tah = obje_tahmini(tip2, r2m2 or 0, r2b2 or 0, None, d2,
                                       t['amp'],
                                       abs(t['amp']) / (self.gurultu_std * self.s_gain.val + 1e-9),
                                       self.r_mode.value_selected)
                except Exception:
                    d2  = self._derinlik(t['amp'] / (self.s_gain.val or 1.0))
                    gv2 = r2m2 = r2b2 = tip2 = None
                    tah = [("?", 0, "")]

                d2v = float(d2) if d2 is not None else 0.0
                hlist.append({
                    'id':         t['id'],
                    'x':          round(t['x'], 3),
                    'y':          round(t['y'], 3),
                    'amp_nT':     round(float(t['amp']), 2),
                    'derinlik_m': round(d2v, 3),
                    'guven_pct':  gv2,
                    'dipol_tip':  tip2,
                    'r2_metal':   round(float(r2m2), 3) if r2m2 is not None else None,
                    'r2_bosluk':  round(float(r2b2), 3) if r2b2 is not None else None,
                    'tahmin_1':   tah[0][0],
                })
                t_sut2  = round(t['x'] / self.adim_m) + self.sutun_min
                t_sat2  = round(t['y'] / self.adim_m) + self.satir_min
                r2m_str = f"{r2m2:.2f}" if r2m2 is not None else "?"
                r2b_str = f"{r2b2:.2f}" if r2b2 is not None else "?"
                f.write(f"  H{t['id']}: Sutun{t_sut2}/Satir{t_sat2} "
                        f"({t['x']:.2f}m,{t['y']:.2f}m) "
                        f"{t['amp']:.1f}nT ~{d2v:.2f}m [{tip2}] "
                        f"R2M={r2m_str} R2B={r2b_str}\n"
                        f"  Tahmin: {tah[0][0]}\n")

            if self.ai_sonuc:
                a = self.ai_sonuc
                f.write(f"\nSECILI NOKTA:\n"
                        f"  Konum:{a['x']:.2f},{a['y']:.2f}  Siddet:{a['val']:.1f}nT\n"
                        f"  Derinlik:~{a.get('depth', 0):.2f}m  Guven:%{a.get('guven','?')}\n"
                        f"  Metal R2:{a.get('r2m','?')}  Bosluk R2:{a.get('r2b','?')}\n"
                        f"  Teshis:{a.get('durum','')}  {a.get('aciklama','')}\n"
                        f"  Tahmin:{a.get('tahminler',[['']])[0][0] if a.get('tahminler') else '?'}\n")
            f.write("\n--- Rapor Sonu ---\n")

        pc = str(base) + "_hedefler.csv"
        with open(pc, 'w', newline='', encoding='utf-8') as f:
            if hlist:
                w = csv.DictWriter(f, fieldnames=hlist[0].keys())
                w.writeheader(); w.writerows(hlist)

        def sf(v):
            return float(v) if isinstance(v, (np.floating, np.integer)) else v

        pj = str(base) + ".json"
        with open(pj, 'w', encoding='utf-8') as f:
            json.dump({
                'tarih':    now.isoformat(),
                'dosya':    self.path.name,
                'adim_cm':  self.adim_m * 100,
                'mod':      self.r_mode.value_selected,
                'hedefler': hlist,
                'secili':   {k: sf(v) for k, v in self.ai_sonuc.items()
                             if not isinstance(v, list)} if self.ai_sonuc else {},
            }, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("Kaydedildi",
            f"3 dosya:\n{Path(str(base)+'.txt').name}\n"
            f"{Path(pc).name}\n{Path(pj).name}")

    # ── Klavye ────────────────────────────────────────────────────────────────
    def _klavye(self, event):
        k = event.key
        if   k == 'r':         self._rapor_kaydet(None)
        elif k == 'g':         self.s_gain.set_val(100)
        elif k in ('+', '='):  self.s_gain.set_val(min(1000, self.s_gain.val * 1.25))
        elif k == '-':         self.s_gain.set_val(max(1, self.s_gain.val * 0.80))
        elif k in ('left', 'right', 'up', 'down'):
            if self.xi_arr is None:
                return
            ax = (self.xi_arr[-1] - self.xi_arr[0]) / 20
            ay = (self.yi_arr[-1] - self.yi_arr[0]) / 20
            dx = ax * (1 if k == 'right' else -1 if k == 'left' else 0)
            dy = ay * (1 if k == 'up'    else -1 if k == 'down'  else 0)
            nx = np.clip(self.sel[0] + dx, self.xi_arr[0], self.xi_arr[-1])
            ny = np.clip(self.sel[1] + dy, self.yi_arr[0], self.yi_arr[-1])
            self.sel = [nx, ny]
            self._profil_ciz(nx, ny)
            self.fig.canvas.draw_idle()

    def _ax_karalat(self, ax):
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='#555', labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222')

    # ── Arayüz ────────────────────────────────────────────────────────────────
    def run(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(17, 9))
        self.fig.patch.set_facecolor('#0a0a0f')
        self.fig.canvas.manager.set_window_title(f'C3 ANALİZ — {self.path.name}')
        self.fig.subplots_adjust(bottom=0.20)

        gs = self.fig.add_gridspec(6, 2, width_ratios=[1.3, 1], hspace=0.6, wspace=0.3)
        self.ax2d   = self.fig.add_subplot(gs[:, 0])
        self.ax_xp  = self.fig.add_subplot(gs[0, 1])
        self.ax_yp  = self.fig.add_subplot(gs[1, 1])
        self.ax_info= self.fig.add_subplot(gs[2:, 1]); self.ax_info.axis('off')

        for ax in [self.ax2d, self.ax_xp, self.ax_yp, self.ax_info]:
            self._ax_karalat(ax)

        sc = {'facecolor': '#1a1a2e'}
        self.s_gain  = Slider(plt.axes([0.05, 0.13, 0.14, 0.025], **sc), 'Kazanc',  1, 1000, valinit=100)
        self.s_esik  = Slider(plt.axes([0.05, 0.08, 0.14, 0.025], **sc), 'Esik',    0,  500, valinit=25)
        self.s_blur  = Slider(plt.axes([0.23, 0.13, 0.14, 0.025], **sc), 'Yumusat', 0,    5, valinit=0.5)
        self.s_noise = Slider(plt.axes([0.23, 0.08, 0.14, 0.025], **sc), 'Parazit', 1,    9, valinit=3, valstep=2)
        self.s_sigma = Slider(plt.axes([0.41, 0.07, 0.04, 0.09],  **sc), 'Sigma',   1,    3, valinit=3, valstep=1, orientation='vertical')

        self.r_mode = RadioButtons(
            plt.axes([0.48, 0.02, 0.12, 0.16], facecolor='#1a1a2e'),
            ('TFA', 'Sadece Z', 'Gradient', 'Analitik', 'FFT Derin', 'FFT Sig'),
            activecolor='#00FF9D')

        self.btn_3d    = Button(plt.axes([0.62, 0.12, 0.10, 0.04]), '3D GOSTER',    color='#1a237e')
        self.btn_rapor = Button(plt.axes([0.62, 0.06, 0.10, 0.04]), 'RAPOR KAYDET', color='#1b5e20')

        for s in [self.s_gain, self.s_esik, self.s_blur, self.s_noise, self.s_sigma]:
            s.on_changed(self._guncelle)
        self.r_mode.on_clicked(self._guncelle)
        self.btn_3d.on_clicked(self._show_3d)
        self.btn_rapor.on_clicked(self._rapor_kaydet)
        self.fig.canvas.mpl_connect('button_press_event', self._tikla)
        self.fig.canvas.mpl_connect('key_press_event', self._klavye)

        self._guncelle()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  BAŞLANGIÇ EKRANI
# ─────────────────────────────────────────────────────────────────────────────
def baslangic_ekrani():
    import tkinter as tk
    root = tk.Tk()
    root.title("C3 ANALİZ — Tarama Ayarlari")
    root.configure(bg='#0a0a0f')
    root.resizable(False, False)
    root.update_idletasks()
    w, h = 380, 270
    root.geometry(f"{w}x{h}+"
                  f"{(root.winfo_screenwidth()-w)//2}+"
                  f"{(root.winfo_screenheight()-h)//2}")

    sonuc = {'path': None, 'adim': None}

    tk.Label(root, text="C3 MANYETİK ANALİZ", bg='#0a0a0f', fg='#00FF9D',
             font=('Courier', 14, 'bold')).pack(pady=(18, 4))
    tk.Label(root, text="Tarama parametrelerini girin", bg='#0a0a0f', fg='#888',
             font=('Courier', 9)).pack()

    dv = tk.StringVar(value="— CSV secilmedi —")

    def csv_sec():
        p = filedialog.askopenfilename(title="C3 CSV Dosyasi",
                                       filetypes=[("CSV", "*.csv"), ("Tumu", "*.*")])
        if p:
            sonuc['path'] = p; dv.set(Path(p).name)

    tk.Button(root, text="CSV Dosyasi Sec", command=csv_sec,
              bg='#1a237e', fg='white', font=('Courier', 10, 'bold'),
              relief='flat', padx=10, pady=6).pack(pady=(14, 4))
    tk.Label(root, textvariable=dv, bg='#0a0a0f', fg='#FFD700',
             font=('Courier', 8)).pack()

    tk.Label(root, text="Sutun/Satir adim mesafesi (cm):",
             bg='#0a0a0f', fg='#aaa', font=('Courier', 9)).pack(pady=(12, 2))

    av = tk.StringVar(value="50")
    af = tk.Frame(root, bg='#0a0a0f'); af.pack()
    for cm, lbl in [("20", "20cm"), ("25", "25cm"), ("50", "50cm")]:
        tk.Radiobutton(af, text=lbl, variable=av, value=cm,
                       bg='#0a0a0f', fg='#aaa', selectcolor='#1a1a2e',
                       activebackground='#0a0a0f', activeforeground='#00FF9D',
                       font=('Courier', 9)).pack(side='left', padx=8)

    mf = tk.Frame(root, bg='#0a0a0f'); mf.pack(pady=2)
    tk.Label(mf, text="veya manuel:", bg='#0a0a0f', fg='#666',
             font=('Courier', 8)).pack(side='left')
    me = tk.Entry(mf, width=5, bg='#1a1a2e', fg='white',
                  font=('Courier', 9), insertbackground='white')
    me.pack(side='left', padx=4)
    tk.Label(mf, text="cm", bg='#0a0a0f', fg='#666',
             font=('Courier', 8)).pack(side='left')

    # FIX: manuel giriş radyo butonunu devre dışı bırakır (kullanıcı niyeti açık)
    def manuel_yazildi(*_):
        if me.get().strip():
            av.set("")   # radyo seçimini temizle

    me.bind("<KeyRelease>", manuel_yazildi)

    def baslat():
        if not sonuc['path']:
            messagebox.showwarning("Uyari", "Lutfen once CSV dosyasi secin."); return
        man = me.get().strip()
        try:
            cm = float(man) if man else float(av.get())
            if cm <= 0: raise ValueError
        except ValueError:
            messagebox.showwarning("Uyari", "Gecerli adim mesafesi girin."); return
        sonuc['adim'] = cm / 100.0
        root.destroy()

    tk.Button(root, text="ANALIZI BASLAT", command=baslat,
              bg='#1b5e20', fg='white', font=('Courier', 11, 'bold'),
              relief='flat', padx=16, pady=8).pack(pady=(12, 0))
    root.mainloop()
    return sonuc['path'], sonuc['adim']


if __name__ == "__main__":
    path, adim = baslangic_ekrani()
    if path and adim:
        C3Analiz(path, adim_m=adim).run()
PYTHON_EOF
print("Yazıldı, satır sayısı:")
wc -l /home/claude/c3_analiz_fixed.py
Output

Yazıldı, satır sayısı:
1163 /home/claude/c3_analiz_fixed.py
Done
