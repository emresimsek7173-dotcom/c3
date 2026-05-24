import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from datetime import datetime
import io

# ── SABİTLER ──────────────────────────────────────────────
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
MAX_DEPTH = 10.0
GRID_RES = 220

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA','#0066FF','#00CCFF','#00CC44','#FFFF00','#FF6600','#CC0000'
], N=512)

# ── SAYFA AYARI ────────────────────────────────────────────
st.set_page_config(page_title="C3 ANALİZ PRO", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
body, .stApp { background:#0a0a0f; color:#e0e0e0; }
h1,h2,h3 { color:#00FF9D; font-family:'Courier New',monospace; }
.stSidebar { background:#0d0d1a; }
.metric { background:#111827; border:1px solid #00FF9D33; border-radius:8px; padding:10px; font-family:monospace; }
.hedef { background:#0d1f1a; border-left:3px solid #00FF9D; padding:8px 12px; margin:4px 0; border-radius:4px; font-family:monospace; font-size:13px; }
.uyari { background:#1a0a0a; border-left:3px solid #FF4444; padding:10px; border-radius:4px; font-family:monospace; }
.anomali { background:#0a1a0a; border-left:3px solid #00FF9D; padding:10px; border-radius:4px; font-family:monospace; }
.faz-panel { background:#0d1a2e; border-left:3px solid #00CFFF; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:13px; margin:4px 0; }
.siv-panel { background:#1a0d2e; border-left:3px solid #FF00FF; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:13px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;letter-spacing:3px;'>⚡ C3 MANYETİK GRADİOMETRE ANALİZ PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;font-family:monospace;'>TFA · Toplam Alan Şiddeti · 3 Sigma Gürültü Filtresi</p>", unsafe_allow_html=True)

# ── SİDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("## 🕹️ Kontrol Paneli")
st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Sinyal**")
gain     = st.sidebar.slider("Kazanç",   1.0, 1000.0, 100.0)
esik     = st.sidebar.slider("Eşik",     0,   500,    20)
blur     = st.sidebar.slider("Yumuşatma",0.0, 5.0,    0.5)
med_size = st.sidebar.slider("Parazit",  0,   9,      3, step=2)
st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Mod**")
mode     = st.sidebar.radio("Analiz Modu", ("TFA", "Sadece Z", "Gradient", "Analitik"))
st.sidebar.markdown("---")
st.sidebar.markdown("**⚖️ Gürültü Eşiği**")
sigma    = st.sidebar.select_slider("Sigma (σ)", options=[1, 2, 3], value=3,
           help="1σ=Duyarlı(mahzen/boşluk) · 2σ=Normal · 3σ=Katı(sadece güçlü)")
st.sidebar.markdown(f"""
<div style='font-size:11px;color:#888;font-family:monospace;padding:4px;'>
1σ → Mahzen, boşluk araması<br>
2σ → Genel kullanım<br>
3σ → Sadece güçlü anomaliler
</div>
""", unsafe_allow_html=True)

# ── YARDIMCI FONKSİYONLAR ──────────────────────────────────
def veri_isle(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace('ı','i').str.replace('ş','s')
                  .str.replace('ğ','g').str.replace('ü','u')
                  .str.replace('ö','o').str.replace('ç','c'))
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    df['tfa1'] = np.sqrt(df['s1_x']**2 + df['s1_y']**2 + df['s1_z']**2)
    df['tfa2'] = np.sqrt(df['s2_x']**2 + df['s2_y']**2 + df['s2_z']**2)
    for col in ['tfa1','tfa2','s1_z','s2_z']:
        lo, hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo, hi)
    df['tfa_diff'] = df['tfa1'] - df['tfa2']
    df['z_diff']   = df['s1_z'] - df['s2_z']
    for col in ['tfa_diff','z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir'] == r
            if m.sum() > 5:
                df.loc[m, col] = detrend(df.loc[m, col], type='linear')
    return df

def olustur_grid(df, veri_col, gain_val):
    xi = np.linspace(df['satir'].min(), df['satir'].max(), GRID_RES)
    yi = np.linspace(df['sutun'].min(), df['sutun'].max(), GRID_RES)
    gx, gy = np.meshgrid(xi, yi)
    zi = griddata((df['satir'], df['sutun']), df[veri_col]*gain_val,
                  (gx, gy), method='linear', fill_value=0)
    return xi, yi, gx, gy, zi.T

def filtrele(zi, std_noise, gain_val, esik_val, blur_val, med_val, sigma_val, mode_str):
    if med_val > 1:
        m = int(med_val); m = m if m%2 else m+1
        zi = median_filter(zi, size=m)
    oto_esik = std_noise * sigma_val * gain_val
    kullanilan_esik = max(oto_esik, esik_val)
    zi = np.where(np.abs(zi) < kullanilan_esik, 0, zi)
    if blur_val > 0:
        zi = gaussian_filter(zi, sigma=blur_val)
    if mode_str == 'Gradient':
        zi = np.sqrt(sobel(zi,1)**2 + sobel(zi,0)**2)
    elif mode_str == 'Analitik':
        zi = np.sqrt(sobel(zi,1)**2 + sobel(zi,0)**2 + zi**2)
    return zi, kullanilan_esik

def hedef_tespit(zi, xi, yi, kullanilan_esik):
    binary = np.abs(zi) > kullanilan_esik
    labeled, num = label(binary)
    targets = []
    for i in range(1, num+1):
        mask = labeled == i
        if mask.sum() < 4: continue
        coords = np.argwhere(mask)
        peak = np.argmax(np.abs(zi[mask]))
        py, px = coords[peak]
        py, px = min(py, GRID_RES-1), min(px, GRID_RES-1)
        targets.append({'id':i, 'x':xi[min(py,len(xi)-1)],
                        'y':yi[min(px,len(yi)-1)], 'amp':zi[py,px]})
    return sorted(targets, key=lambda t: abs(t['amp']), reverse=True)[:6]

def derinlik(peak_nt):
    grad = abs(peak_nt) / SENSOR_MESAFESI
    if grad < 0.005: return 0.0
    return min(YUKSEKLIK_SABITI + 2.5*abs(peak_nt)/(grad+0.08), MAX_DEPTH)

def ai_yorum(x_prof, val, esik_val, mode_str):
    if abs(val) < esik_val:
        return "TEMİZ / SİNYAL YOK", "#AAAAAA", "Anomali yok."
    if mode_str == 'Analitik':
        return "ENERJİ MERKEZİ", "#FF00FF", "Hedefin odak noktası."
    if mode_str == 'Gradient':
        return "KENAR / SINIR", "#FFA500", "Anomali sınırı."
    vmax, vmin = float(np.max(x_prof)), float(np.min(x_prof))
    vr = max(vmax-vmin, 1e-5)
    if abs(x_prof[0]) > abs(val)*0.88 or abs(x_prof[-1]) > abs(val)*0.88:
        return ("KENAR/DEĞERLİ?","#FFA500","Sınırda kesilmiş. Alanı büyüt!") \
               if vmin < 0 else ("KENAR/METAL?","#FF8C00","Sınırda metal sinyali.")
    hp, ht = vmax > vr*0.15, vmin < -vr*0.15
    if hp and ht:
        return ("DEĞERLİ/DOLU YAPI","#FFD700","Zıplama+çukur: Hacimli yapı!") \
               if abs(vmin)>vmax else ("METAL/DİPOL","#FF4500","Net metal kütlesi.")
    if ht: return "BOŞLUK/TÜNEL","#00CFFF","Negatif çöküş: Boşluk/tünel."
    if hp: return "YÜZEY METALİ","#FF6B35","Yüzeye yakın metal."
    return "BELİRSİZ","#888888","Form tanımlanamadı."

def faz_kaymasi(x_prof):
    """Raw mavi çukur merkezi ile Analitik tepe örtüşmesini hesaplar."""
    try:
        neg_mask = x_prof < 0
        if neg_mask.sum() < 2:
            return 0.0, "Negatif bölge yok", "#888888"
        neg_idx = np.where(neg_mask)[0]
        cukur_merkez = int(np.mean(neg_idx))
        analitik = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tepe_merkez = int(np.argmax(analitik))
        n = len(x_prof)
        mesafe = abs(tepe_merkez - cukur_merkez)
        ortusme = max(0.0, 1.0 - mesafe / (n * 0.3))
        if ortusme > 0.80:
            yorum = "Manyetik olmayan kütle / boşluk"
            renk = "#00CFFF"
        elif ortusme > 0.50:
            yorum = "Karışık sinyal / belirsiz"
            renk = "#FFD700"
        else:
            yorum = "Demir dipol kalıntısı"
            renk = "#FF4444"
        return ortusme, yorum, renk
    except Exception:
        return 0.0, "Hesaplanamadı", "#888888"

def tepe_sivrilik(x_prof, xi):
    """FWHM (yarı yükseklik genişliği) ile tepe karakterini ölçer."""
    try:
        analitik = np.sqrt(np.gradient(x_prof)**2 + x_prof**2)
        tepe_val = analitik.max()
        if tepe_val < 1e-6:
            return None, "Tepe yok", "#888888"
        yari = tepe_val * 0.5
        ustu = analitik >= yari
        idx = np.where(ustu)[0]
        if len(idx) < 2:
            return None, "Tepe çok dar", "#888888"
        adim = (xi[-1] - xi[0]) / len(xi)
        fwhm = (idx[-1] - idx[0]) * adim
        if fwhm < 0.3:
            form = "Sivri → Küçük yoğun kütle (sikke/kap/obje)"
            renk = "#FF00FF"
        elif fwhm < 0.8:
            form = "Orta → Hacimli kütle (sandık/küp)"
            renk = "#FFA500"
        else:
            form = "Yayvan → Büyük yapı (duvar/horasan/oda)"
            renk = "#00CFFF"
        return fwhm, form, renk
    except Exception:
        return None, "Hesaplanamadı", "#888888"

def guc_bar(amp, max_amp):
    g = int(abs(amp) / max(max_amp,1e-5) * 10)
    return '█'*g + '░'*(10-g)

def rapor_olustur(df, targets, std_noise, sigma_val, mode_str, filename,
                  ortusme=None, faz_yorum=None, fwhm=None, siv_form=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "="*50,
        "  C3 MANYETİK GRADİOMETRE ANALİZ RAPORU",
        "="*50,
        f"Tarih   : {now}",
        f"Dosya   : {filename}",
        f"Mod     : {mode_str}",
        f"Nokta   : {len(df)}",
        f"Std     : {std_noise:.4f} nT",
        f"Sigma   : {sigma_val}σ  →  Oto Eşik: {std_noise*sigma_val:.4f} nT",
        "",
        "-"*50,
        "OTOMATİK HEDEFLER:",
        "-"*50,
    ]
    for t in targets:
        d = derinlik(t['amp'])
        lines.append(f"H{t['id']:02d}  Satır:{t['x']:.2f}  Sütun:{t['y']:.2f}  Şiddet:{t['amp']:.1f}nT  ~{d:.2f}m")
    if not targets:
        lines.append("Kayda değer anomali tespit edilmedi.")
    if ortusme is not None:
        lines += [
            "",
            "-"*50,
            "H1 FAZ KAYMASI ANALİZİ:",
            "-"*50,
            f"Örtüşme : %{int(ortusme*100)}",
            f"Yorum   : {faz_yorum}",
        ]
    if fwhm is not None:
        lines += [
            "",
            "-"*50,
            "H1 TEPE KARAKTERİ:",
            "-"*50,
            f"Genişlik (FWHM): {fwhm:.2f} m",
            f"Form    : {siv_form}",
        ]
    lines += ["","="*50,"--- Rapor Sonu ---"]
    return "\n".join(lines)

# ── DOSYA YÜKLEME ───────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 CSV Yükle  (sutun · satir · S1_X · S1_Y · S1_Z · S2_X · S2_Y · S2_Z)",
    type=["csv"]
)

if not uploaded:
    st.markdown("""
    <div style='text-align:center;padding:80px;color:#333;font-family:monospace;'>
        <div style='font-size:56px;'>📡</div>
        <div style='font-size:16px;margin-top:12px;'>CSV dosyasını yükleyin</div>
        <div style='font-size:12px;margin-top:6px;color:#222;'>
            sutun · satir · S1_X · S1_Y · S1_Z · S2_X · S2_Y · S2_Z
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── VERİ İŞLE ──────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded)
    df = veri_isle(df_raw.copy())
    eksik = [c for c in ['satir','sutun','s1_x','s1_y','s1_z','s2_x','s2_y','s2_z']
             if c not in df.columns]
    if eksik:
        st.error(f"Eksik sütunlar: {eksik}  |  Mevcut: {list(df.columns)}")
        st.stop()
except Exception as e:
    st.error(f"❌ Dosya okunamadı: {e}")
    st.stop()

std_noise   = df['tfa_diff'].std()
oto_esik    = std_noise * sigma
veri_col    = 'z_diff' if mode == 'Sadece Z' else 'tfa_diff'
xi, yi, gx, gy, zi_raw = olustur_grid(df, veri_col, gain)
zi, k_esik  = filtrele(zi_raw, std_noise, gain, esik, blur, med_size, sigma, mode)
targets     = hedef_tespit(zi, xi, yi, k_esik)
anomali_var = len(targets) > 0

# ── ÜST METRİKLER ──────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📍 Ölçüm", f"{len(df)}")
c2.metric("🎯 Hedef", f"{len(targets)}")
c3.metric("📶 Gürültü Std", f"{std_noise:.3f} nT")
c4.metric(f"⚖️ {sigma}σ Eşik", f"{oto_esik:.3f} nT")
c5.metric("🧲 TFA Ort S1", f"{df['tfa1'].mean():.1f} nT")

st.markdown("---")

# ── GÜRÜLTÜ DURUMU ─────────────────────────────────────────
if anomali_var:
    st.markdown(f"""<div class='anomali'>
    ✅  <b>{len(targets)} KAYDA DEĞER ANOMALİ TESPİT EDİLDİ</b> &nbsp;|&nbsp;
    Mod: {mode} &nbsp;|&nbsp; Eşik: {k_esik:.2f} nT ({sigma}σ)
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='uyari'>
    ⚠️  <b>KAYDA DEĞER ANOMALİ YOK</b> &nbsp;|&nbsp;
    Gürültü: {std_noise:.3f} nT &nbsp;|&nbsp; {sigma}σ Eşik: {oto_esik:.3f} nT<br>
    <small>Sigmayı düşür veya kazancı artır. Plastik/beton bu cihazla görünmez.</small>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── 2D HARİTA + HEDEF PANELİ ───────────────────────────────
col_map, col_panel = st.columns([2.2, 1])

with col_map:
    st.subheader(f"📍 2D Anomali Haritası — {mode}")
    fig2d, ax2d = plt.subplots(figsize=(10,7))
    fig2d.patch.set_facecolor('#0a0a0f')
    ax2d.set_facecolor('#0d0d0d')

    zmin, zmax = zi.min(), zi.max()
    if zmin < 0 < zmax:
        nonzero = zi[zi != 0]
        p_hi = np.percentile(np.abs(nonzero), 98) if len(nonzero) > 0 else max(abs(zmin), abs(zmax))
        sym = max(p_hi, 0.001)
        norm = TwoSlopeNorm(vmin=-sym, vcenter=0, vmax=sym)
    else:
        norm = Normalize(vmin=zmin, vmax=zmax)

    im = ax2d.imshow(zi, extent=[yi.min(), yi.max(), xi.min(), xi.max()],
                     origin='lower', cmap=C3_CMAP, norm=norm,
                     aspect='auto', interpolation='bilinear')
    cb = plt.colorbar(im, ax=ax2d, fraction=0.03)
    cb.set_label('nT  (🔵Boşluk · 🟢Zemin · 🔴Metal)', color='#aaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#555', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

    for t in targets:
        ax2d.plot(t['y'], t['x'], 'w+', ms=14, mew=2)
        ax2d.text(t['y']+0.05, t['x']+0.05, f"H{t['id']}",
                  color='white', fontsize=10, weight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

    ax2d.plot(df['sutun'].min(), df['satir'].min(), '*', color='yellow', ms=14, zorder=5)
    ax2d.text(df['sutun'].min(), df['satir'].min(), ' ★BAŞLANGIÇ', color='yellow', fontsize=8, weight='bold')
    ax2d.set_xlabel("Sütun →", color='#666', fontsize=9)
    ax2d.set_ylabel("Satır ↑", color='#666', fontsize=9)
    ax2d.tick_params(colors='#444')
    for sp in ax2d.spines.values(): sp.set_edgecolor('#222')
    plt.tight_layout()
    st.pyplot(fig2d)

with col_panel:
    st.subheader("🎯 Hedef Listesi")
    if not targets:
        st.warning("Anomali yok. Sigmayı düşür veya kazancı artır.")
    else:
        max_amp = max(abs(t['amp']) for t in targets)
        for t in targets:
            d = derinlik(t['amp'])
            bar = guc_bar(t['amp'], max_amp)
            st.markdown(f"""<div class='hedef'>
            <b style='color:#00FF9D;font-size:15px;'>H{t['id']}</b>
            &nbsp; Satır:{t['x']:.1f} · Sütun:{t['y']:.1f}<br>
            Şiddet : <b>{t['amp']:.1f} nT</b><br>
            Derinlik: <b>~{d:.2f} m</b><br>
            Güç : <span style='color:#FFD700;'>[{bar}]</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div style='font-family:monospace;font-size:12px;
    background:#0d0d1a;padding:10px;border-radius:6px;border:1px solid #333;'>
    <b style='color:#00FF9D;'>── GÜRÜLTÜ ANALİZİ ──</b><br>
    Std &nbsp;&nbsp;&nbsp;&nbsp;: {std_noise:.4f} nT<br>
    Sigma &nbsp;: {sigma}σ<br>
    Oto Eşik: {oto_esik:.4f} nT<br>
    Durum &nbsp;: <b style='color:{"#00FF9D" if anomali_var else "#FF4444"}'>
    {"✓ ANOMALİ VAR" if anomali_var else "✗ YOK"}</b>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── KESİT PROFİLLERİ + FAZ + SİVRİLİK ─────────────────────
ortusme_r = faz_yorum_r = fwhm_r = siv_form_r = siv_renk_r = faz_renk_r = None

if targets:
    st.subheader("📈 Kesit Profilleri — H1 (En Güçlü Hedef)")
    t0 = targets[0]
    ri = np.argmin(np.abs(xi - t0['x']))
    ci = np.argmin(np.abs(yi - t0['y']))
    x_prof = zi[ri, :]
    y_prof = zi[:, ci]
    val0   = zi[ri, ci]

    durum, renk, aciklama = ai_yorum(x_prof, val0, k_esik, mode)

    # Faz kayması ve sivrilik hesapla
    ortusme_r, faz_yorum_r, faz_renk_r = faz_kaymasi(x_prof)
    fwhm_r, siv_form_r, siv_renk_r     = tepe_sivrilik(x_prof, yi)

    cp1, cp2 = st.columns(2)
    with cp1:
        fig_xp, ax_xp = plt.subplots(figsize=(6,2.5))
        fig_xp.patch.set_facecolor('#0a0a0f')
        ax_xp.set_facecolor('#0d0d0d')
        ax_xp.plot(yi, x_prof, color='#FFD700', lw=1.5)
        ax_xp.axhline(0, color='#333', lw=0.5)
        ax_xp.axvline(t0['y'], color='#FF4500', lw=1, ls='--')
        ax_xp.set_title("Sütun Kesiti (Doğu-Batı)", color='#00FF9D', fontsize=9)
        ax_xp.set_xlabel("Sütun →", color='#666', fontsize=8)
        ax_xp.tick_params(colors='#444', labelsize=7)
        for sp in ax_xp.spines.values(): sp.set_edgecolor('#222')
        plt.tight_layout()
        st.pyplot(fig_xp)

    with cp2:
        fig_yp, ax_yp = plt.subplots(figsize=(6,2.5))
        fig_yp.patch.set_facecolor('#0a0a0f')
        ax_yp.set_facecolor('#0d0d0d')
        ax_yp.plot(xi, y_prof, color='#00CFFF', lw=1.5)
        ax_yp.axhline(0, color='#333', lw=0.5)
        ax_yp.axvline(t0['x'], color='#FF4500', lw=1, ls='--')
        ax_yp.set_title("Satır Kesiti (Kuzey-Güney)", color='#00FF9D', fontsize=9)
        ax_yp.set_xlabel("Satır ↑", color='#666', fontsize=8)
        ax_yp.tick_params(colors='#444', labelsize=7)
        for sp in ax_yp.spines.values(): sp.set_edgecolor('#222')
        plt.tight_layout()
        st.pyplot(fig_yp)

    # AI Teşhis
    st.markdown(f"""<div style='padding:12px;background:#0a1a0f;
    border:1px solid {renk}55;border-radius:8px;font-family:monospace;margin-top:8px;'>
    <span style='color:#888;font-size:12px;'>🤖 AI TEŞHİS (H1)</span><br>
    <span style='color:{renk};font-size:16px;font-weight:bold;'>{durum}</span><br>
    <span style='color:#aaa;font-size:12px;font-style:italic;'>{aciklama}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Faz Kayması + Tepe Sivrilik yan yana
    fa_col, siv_col = st.columns(2)

    with fa_col:
        guc_dolu = int(ortusme_r * 5)
        guc_bar_str = "●" * guc_dolu + "○" * (5 - guc_dolu)
        st.markdown(f"""<div class='faz-panel'>
        <b style='color:#00CFFF;'>── FAZ KAYMASI ANALİZİ ──</b><br>
        Örtüşme &nbsp;: <b style='color:{faz_renk_r};'>%{int(ortusme_r*100)}</b><br>
        Güven &nbsp;&nbsp;&nbsp;: <span style='color:{faz_renk_r};letter-spacing:2px;'>{guc_bar_str}</span><br>
        Yorum &nbsp;&nbsp;&nbsp;: <span style='color:{faz_renk_r};'>{faz_yorum_r}</span>
        </div>""", unsafe_allow_html=True)

    with siv_col:
        fwhm_str = f"{fwhm_r:.2f} m" if fwhm_r is not None else "—"
        st.markdown(f"""<div class='siv-panel'>
        <b style='color:#FF00FF;'>── TEPE KARAKTERİ ──</b><br>
        Genişlik : <b style='color:{siv_renk_r};'>{fwhm_str}</b><br>
        Form &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{siv_renk_r};'>{siv_form_r}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── 3D İNTERAKTİF ──────────────────────────────────────────
st.subheader("🧊 3D İnteraktif Topografya")
st.caption("💡 Mouse ile döndür · Tekerlek ile yakınlaş · Hover ile değer gör")

fig3d = go.Figure(data=[go.Surface(
    z=zi, x=yi, y=xi,
    colorscale=[
        [0.0,  '#0000AA'], [0.2,  '#0066FF'],
        [0.35, '#00CCFF'], [0.5,  '#00CC44'],
        [0.65, '#FFFF00'], [0.8,  '#FF6600'],
        [1.0,  '#CC0000']
    ],
    opacity=0.95
)])

if targets:
    fig3d.add_trace(go.Scatter3d(
        x=[t['y'] for t in targets],
        y=[t['x'] for t in targets],
        z=[t['amp'] for t in targets],
        mode='markers+text',
        marker=dict(size=6, color='white', symbol='cross'),
        text=[f"H{t['id']}" for t in targets],
        textfont=dict(color='white', size=11),
        name='Hedefler'
    ))

fig3d.add_trace(go.Scatter3d(
    x=[df['sutun'].min()], y=[df['satir'].min()], z=[0],
    mode='markers+text',
    marker=dict(size=8, color='yellow'),
    text=['★BAŞLANGIÇ'], textfont=dict(color='yellow', size=10),
    name='Başlangıç'
))

fig3d.update_layout(
    scene=dict(
        xaxis_title='SUTUN (Doğu →)',
        yaxis_title='SATIR (Kuzey ↑)',
        zaxis_title='Şiddet (nT)',
        xaxis=dict(backgroundcolor='#0a0a0f', gridcolor='#1a1a2e', color='#666'),
        yaxis=dict(backgroundcolor='#0a0a0f', gridcolor='#1a1a2e', color='#666'),
        zaxis=dict(backgroundcolor='#0a0a0f', gridcolor='#1a1a2e', color='#666'),
        aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.45)
    ),
    paper_bgcolor='#0a0a0f',
    margin=dict(l=0,r=0,b=0,t=0),
    height=600
)
st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")

# ── RAPOR İNDİR ────────────────────────────────────────────
st.subheader("💾 Rapor")
rapor = rapor_olustur(df, targets, std_noise, sigma, mode, uploaded.name,
                      ortusme_r, faz_yorum_r, fwhm_r, siv_form_r)
st.download_button(
    "📄 TXT Raporu İndir",
    data=rapor.encode('utf-8'),
    file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)
