import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter, label, sobel
from scipy.signal import detrend
from datetime import datetime

# ── SABİTLER ──────────────────────────────────────────────
SENSOR_MESAFESI = 0.80
YUKSEKLIK_SABITI = 0.20
MAX_DEPTH = 10.0
GRID_RES = 250

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA','#0066FF','#00CCFF','#00CC44','#FFFF00','#FF6600','#CC0000'
], N=512)

# ── SAYFA AYARI ────────────────────────────────────────────
st.set_page_config(page_title="C3 ANALİZ PRO", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');
body, .stApp { background:#060610; color:#c8d0e0; }
h1,h2,h3 { color:#00FF9D; font-family:'Orbitron',monospace; letter-spacing:2px; }
.stSidebar { background:#08081a; border-right:1px solid #00FF9D22; }
.stSidebar * { font-family:'Share Tech Mono',monospace; }
.metric-box {
    background:linear-gradient(135deg,#0d1117 0%,#111827 100%);
    border:1px solid #00FF9D33; border-top:2px solid #00FF9D;
    border-radius:6px; padding:12px 16px;
    font-family:'Share Tech Mono',monospace; margin-bottom:8px;
}
.metric-label { color:#556; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#00FF9D; font-size:22px; font-weight:bold; margin-top:2px; }
.hedef {
    background:linear-gradient(90deg,#0d1f1a 0%,#0a1410 100%);
    border-left:3px solid #00FF9D; border-bottom:1px solid #00FF9D11;
    padding:10px 14px; margin:6px 0; border-radius:0 6px 6px 0;
    font-family:'Share Tech Mono',monospace; font-size:13px;
}
.uyari {
    background:#150a0a; border:1px solid #FF444433;
    border-left:3px solid #FF4444; padding:12px 16px;
    border-radius:0 6px 6px 0; font-family:'Share Tech Mono',monospace;
}
.anomali {
    background:#0a1a12; border:1px solid #00FF9D33;
    border-left:3px solid #00FF9D; padding:12px 16px;
    border-radius:0 6px 6px 0; font-family:'Share Tech Mono',monospace;
}
.faz-panel {
    background:linear-gradient(135deg,#0d1a2e,#091422);
    border-left:3px solid #00CFFF; border-bottom:1px solid #00CFFF22;
    padding:12px 16px; border-radius:0 6px 6px 0;
    font-family:'Share Tech Mono',monospace; font-size:13px; margin:4px 0;
}
.siv-panel {
    background:linear-gradient(135deg,#1a0d2e,#120920);
    border-left:3px solid #FF00FF; border-bottom:1px solid #FF00FF22;
    padding:12px 16px; border-radius:0 6px 6px 0;
    font-family:'Share Tech Mono',monospace; font-size:13px; margin:4px 0;
}
.ai-panel {
    background:linear-gradient(135deg,#0a1a10,#080f08);
    border:1px solid; border-radius:8px; padding:14px 18px;
    font-family:'Share Tech Mono',monospace; margin-top:10px;
}
.gurultu-panel {
    background:#0d0d1a; border:1px solid #1a1a33; border-radius:8px;
    padding:14px; font-family:'Share Tech Mono',monospace;
    font-size:12px; line-height:1.8;
}
hr { border-color:#1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:8px 0 4px;'>
<span style='font-family:Orbitron,monospace;font-size:22px;color:#00FF9D;letter-spacing:4px;'>
⚡ C3 MANYETİK GRADİOMETRE ANALİZ PRO
</span><br>
<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#334;letter-spacing:2px;'>
TFA · Toplam Alan Şiddeti · Çift Eksen Faz Analizi · Akıllı Tepe Sivrilik
</span>
</div>
""", unsafe_allow_html=True)

# ── SİDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("## 🕹️ Kontrol Paneli")
st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Sinyal**")
gain     = st.sidebar.slider("Kazanç",    1.0, 1000.0, 100.0)
esik     = st.sidebar.slider("Eşik",      0,   500,    25)
blur     = st.sidebar.slider("Yumuşatma", 0.0, 5.0,    0.5)
med_size = st.sidebar.slider("Parazit",   1,   9,      3, step=2)
st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Analiz Modu**")
mode     = st.sidebar.radio("Mod", ("TFA", "Sadece Z", "Gradient", "Analitik"))
st.sidebar.markdown("---")
st.sidebar.markdown("**⚖️ Gürültü Eşiği**")
sigma    = st.sidebar.select_slider("Sigma (σ)", options=[1, 2, 3], value=3,
           help="1σ=Duyarlı · 2σ=Normal · 3σ=Katı")
st.sidebar.markdown("""
<div style='font-size:11px;color:#556;font-family:Share Tech Mono,monospace;
padding:4px;line-height:1.8;'>
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
    # _c3_analiz.py: çift medyan normalize + satır bazlı detrend
    for col in ['tfa_diff', 'z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir'] == r
            if m.sum() > 4:
                df.loc[m, col] = detrend(df.loc[m, col], type='linear')
        df[col] -= df[col].median()   # detrend sonrası ikinci normalize
    return df


def olustur_grid(df, veri_col):
    """
    xi = satır ekseni, yi = sütun ekseni
    meshgrid(xi,yi) → gx shape (len(yi), len(xi))
    zi.T → shape (len(xi), len(yi)) = (satir, sutun)
    grid_x[0,:] = xi = satır değerleri (sütun boyunca sabit)
    grid_y[:,0] = yi = sütun değerleri (satır boyunca sabit)
    Masaüstü _profil_ciz ile aynı mantık korunuyor.
    """
    xi = np.linspace(df['satir'].min(), df['satir'].max(), GRID_RES)
    yi = np.linspace(df['sutun'].min(), df['sutun'].max(), GRID_RES)
    grid_x, grid_y = np.meshgrid(xi, yi)
    # Kazanç grid'e değil, filtre aşamasında uygulanır
    zi = griddata(
        (df['satir'], df['sutun']),
        df[veri_col],
        (grid_x, grid_y),
        method='linear', fill_value=0
    )
    # grid_x[0,:] = xi (satır ekseni)
    # grid_y[:,0] = yi (sütun ekseni)
    return xi, yi, grid_x, grid_y, zi.T


def filtrele(zi, std_noise, gain_val, esik_val, blur_val, med_val, sigma_val, mode_str):
    zi = zi * gain_val   # kazanç normalize veriye uygulanır
    if med_val > 1:
        m = int(med_val); m = m if m % 2 else m + 1
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
        return "TEMİZ / SİNYAL YOK", "#FFFFFF", "Anomali yok."
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
    if ht: return "BOŞLUK/TÜNEL","#00FFFF","Negatif çöküş: Boşluk/tünel."
    if hp: return "YÜZEY METALİ","#FF6B35","Yüzeye yakın metal."
    return "BELİRSİZ","#888888","Form tanımlanamadı."


def faz_kaymasi(prof, esik_val):
    """
    _c3_analiz.py'den — eşik tabanlı maske, None dönüşü.
    Tek profil için hesaplar. Dışarıdan iki profil için iki kez çağrılır.
    """
    try:
        if np.max(np.abs(prof)) < esik_val:
            return None, "Sinyal yok"
        neg_mask = prof < -esik_val * 0.3
        pos_mask = prof >  esik_val * 0.3
        if neg_mask.sum() < 2:
            return None, "Negatif bölge yok — tek kutuplu sinyal"
        if pos_mask.sum() < 2:
            return None, "Pozitif bölge yok — sadece çukur"
        neg_idx = np.where(neg_mask)[0]
        cukur_merkez = int(np.mean(neg_idx))
        analitik = np.sqrt(np.gradient(prof)**2 + prof**2)
        tepe_merkez = int(np.argmax(analitik))
        n = len(prof)
        mesafe = abs(tepe_merkez - cukur_merkez)
        ortusme = max(0.0, 1.0 - mesafe / (n * 0.3))
        if ortusme > 0.80:
            return ortusme, "Manyetik olmayan kütle / boşluk"
        elif ortusme > 0.50:
            return ortusme, "Karışık sinyal / belirsiz"
        else:
            return ortusme, "Demir dipol kalıntısı"
    except Exception:
        return None, "Hesaplanamadı"


def faz_kaymasi_cift(xp, yp, esik_val):
    """
    _c3_analiz.py mantığı: her iki profili dene, anlamlı olanı seç.
    İkisi de anlamlıysa daha yüksek örtüşmeyi kullan.
    """
    ox, yx = faz_kaymasi(xp, esik_val)
    oy, yy = faz_kaymasi(yp, esik_val)
    if ox is not None and oy is not None:
        if ox >= oy:
            return ox, yx
        else:
            return oy, yy
    elif ox is not None:
        return ox, yx
    elif oy is not None:
        return oy, yy
    else:
        return None, yx or yy


def tepe_sivrilik(prof, eks, esik_val):
    """
    _c3_analiz.py'den — eşik kontrolü ekli, None dönüşü tutarlı.
    """
    try:
        if np.max(np.abs(prof)) < esik_val:
            return None, "Sinyal yok", "#888888"
        analitik = np.sqrt(np.gradient(prof)**2 + prof**2)
        tepe_val = analitik.max()
        if tepe_val < 1e-6:
            return None, "Tepe yok", "#888888"
        yari = tepe_val * 0.5
        ustu = analitik >= yari
        idx = np.where(ustu)[0]
        if len(idx) < 2:
            return None, "Tepe çok dar", "#888888"
        adim = (eks[-1] - eks[0]) / len(eks)
        fwhm = (idx[-1] - idx[0]) * adim
        if fwhm < 0.3:
            return fwhm, "Sivri → Küçük yoğun kütle (sikke/kap/obje)", "#FF00FF"
        elif fwhm < 0.8:
            return fwhm, "Orta → Hacimli kütle (sandık/küp)", "#FFA500"
        else:
            return fwhm, "Yayvan → Büyük yapı (duvar/horasan/oda)", "#00CFFF"
    except Exception:
        return None, "Hesaplanamadı", "#888888"


def tepe_sivrilik_akilli(xp, yp, sutun_eks, satir_eks, esik_val):
    """
    _c3_analiz.py mantığı: daha büyük varyasyonlu profili kullan.
    """
    if abs(np.max(xp) - np.min(xp)) >= abs(np.max(yp) - np.min(yp)):
        return tepe_sivrilik(xp, sutun_eks, esik_val)
    else:
        return tepe_sivrilik(yp, satir_eks, esik_val)


def guc_bar(amp, max_amp):
    g = int(abs(amp) / max(max_amp, 1e-5) * 10)
    return '█'*g + '░'*(10-g)


def rapor_olustur(df, targets, std_noise, sigma_val, mode_str, filename,
                  ortusme=None, faz_yorum=None, fwhm=None, siv_form=None, secili=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "="*52,
        "   C3 MANYETİK GRADİOMETRE ANALİZ RAPORU",
        "="*52,
        f"Tarih   : {now}",
        f"Dosya   : {filename}",
        f"Mod     : {mode_str}",
        f"Nokta   : {len(df)}",
        f"Std     : {std_noise:.4f} nT",
        f"Sigma   : {sigma_val}σ  →  Oto Eşik: {std_noise*sigma_val:.4f} nT",
        "","-"*52,"OTOMATİK HEDEFLER:","-"*52,
    ]
    for t in targets:
        d = derinlik(t['amp'])
        lines.append(f"H{t['id']:02d}  Satır:{t['x']:.2f}  Sütun:{t['y']:.2f}  "
                     f"Şiddet:{t['amp']:.1f}nT  ~{d:.2f}m")
    if not targets:
        lines.append("Kayda değer anomali tespit edilmedi.")
    if ortusme is not None:
        lines += ["","-"*52,"FAZ KAYMASI (Çift Eksen):","-"*52,
                  f"Örtüşme : %{int(ortusme*100)}", f"Yorum   : {faz_yorum}"]
    if fwhm is not None:
        lines += ["","-"*52,"TEPE KARAKTERİ:","-"*52,
                  f"Genişlik (FWHM): {fwhm:.2f} m", f"Form    : {siv_form}"]
    if secili:
        lines += ["","-"*52,"H1 ANALİZ DETAYI:","-"*52,
                  f"Konum   : Satır={secili['x']:.2f}  Sütun={secili['y']:.2f}",
                  f"Şiddet  : {secili['val']:.1f} nT",
                  f"Derinlik: ~{secili['depth']:.2f} m",
                  f"Teşhis  : {secili['durum']}",
                  f"Not     : {secili['aciklama']}"]
    lines += ["","="*52,"--- Rapor Sonu ---"]
    return "\n".join(lines)


# ── DOSYA YÜKLEME ───────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 CSV Yükle  (sutun · satir · S1_X · S1_Y · S1_Z · S2_X · S2_Y · S2_Z)",
    type=["csv"]
)

if not uploaded:
    st.markdown("""
    <div style='text-align:center;padding:80px;font-family:Share Tech Mono,monospace;'>
        <div style='font-size:64px;'>📡</div>
        <div style='font-size:16px;margin-top:16px;color:#334;'>CSV dosyasını yükleyin</div>
        <div style='font-size:12px;margin-top:6px;color:#223;'>
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

std_noise = df['tfa_diff'].std()
oto_esik  = std_noise * sigma
veri_col  = 'z_diff' if mode == 'Sadece Z' else 'tfa_diff'

xi, yi, grid_x, grid_y, zi_raw = olustur_grid(df, veri_col)
zi, k_esik = filtrele(zi_raw, std_noise, gain, esik, blur, med_size, sigma, mode)
targets    = hedef_tespit(zi, xi, yi, k_esik)
anomali_var = len(targets) > 0

# Masaüstü _profil_ciz ile aynı eksen mantığı:
# grid_x[0,:] = xi = satır ekseni
# grid_y[:,0] = yi = sütun ekseni
satir_eks = grid_x[0, :]   # satır koordinatları
sutun_eks = grid_y[:, 0]   # sütun koordinatları

# ── ÜST METRİKLER ──────────────────────────────────────────
m1,m2,m3,m4,m5 = st.columns(5)
for col, lbl, val in zip(
    [m1,m2,m3,m4,m5],
    ["📍 Ölçüm Noktası","🎯 Hedef Sayısı","📶 Gürültü Std",
     f"⚖️ {sigma}σ Eşik","🧲 TFA Ort (S1)"],
    [f"{len(df)}", f"{len(targets)}", f"{std_noise:.4f} nT",
     f"{oto_esik:.4f} nT", f"{df['tfa1'].mean():.1f} nT"]
):
    with col:
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>{lbl}</div>
        <div class='metric-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── ANOMALİ DURUM BANNER ────────────────────────────────────
if anomali_var:
    st.markdown(f"""<div class='anomali'>
    ✅ &nbsp;<b>{len(targets)} KAYDA DEĞER ANOMALİ TESPİT EDİLDİ</b>
    &nbsp;|&nbsp; Mod: {mode} &nbsp;|&nbsp; Eşik: {k_esik:.2f} nT ({sigma}σ)
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='uyari'>
    ⚠️ &nbsp;<b>KAYDA DEĞER ANOMALİ YOK</b>
    &nbsp;|&nbsp; Std: {std_noise:.4f} nT &nbsp;|&nbsp; {sigma}σ Eşik: {oto_esik:.4f} nT<br>
    <small>Sigmayı düşür veya kazancı artır. Plastik/beton bu cihazla görünmez.</small>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── 2D HARİTA + HEDEF PANELİ ───────────────────────────────
col_map, col_panel = st.columns([2.4, 1])

with col_map:
    st.subheader(f"📍 2D Anomali Haritası — {mode}")
    fig2d, ax2d = plt.subplots(figsize=(10, 7))
    fig2d.patch.set_facecolor('#060610')
    ax2d.set_facecolor('#0d0d14')

    zmin, zmax = zi.min(), zi.max()
    if zmin < 0 < zmax:
        nonzero = zi[zi != 0]
        p_hi = np.percentile(np.abs(nonzero), 98) if len(nonzero) > 0 else max(abs(zmin), abs(zmax))
        sym  = max(p_hi, 0.001)
        norm = TwoSlopeNorm(vmin=-sym, vcenter=0, vmax=sym)
    else:
        norm = Normalize(vmin=zmin, vmax=zmax)

    # extent=[sutun_min, sutun_max, satir_min, satir_max]
    # Masaüstü ile aynı: extent=[yi.min(), yi.max(), xi.min(), xi.max()]
    im = ax2d.imshow(zi, extent=[yi.min(), yi.max(), xi.min(), xi.max()],
                     origin='lower', cmap=C3_CMAP, norm=norm,
                     aspect='auto', interpolation='bilinear')
    cb = plt.colorbar(im, ax=ax2d, fraction=0.03)
    cb.set_label('nT  (🔵Boşluk · 🟢Zemin · 🔴Metal)', color='#aaa', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#555', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

    # Hedef işaretleri: t['y']=sütun(x ekseni), t['x']=satır(y ekseni) — masaüstü ile aynı
    for t in targets:
        ax2d.plot(t['y'], t['x'], 'w+', ms=14, mew=2)
        ax2d.text(t['y']+0.05, t['x']+0.05, f"H{t['id']}",
                  color='white', fontsize=10, weight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

    ax2d.plot(df['sutun'].min(), df['satir'].min(), '*', color='yellow', ms=14, zorder=5)
    ax2d.text(df['sutun'].min(), df['satir'].min(), ' ★BAŞLANGIÇ',
              color='yellow', fontsize=8, weight='bold')
    ax2d.set_xlabel("Sütun →", color='#556', fontsize=9)
    ax2d.set_ylabel("Satır ↑", color='#556', fontsize=9)
    ax2d.tick_params(colors='#444')
    for sp in ax2d.spines.values(): sp.set_edgecolor('#1a1a2e')
    plt.tight_layout()
    st.pyplot(fig2d)

with col_panel:
    st.subheader("🎯 Hedef Listesi")
    if not targets:
        st.warning("Anomali yok. Sigmayı düşür veya kazancı artır.")
    else:
        max_amp = max(abs(t['amp']) for t in targets)
        for t in targets:
            d   = derinlik(t['amp'])
            bar = guc_bar(t['amp'], max_amp)
            st.markdown(f"""<div class='hedef'>
            <b style='color:#00FF9D;font-size:15px;'>H{t['id']}</b>
            &nbsp; Satır:{t['x']:.1f} · Sütun:{t['y']:.1f}<br>
            Şiddet &nbsp;: <b>{t['amp']:.1f} nT</b><br>
            Derinlik: <b>~{d:.2f} m</b><br>
            Güç &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:#FFD700;'>[{bar}]</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div class='gurultu-panel'>
    <b style='color:#00FF9D;'>── GÜRÜLTÜ ANALİZİ ──</b><br>
    Std &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {std_noise:.4f} nT<br>
    Sigma &nbsp;&nbsp;&nbsp;: {sigma}σ<br>
    Oto Eşik : {oto_esik:.4f} nT<br>
    Durum &nbsp;&nbsp;&nbsp;: <b style='color:{"#00FF9D" if anomali_var else "#FF4444"}'>
    {"✓ ANOMALİ VAR" if anomali_var else "✗ YOK"}</b>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── KESİT PROFİLLERİ + FAZ + SİVRİLİK ─────────────────────
# Masaüstü _profil_ciz mantığı — H1 hedefini otomatik seç
ortusme_r = faz_yorum_r = fwhm_r = siv_form_r = siv_renk_r = None
secili_r  = None

if targets:
    st.subheader("📈 Kesit Profilleri — H1 (En Güçlü Hedef)")
    t0 = targets[0]

    # Masaüstü _profil_ciz ile AYNI eksen mantığı:
    # satir_eks = grid_x[0,:] = xi  (yatay meshgrid satırı = satır koordinatları)
    # sutun_eks = grid_y[:,0] = yi  (dikey meshgrid sütunu = sütun koordinatları)
    # t0['x'] = satır koordinatı → satir_eks üzerinde ara
    # t0['y'] = sütun koordinatı → sutun_eks üzerinde ara
    ri = np.argmin(np.abs(satir_eks - t0['x']))   # satır indeksi
    ci = np.argmin(np.abs(sutun_eks - t0['y']))   # sütun indeksi
    ri = int(np.clip(ri, 0, zi.shape[0]-1))
    ci = int(np.clip(ci, 0, zi.shape[1]-1))

    xp = zi[ri, :]    # sabit satırda sütun boyunca profil (Doğu-Batı)
    yp = zi[:, ci]    # sabit sütunda satır boyunca profil (Kuzey-Güney)
    val0 = zi[ri, ci]

    durum, renk, aciklama = ai_yorum(xp, val0, k_esik, mode)
    depth0 = derinlik(val0)
    secili_r = {'x':t0['x'],'y':t0['y'],'val':val0,
                'depth':depth0,'durum':durum,'aciklama':aciklama}

    # _c3_analiz.py: çift eksen faz kayması
    ortusme_r, faz_yorum_r = faz_kaymasi_cift(xp, yp, k_esik)
    faz_renk_r = ("#00CFFF" if ortusme_r is not None and ortusme_r > 0.80 else
                  "#FFD700" if ortusme_r is not None and ortusme_r > 0.50 else
                  "#FF4444" if ortusme_r is not None else "#888888")

    # _c3_analiz.py: daha büyük varyasyonlu profil
    fwhm_r, siv_form_r, siv_renk_r = tepe_sivrilik_akilli(xp, yp, sutun_eks, satir_eks, k_esik)

    cp1, cp2 = st.columns(2)
    with cp1:
        fig_xp, ax_xp = plt.subplots(figsize=(6, 2.5))
        fig_xp.patch.set_facecolor('#060610')
        ax_xp.set_facecolor('#0d0d14')
        ax_xp.plot(sutun_eks, xp, color='#FFD700', lw=1.5)
        ax_xp.axhline(0, color='#333', lw=0.5)
        ax_xp.axvline(t0['y'], color='#FF4500', lw=1, ls='--')
        ax_xp.set_title("Sütun Kesiti (Doğu-Batı)", color='#00FF9D', fontsize=9)
        ax_xp.set_xlabel("Sütun →", color='#556', fontsize=8)
        ax_xp.tick_params(colors='#444', labelsize=7)
        for sp in ax_xp.spines.values(): sp.set_edgecolor('#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig_xp)

    with cp2:
        fig_yp, ax_yp = plt.subplots(figsize=(6, 2.5))
        fig_yp.patch.set_facecolor('#060610')
        ax_yp.set_facecolor('#0d0d14')
        ax_yp.plot(satir_eks, yp, color='#00CFFF', lw=1.5)
        ax_yp.axhline(0, color='#333', lw=0.5)
        ax_yp.axvline(t0['x'], color='#FF4500', lw=1, ls='--')
        ax_yp.set_title("Satır Kesiti (Kuzey-Güney)", color='#00FF9D', fontsize=9)
        ax_yp.set_xlabel("Satır ↑", color='#556', fontsize=8)
        ax_yp.tick_params(colors='#444', labelsize=7)
        for sp in ax_yp.spines.values(): sp.set_edgecolor('#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig_yp)

    # AI Teşhis
    st.markdown(f"""<div class='ai-panel' style='border-color:{renk}44;'>
    <span style='color:#556;font-size:11px;letter-spacing:1px;'>🤖 AI TEŞHİS — H1</span><br>
    <span style='color:{renk};font-size:18px;font-weight:bold;'>{durum}</span><br>
    <span style='color:#aaa;font-size:12px;font-style:italic;'>{aciklama}</span><br>
    <span style='color:#445;font-size:11px;'>
    Konum: Satır={t0['x']:.2f} · Sütun={t0['y']:.2f}
    &nbsp;|&nbsp; Şiddet: {val0:.1f} nT &nbsp;|&nbsp; Derinlik: ~{depth0:.2f} m
    </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    fa_col, siv_col = st.columns(2)
    with fa_col:
        if ortusme_r is not None:
            guc_dolu = int(ortusme_r * 5)
            guc_bar_str = "●" * guc_dolu + "○" * (5 - guc_dolu)
            st.markdown(f"""<div class='faz-panel'>
            <b style='color:#00CFFF;'>── FAZ KAYMASI (Çift Eksen) ──</b><br>
            Örtüşme &nbsp;: <b style='color:{faz_renk_r};'>%{int(ortusme_r*100)}</b><br>
            Güven &nbsp;&nbsp;&nbsp;: <span style='color:{faz_renk_r};letter-spacing:3px;'>{guc_bar_str}</span><br>
            Yorum &nbsp;&nbsp;&nbsp;: <span style='color:{faz_renk_r};'>{faz_yorum_r}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='faz-panel'>
            <b style='color:#00CFFF;'>── FAZ KAYMASI (Çift Eksen) ──</b><br>
            <span style='color:#556;'>{faz_yorum_r or "Dipol yok"}</span>
            </div>""", unsafe_allow_html=True)

    with siv_col:
        fwhm_str = f"{fwhm_r:.2f} m" if fwhm_r is not None else "—"
        st.markdown(f"""<div class='siv-panel'>
        <b style='color:#FF00FF;'>── TEPE KARAKTERİ (Akıllı Eksen) ──</b><br>
        Genişlik : <b style='color:{siv_renk_r};'>{fwhm_str}</b><br>
        Form &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{siv_renk_r};'>{siv_form_r}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── 3D İNTERAKTİF ──────────────────────────────────────────
st.subheader("🧊 3D İnteraktif Topografya")
st.caption("💡 Mouse ile döndür · Tekerlek ile yakınlaş · Hover ile değer gör")

fig3d = go.Figure(data=[go.Surface(
    z=zi, x=yi, y=xi,
    colorscale=[
        [0.0,'#0000AA'],[0.2,'#0066FF'],[0.35,'#00CCFF'],
        [0.5,'#00CC44'],[0.65,'#FFFF00'],[0.8,'#FF6600'],[1.0,'#CC0000']
    ],
    opacity=0.95
)])
if targets:
    fig3d.add_trace(go.Scatter3d(
        x=[t['y'] for t in targets], y=[t['x'] for t in targets],
        z=[t['amp'] for t in targets],
        mode='markers+text',
        marker=dict(size=6, color='white', symbol='cross'),
        text=[f"H{t['id']}" for t in targets],
        textfont=dict(color='white', size=11), name='Hedefler'
    ))
fig3d.add_trace(go.Scatter3d(
    x=[df['sutun'].min()], y=[df['satir'].min()], z=[0],
    mode='markers+text', marker=dict(size=8, color='yellow'),
    text=['★BAŞLANGIÇ'], textfont=dict(color='yellow', size=10),
    name='Başlangıç'
))
fig3d.update_layout(
    scene=dict(
        xaxis_title='SUTUN (Doğu →)', yaxis_title='SATIR (Kuzey ↑)',
        zaxis_title='Şiddet (nT)',
        xaxis=dict(backgroundcolor='#060610', gridcolor='#1a1a2e', color='#556'),
        yaxis=dict(backgroundcolor='#060610', gridcolor='#1a1a2e', color='#556'),
        zaxis=dict(backgroundcolor='#060610', gridcolor='#1a1a2e', color='#556'),
        aspectmode='manual', aspectratio=dict(x=1.2, y=1.2, z=0.45)
    ),
    paper_bgcolor='#060610',
    margin=dict(l=0,r=0,b=0,t=0), height=600
)
st.plotly_chart(fig3d, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── RAPOR İNDİR ────────────────────────────────────────────
st.subheader("💾 Rapor")
rapor = rapor_olustur(df, targets, std_noise, sigma, mode, uploaded.name,
                      ortusme_r, faz_yorum_r, fwhm_r, siv_form_r, secili_r)
st.download_button(
    "📄 TXT Raporu İndir",
    data=rapor.encode('utf-8'),
    file_name=f"C3_Rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)
