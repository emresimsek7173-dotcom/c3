import unicodedata, io, json, csv
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

# ── SABİTLER ──────────────────────────────────────────────────────────────────
SENSOR_MESAFESI  = 0.80
YUKSEKLIK_SABITI = 0.05
MAX_DEPTH        = 5.0
DEFAULT_GRID_RES = 120

C3_CMAP = LinearSegmentedColormap.from_list('c3', [
    '#0000AA','#0066FF','#00CCFF','#00CC44',
    '#FFFF00','#FF6600','#CC0000'], N=512)

# ── YARDIMCI ──────────────────────────────────────────────────────────────────
def _nc(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    return s.strip().lower().replace(' ','_')

def bar5(v):
    d = max(0, min(5, int(v*5)))
    return '█'*d + '░'*(5-d)

# ── OBJE TAHMİN MOTORU ────────────────────────────────────────────────────────
def obje_tahmini(tip, r2m, r2b, fwhm, depth, val, snr, mod):
    T = []
    d = depth or 0.0; fw = fwhm or 0.5; am = abs(val)
    if mod in ('Gradient','Analitik'):
        return [("Kenar/Sınır anomalisi", 0.50, '#FFA500')]
    if tip == 'metal' and r2m >= 0.35:
        if fw < 0.25 and d < 0.5:   T.append(("Küçük metal obje (sikke/parça)", r2m*0.90+(1-d)*0.10, '#FF6600'))
        elif fw < 0.7 and d < 1.2:  T.append(("Metal küp / sandık / kap", r2m*0.85, '#FF4500'))
        elif fw >= 0.7 and d < 1.5: T.append(("Metal boru / ray / levha", r2m*0.80, '#FF2200'))
        elif d >= 1.5:               T.append(("Derin metal yapı", r2m*0.70, '#FF0000'))
        T.append(("Metal obje", r2m*0.60, '#FF8C00'))
    if tip == 'bosluk' and r2b >= 0.35:
        if fw < 0.5 and d < 0.8:
            T.append(("Çömlek / pişmiş toprak kap", r2b*0.85, '#00CFFF'))
            T.append(("Küçük boşluk / hava cebi",   r2b*0.70, '#00AACC'))
        elif fw < 1.0 and d < 1.5:
            T.append(("Çömlek küp / seramik obje",  r2b*0.80, '#00CFFF'))
            T.append(("Küçük yapı boşluğu",          r2b*0.65, '#0088AA'))
        else:
            T.append(("Tünel / oda / mezar odası",   r2b*0.75, '#0066FF'))
            T.append(("Büyük boşluk yapısı",         r2b*0.60, '#0044CC'))
        T.append(("Boşluk / içi dolu olmayan yapı",  r2b*0.50, '#88AAFF'))
    if tip == 'belirsiz' or (r2m>0.30 and r2b>0.30 and abs(r2m-r2b)<0.20):
        if fw < 0.6 and d < 1.0:
            T.append(("Çömlek (metal içerikli?)", 0.50, '#FFD700'))
            T.append(("Pişmiş kil / karma malzeme", 0.45, '#FFA500'))
        else:
            T.append(("Karma yapı / belirsiz", 0.30, '#888888'))
    if snr < 2.0 or am < 0.5:
        T.append(("Zemin gürültüsü / mineral iz", 0.40, '#666666'))
    if not T:
        return [("Tanımlanamadı", 0.0, '#555555')]
    T.sort(key=lambda x: x[1], reverse=True)
    return T[:3]

# ── VERİ ──────────────────────────────────────────────────────────────────────
def veri_yukle(file_bytes, adim_m):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [_nc(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    df['tfa1'] = np.sqrt(df['s1_x']**2+df['s1_y']**2+df['s1_z']**2)
    df['tfa2'] = np.sqrt(df['s2_x']**2+df['s2_y']**2+df['s2_z']**2)
    for col in ['tfa1','tfa2','s1_z','s2_z']:
        lo,hi = df[col].quantile(0.02), df[col].quantile(0.98)
        df[col] = df[col].clip(lo,hi)
    df['tfa_diff'] = df['tfa1']-df['tfa2']
    df['z_diff']   = df['s1_z']-df['s2_z']
    for col in ['tfa_diff','z_diff']:
        df[col] -= df[col].median()
        for r in df['satir'].unique():
            m = df['satir']==r
            if m.sum()>4:
                from scipy.signal import detrend as _dt
                df.loc[m,col] = _dt(df.loc[m,col], type='linear')
        df[col] -= df[col].median()
    df['sutun_m'] = (df['sutun']-df['sutun'].min())*adim_m
    df['satir_m'] = (df['satir']-df['satir'].min())*adim_m

    n_sat = int(df['satir'].nunique())
    n_sut = int(df['sutun'].nunique())
    meta = {
        'n_satir': n_sat, 'n_sutun': n_sut,
        'sutun_min': int(df['sutun'].min()), 'satir_min': int(df['satir'].min()),
        'sutun_max': int(df['sutun'].max()), 'satir_max': int(df['satir'].max()),
        'gurultu_std': df['tfa_diff'].std(), 'adim_m': adim_m,
        'alan_x': n_sut * adim_m,   # ← n×adım = toplam mesafe
        'alan_y': n_sat * adim_m,
        'grid_res': max(20, min(DEFAULT_GRID_RES, max(n_sat,n_sut)*4)),
    }
    return df, meta

def grid_olustur(df, vcol, grid_res):
    xi = np.linspace(df['sutun_m'].min(), df['sutun_m'].max(), grid_res)
    yi = np.linspace(df['satir_m'].min(), df['satir_m'].max(), grid_res)
    gX,gY = np.meshgrid(xi,yi)
    zi = griddata((df['sutun_m'],df['satir_m']), df[vcol], (gX,gY),
                  method='linear', fill_value=0)
    return xi, yi, gX, gY, zi

def fft_filtre(zi, mod):
    F = np.fft.fftshift(np.fft.fft2(zi))
    rows,cols = zi.shape
    u = np.fft.fftshift(np.fft.fftfreq(cols))
    v = np.fft.fftshift(np.fft.fftfreq(rows))
    UU,VV = np.meshgrid(u,v)
    R = np.sqrt(UU**2+VV**2)
    def han(r,fc,bw=0.04):
        m = np.ones_like(r)
        lo,hi = fc-bw, fc+bw
        tr = (r>lo)&(r<hi)
        m[tr] = 0.5*(1+np.cos(np.pi*(r[tr]-lo)/bw))
        m[r>=hi] = 0.0
        return m
    f = han(R,0.08) if mod=='derin' else 1.0-han(R,0.10)
    return np.real(np.fft.ifft2(np.fft.ifftshift(F*f)))

def filtrele(zi, gain, med, blur, sigma, esik_m, gstd, mod):
    zi = zi*gain
    if med>1:
        s=med if med%2 else med+1
        zi = median_filter(zi,size=s)
    if blur>0: zi = gaussian_filter(zi,sigma=blur)
    oto  = gstd*sigma*gain
    esik = max(oto, esik_m)
    zi   = np.where(np.abs(zi)<esik, 0, zi)
    if   mod=='Gradient':  zi=np.sqrt(sobel(zi,1)**2+sobel(zi,0)**2)
    elif mod=='Analitik':  zi=np.sqrt(sobel(zi,1)**2+sobel(zi,0)**2+zi**2)
    elif mod=='FFT Derin': zi=fft_filtre(zi,'derin')
    elif mod=='FFT Sig':   zi=fft_filtre(zi,'sig')
    return zi, esik

def hedef_tespit(zi, xi, yi, f_esik, gstd, gain):
    h_esik = max(f_esik*0.60, gstd*0.8*gain)
    labeled,num = label(np.abs(zi)>h_esik)
    rows,cols = zi.shape
    T=[]
    for i in range(1,num+1):
        mask = labeled==i
        if mask.sum()<2: continue
        coords = np.argwhere(mask)
        pk = np.argmax(np.abs(zi[mask]))
        py,px = coords[pk]
        py,px = min(py,rows-1), min(px,cols-1)
        T.append({'id':i,'x':xi[px],'y':yi[py],'amp':zi[py,px]})
    return sorted(T,key=lambda t:abs(t['amp']),reverse=True)[:8]

# ── ANALİZ FONKSİYONLARI ──────────────────────────────────────────────────────
def derinlik_simple(pnt):
    if abs(pnt)<1e-9: return 0.0
    return min(max(0.0, SENSOR_MESAFESI*0.5-YUKSEKLIK_SABITI), MAX_DEPTH)

def derinlik_pro(profil, eks, gstd):
    try:
        if len(profil)<4 or (eks[-1]-eks[0])<0.01:
            return derinlik_simple(np.max(np.abs(profil))), None, "Profil yetersiz"
        adim = (eks[-1]-eks[0])/max(len(eks)-1,1)
        abs_p = np.abs(profil); tv=abs_p.max(); snr=tv/(gstd+1e-9)
        sonuclar=[]; yon=[]
        idx_h = np.where(abs_p>=tv*0.5)[0]
        if len(idx_h)>=2:
            wh = (idx_h[-1]-idx_h[0])*adim
            if wh>1e-6:
                dp=min(max(0.0,wh*0.5-YUKSEKLIK_SABITI),MAX_DEPTH)
                sonuclar.append((dp, min(snr/10,1)*0.4+0.2)); yon.append(f"P½={dp:.2f}m")
        try:
            g=np.gradient(abs_p,adim); mg=np.max(np.abs(g))
            if mg>1e-9 and tv>1e-9:
                dg=min(max(0.0,tv/(2*mg+1e-9)-YUKSEKLIK_SABITI),MAX_DEPTH)
                sonuclar.append((dg, min(snr/12,1)*0.35+0.15)); yon.append(f"Eğim={dg:.2f}m")
        except: pass
        try:
            x=eks-eks[np.argmax(abs_p)]; g=np.gradient(abs_p,adim)
            m=abs_p>tv*0.15
            if m.sum()>=4:
                A=np.column_stack([x[m],np.ones(m.sum())]); b=-g[m]*x[m]+abs_p[m]
                sol,*_=np.linalg.lstsq(A,b,rcond=None)
                de=min(max(0.0,abs(sol[0])-YUKSEKLIK_SABITI),MAX_DEPTH)
                sonuclar.append((de, min(snr/15,1)*0.25+0.10)); yon.append(f"Euler={de:.2f}m")
        except: pass
        if not sonuclar: return derinlik_simple(tv), None, "Yeterli veri yok"
        tw=sum(w for _,w in sonuclar)
        df_=min(max(0.0,sum(d*w for d,w in sonuclar)/tw),MAX_DEPTH)
        if len(sonuclar)>=2:
            vals=[d for d,_ in sonuclar]
            tut=1.0-min(np.std(vals)/(np.mean(vals)+0.01),1.0)
            guven=int((tut*0.7+min(snr/15,1)*0.3)*100)
        else:
            guven=max(20,int(min(snr/20,1)*50))
        return df_, guven, " | ".join(yon)
    except:
        return derinlik_simple(np.max(np.abs(profil)) if len(profil)>0 else 0.0), None, "Hata"

def dipol_fit(profil, eks):
    try:
        if len(profil)<5: return None,None,None,None,None,None,"Profil yetersiz"
        x=eks-np.mean(eks); amp=float(profil[np.argmax(np.abs(profil))])
        if abs(amp)<1e-9: return None,None,None,None,None,None,"Sinyal yok"
        def Mm(x,M,z,x0):
            xc=x-x0; z2=max(abs(z),0.01)**2
            return M*(2*z2-xc**2)/(xc**2+z2)**2.5
        def Bm(x,K,z,x0):
            xc=x-x0; z2=max(abs(z),0.01)**2
            return -abs(K)*abs(z)/(xc**2+z2)**1.5
        def r2(g,t): return max(0.0,1.0-np.sum((g-t)**2)/(np.sum((g-np.mean(g))**2)+1e-12))
        r2m,zm,Mv,fitm=0.0,None,None,None
        try:
            po,_=curve_fit(Mm,x,profil,p0=[amp*0.1,0.3,0.0],
                bounds=([-abs(amp)*10,0.01,-2.0],[abs(amp)*10,MAX_DEPTH,2.0]),maxfev=2000,ftol=1e-6)
            Mv,zfm,_=po; fitm=Mm(x,*po); r2m=r2(profil,fitm)
            zm=max(0.0,abs(zfm)-YUKSEKLIK_SABITI)
        except: pass
        r2b,zb,fitb=0.0,None,None
        try:
            po,_=curve_fit(Bm,x,profil,p0=[abs(amp)*0.5,0.3,0.0],
                bounds=([0.0,0.01,-2.0],[abs(amp)*20,MAX_DEPTH,2.0]),maxfev=2000,ftol=1e-6)
            fitb=Bm(x,*po); r2b=r2(profil,fitb); zb=max(0.0,abs(po[1])-YUKSEKLIK_SABITI)
        except: pass
        if Mv is not None and Mv<0 and r2m>0.60:
            if fitb is None: fitb,r2b,zb=fitm,r2m,zm
            r2m,zm,Mv=0.0,None,None
        m_ok=r2m>=0.35 and Mv is not None and Mv>0
        b_ok=r2b>=0.35
        if not m_ok and not b_ok:
            return (zm or zb or 0.0),max(r2m,r2b),r2m,r2b,(fitm or fitb),'belirsiz',"Gürültü/zemin"
        tip='metal' if (m_ok and (not b_ok or r2m>=r2b)) else 'bosluk'
        if tip=='metal':
            zs,r2s,fs=zm,r2m,fitm
            yorum=("Güçlü metal dipol" if r2s>=0.85 else "Metal kütle orta güven" if r2s>=0.60 else "Zayıf metal")
        else:
            zs=zb or zm; r2s=r2b if b_ok else r2m; fs=fitb or fitm
            yorum=("Güçlü boşluk/tünel" if r2s>=0.75 else "Olası boşluk" if r2s>=0.50 else "Zayıf boşluk")
        return (zs or 0.0),r2s,r2m,r2b,fs,tip,yorum
    except:
        return None,None,None,None,None,None,"Hesaplanamadı"

def faz_kaymasi(xp, yp):
    try:
        neg=xp<0
        if neg.sum()<2: return 0.0,"Negatif bölge yok"
        cukur=int(np.mean(np.where(neg)[0]))
        ana=np.sqrt(np.gradient(xp)**2+xp**2)
        tepe=int(np.argmax(ana))
        ortusme=max(0.0,1.0-abs(tepe-cukur)/(len(xp)*0.3))
        y=("Manyetik olmayan/boşluk" if ortusme>0.80 else "Karışık sinyal" if ortusme>0.50 else "Demir dipol")
        return ortusme,y
    except: return 0.0,"Hesaplanamadı"

def tepe_sivrilik(xp, xi):
    try:
        ana=np.sqrt(np.gradient(xp)**2+xp**2); tv=ana.max()
        if tv<1e-6: return None,"Tepe yok"
        idx=np.where(ana>=tv*0.5)[0]
        if len(idx)<2: return None,"Tepe dar"
        fwhm=(idx[-1]-idx[0])*(xi[-1]-xi[0])/max(len(xi)-1,1)
        form=("Sivri → küçük yoğun" if fwhm<0.3 else "Orta → hacimli" if fwhm<0.8 else "Yayvan → büyük yapı")
        return fwhm,form
    except: return None,"Hesaplanamadı"

def teshis(xp, val, tip, r2m, r2b, ortusme, fwhm, esik_m, mod):
    if abs(val)<esik_m: return "TEMİZ / SİNYAL YOK","#AAAAAA","Anomali yok."
    if mod=='Analitik': return "ENERJİ MERKEZİ","#FF00FF","Hedefin odak noktası."
    if mod=='Gradient': return "KENAR / SINIR","#FFA500","Anomali sınırı."
    vmax=float(np.max(xp)); vmin=float(np.min(xp)); vr=max(vmax-vmin,1e-5)
    if abs(xp[0])>abs(val)*0.88 or abs(xp[-1])>abs(val)*0.88:
        return "KENAR / DEĞERLİ?","#FFA500","Sınırda kesilmiş — alanı büyüt!"
    mo=bo=blo=0.0
    hp=vmax>vr*0.15; ht=vmin<-vr*0.15
    if hp and ht:
        if abs(vmin)>vmax: mo+=0.25
        else: bo+=0.15; mo+=0.10
    elif ht: bo+=0.25
    elif hp: mo+=0.20; blo+=0.05
    else: blo+=0.25
    r2m_=r2m or 0.0; r2b_=r2b or 0.0
    mo+=0.40*r2m_; bo+=0.40*r2b_; blo+=0.10*max(0.0,0.35-max(r2m_,r2b_))
    if ortusme is not None:
        if ortusme>0.80: bo+=0.20
        elif ortusme>0.50: blo+=0.20
        else: mo+=0.20
    if fwhm is not None:
        if fwhm<0.3: mo+=0.15
        elif fwhm<0.8: mo+=0.08; blo+=0.07
        else: bo+=0.10; blo+=0.05
    t=mo+bo+blo+1e-9
    pm,pb,pbl=mo/t,bo/t,blo/t
    if abs(pm-pb)<0.15 and max(pm,pb)>0.30:
        return "KARMA / ÇELİŞKİLİ","#FFD700",f"Metal%{int(pm*100)} Boşluk%{int(pb*100)}"
    kaz=max([('metal',pm),('bosluk',pb),('belirsiz',pbl)],key=lambda x:x[1])
    if kaz[0]=='metal':
        if pm>0.70: return "METAL KÜTLE","#FF4500",f"Güçlü metal (%{int(pm*100)})"
        return "MUHTEMEL METAL","#FF8C00",f"Metal ihtimali %{int(pm*100)}"
    elif kaz[0]=='bosluk':
        if pb>0.70: return "BOŞLUK / TÜNEL","#00FFFF",f"Güçlü boşluk (%{int(pb*100)})"
        return "MUHTEMEL BOŞLUK","#88AAFF",f"Boşluk ihtimali %{int(pb*100)}"
    return "BELİRSİZ","#888888","Yetersiz sinyal"

# ── BİR HEDEFİN TAM ANALİZİ ──────────────────────────────────────────────────
def hedef_analiz(t, zi, zi_raw, xi, yi, gain, gstd, esik_m, mod):
    ci=np.argmin(np.abs(xi-t['x'])); ri=np.argmin(np.abs(yi-t['y']))
    xp = zi_raw[ri,:]*gain; yp = zi_raw[:,ci]*gain
    val= float(zi[ri,ci])
    if abs(val)<abs(xp[ci])*0.1: val=float(xp[ci])
    xn = zi_raw[ri,:]

    depth,guven,ystr     = derinlik_pro(xn,xi,gstd)
    ortusme,faz_y        = faz_kaymasi(xp,yp)
    fwhm,siv_y           = tepe_sivrilik(xp,xi)
    z_d,r2s,r2m,r2b,fp,tip,dyorum = dipol_fit(xn,xi)
    snr=abs(val)/(gstd*gain+1e-9)

    depth_val=float(depth) if depth is not None else 0.0
    if z_d is not None and r2s is not None and tip=='metal' and r2s>=0.60:
        wd=r2s*0.5
        if depth is not None and guven is not None:
            depth_val=(depth_val*0.6+z_d*wd)/(0.6+wd)
            guven=min(100,int(guven*0.7+r2s*30))
            ystr=(ystr or "")+f" | Dipol={z_d:.2f}m"
        else:
            depth_val,guven=z_d,int(r2s*80); ystr=f"Dipol={z_d:.2f}m"
    elif z_d is not None and r2s is not None and tip=='bosluk' and r2s>=0.50:
        ystr=(ystr or "")+f" | Boşluk={z_d:.2f}m"

    durum,renk,aciklama = teshis(xp,val,tip,r2m,r2b,ortusme,fwhm,esik_m,mod)
    tahminler = obje_tahmini(tip,r2m or 0,r2b or 0,fwhm,depth_val,val,snr,mod)

    return dict(val=val,depth=depth_val,guven=guven,ystr=ystr,
                ortusme=ortusme,faz_y=faz_y,fwhm=fwhm,siv_y=siv_y,
                r2m=r2m,r2b=r2b,r2s=r2s,tip=tip,dyorum=dyorum,snr=snr,
                durum=durum,renk=renk,aciklama=aciklama,tahminler=tahminler,
                xp=xp,yp=yp,fp=fp,xi=xi,yi=yi)

# ── GRAFİKLER ─────────────────────────────────────────────────────────────────
def fig_harita(zi, xi, yi, targets, meta, mod, sel_id):
    plt.style.use('dark_background')
    fig,ax=plt.subplots(figsize=(7,6))
    fig.patch.set_facecolor('#0a0a0f'); ax.set_facecolor('#0d0d0d')
    zmin,zmax=zi.min(),zi.max()
    if zmin<0<zmax:
        nz=zi[zi!=0]
        ph=np.percentile(np.abs(nz),98) if len(nz)>0 else max(abs(zmin),abs(zmax))
        norm=TwoSlopeNorm(vmin=-max(ph,0.001),vcenter=0,vmax=max(ph,0.001))
    else:
        norm=Normalize(vmin=zmin,vmax=zmax)
    im=ax.imshow(zi,extent=[xi.min(),xi.max(),yi.min(),yi.max()],
                 origin='lower',cmap=C3_CMAP,norm=norm,aspect='equal',interpolation='bilinear')
    cb=fig.colorbar(im,ax=ax,shrink=0.75,pad=0.02)
    cb.set_label('nT',color='#aaa',fontsize=8); plt.setp(cb.ax.yaxis.get_ticklabels(),color='#aaa')

    adim=meta['adim_m']
    for t in targets:
        seçili = (t['id']==sel_id)
        col='#FF4500' if seçili else 'white'
        ms=16 if seçili else 12; mew=3 if seçili else 2
        ax.plot(t['x'],t['y'],'+',color=col,ms=ms,mew=mew,zorder=10)
        ax.text(t['x']+0.02,t['y']+0.02,f"H{t['id']}",color=col,
                fontsize=9 if seçili else 8,weight='bold',zorder=11)

    # başlangıç yıldızı
    ax.plot(0,0,'*',color='yellow',ms=12,zorder=5)
    ax.text(0.02,0.02,'★BAŞLANGIÇ',color='yellow',fontsize=7,weight='bold')

    # eksen etiketleri: Satır/Sütun numaraları
    sut_nums=list(range(meta['sutun_min'],meta['sutun_max']+1))
    sut_m   =[(s-meta['sutun_min'])*adim for s in sut_nums]
    sat_nums=list(range(meta['satir_min'],meta['satir_max']+1))
    sat_m   =[(s-meta['satir_min'])*adim for s in sat_nums]
    ax.set_xticks(sut_m); ax.set_xticklabels([f"S{s}" for s in sut_nums],fontsize=7,color='#aaa')
    ax.set_yticks(sat_m); ax.set_yticklabels([f"R{s}" for s in sat_nums],fontsize=7,color='#aaa')
    ax.set_xlabel(f"SÜTUN  ({meta['n_sutun']} sütun × {adim*100:.0f}cm = {meta['alan_x']:.1f}m)",fontsize=8,color='#aaa')
    ax.set_ylabel(f"SATIR  ({meta['n_satir']} satır × {adim*100:.0f}cm = {meta['alan_y']:.1f}m)",fontsize=8,color='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.tick_params(colors='#555')
    br='#00FF9D' if targets else '#FF4444'
    ax.set_title(f"C3 — {mod}  |  {len(targets)} Hedef" if targets else f"C3 — {mod}  |  ⚠ ANOMALİ YOK",
                 color=br,fontsize=11,pad=8)
    plt.tight_layout(pad=0.5)
    return fig

def fig_profil(a):
    plt.style.use('dark_background')
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,3.8))
    fig.patch.set_facecolor('#0a0a0f')
    for ax,xdata,ydata,xref,col,ttl in [
        (ax1,a['xi'],a['xp'],None,'#FFD700',"X Kesiti"),
        (ax2,a['yi'],a['yp'],None,'#00CFFF',"Y Kesiti"),
    ]:
        ax.set_facecolor('#0d0d0d')
        ax.plot(xdata,ydata,color=col,lw=1.3)
        ax.axhline(0,color='#444',lw=0.5)
        for sp in ax.spines.values(): sp.set_edgecolor('#222')
        ax.tick_params(colors='#555',labelsize=7)
        ax.set_title(ttl,fontsize=8,color='#aaa')
    # dipol fit sadece X'te
    if a['fp'] is not None and a['r2s'] is not None:
        tip=a['tip'] or '?'
        r2s=a['r2s']
        fc=('#00FF9D' if (tip=='metal' and r2s>=0.80) else
            '#00CFFF' if (tip=='bosluk' and r2s>=0.75) else '#888888')
        ax1.plot(a['xi'],a['fp']*a['xp'].max()/(np.max(np.abs(a['fp']))+1e-9)*
                 (1 if a['xp'].max()>0 else -1),
                 color=fc,lw=1.0,ls='--',alpha=0.85,label=f"{tip.upper()} R²={r2s:.2f}")
        ax1.legend(fontsize=7,facecolor='#1a1a2e',edgecolor='#333',labelcolor='#aaa')
    plt.tight_layout(pad=0.4)
    return fig

def fig_3d(zi, gX, gY, xi, yi, targets, gain):
    plt.style.use('dark_background')
    fig=plt.figure(figsize=(8,5)); fig.patch.set_facecolor('#0a0a0f')
    ax=fig.add_subplot(111,projection='3d'); ax.set_facecolor('#0a0a0f')
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor('#222')
    ax.tick_params(colors='#777',labelsize=7)
    surf=ax.plot_surface(gX,gY,zi,cmap=C3_CMAP,edgecolor='none',alpha=0.95)
    cb=fig.colorbar(surf,ax=ax,shrink=0.4,aspect=8)
    cb.set_label('nT',color='#aaa',fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(),color='#aaa')
    ax.set_xlabel('X(m)—Sütun',color='#aaa',fontsize=8,labelpad=6)
    ax.set_ylabel('Y(m)—Satır', color='#aaa',fontsize=8,labelpad=6)
    ax.set_zlabel('nT',          color='#aaa',fontsize=8,labelpad=6)
    zr=zi.max()-zi.min(); ph=zr*0.12
    for t in targets:
        try:
            ri2=np.argmin(np.abs(yi-t['y'])); ci2=np.argmin(np.abs(xi-t['x']))
            pt=zi[ri2,:]/(gain or 1.0)
            _,_,_,_,_,tip2,_=dipol_fit(pt,xi)
            pr={'metal':'#FF4444','bosluk':'#00CFFF','belirsiz':'#FFD700'}.get(tip2,'#FFD700')
            zt=zi[ri2,ci2]
            ax.scatter([t['x']],[t['y']],[zt+ph],color=pr,s=100,marker='^',zorder=10)
            ax.text(t['x'],t['y'],zt+ph*1.2,f"H{t['id']}",color='white',fontsize=8,weight='bold',ha='center')
        except: pass
    ax.set_title("3D  (Kırmızı=Metal · Mavi=Boşluk · Sarı=Belirsiz)",color='#00FF9D',pad=8)
    plt.tight_layout()
    return fig

# ── RAPOR ─────────────────────────────────────────────────────────────────────
def rapor_uret(meta, mod, targets, zi, xi, yi, gain, gstd, dosya_adi, esik_m):
    now=datetime.now(); hlist=[]; lines=[]
    lines+=[f"C3 MANYETİK GRADİOMETRE RAPORU","="*44,
            f"Tarih   : {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dosya   : {dosya_adi}",
            f"Adım    : {meta['adim_m']*100:.0f}cm",
            f"Alan    : {meta['alan_x']:.1f}m × {meta['alan_y']:.1f}m",
            f"Mod     : {mod}","","HEDEFLER:"]
    for t in targets:
        try:
            ri2=np.argmin(np.abs(yi-t['y']))
            pt=zi[ri2,:]/(gain or 1.0)
            d2,gv2,_ =derinlik_pro(pt,xi,gstd)
            _,r2s2,r2m2,r2b2,_,tip2,_ =dipol_fit(pt,xi)
            tah=obje_tahmini(tip2,r2m2 or 0,r2b2 or 0,None,d2,
                             t['amp'],abs(t['amp'])/(gstd*gain+1e-9),mod)
        except:
            d2=derinlik_simple(t['amp']/(gain or 1.0))
            gv2=r2m2=r2b2=tip2=None; tah=[("?",0,"")]
        d2v=float(d2) if d2 else 0.0
        ts=round(t['x']/meta['adim_m'])+meta['sutun_min']
        tr=round(t['y']/meta['adim_m'])+meta['satir_min']
        lines.append(f"  H{t['id']}: Sütun{ts}/Satır{tr} ({t['x']:.2f}m,{t['y']:.2f}m) "
                     f"{t['amp']:.1f}nT ~{d2v:.2f}m [{tip2}] "
                     f"R²M={r2m2:.2f if r2m2 else '?'} R²B={r2b2:.2f if r2b2 else '?'}")
        lines.append(f"  Tahmin: {tah[0][0]}")
        hlist.append({'id':t['id'],'x':round(t['x'],3),'y':round(t['y'],3),
                      'amp_nT':round(float(t['amp']),2),'derinlik_m':round(d2v,3),
                      'guven_pct':gv2,'tip':tip2,
                      'r2_metal':round(float(r2m2),3) if r2m2 else None,
                      'r2_bosluk':round(float(r2b2),3) if r2b2 else None,
                      'tahmin_1':tah[0][0]})
    lines+=["","--- Rapor Sonu ---"]
    txt="\n".join(lines)
    csv_buf=io.StringIO()
    if hlist:
        w=csv.DictWriter(csv_buf,fieldnames=hlist[0].keys())
        w.writeheader(); w.writerows(hlist)
    def sf(v): return float(v) if isinstance(v,(np.floating,np.integer)) else v
    jsn=json.dumps({'tarih':now.isoformat(),'dosya':dosya_adi,
                    'adim_cm':meta['adim_m']*100,'mod':mod,'hedefler':hlist},
                   ensure_ascii=False,indent=2)
    return txt, csv_buf.getvalue(), jsn

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLİT ARAYÜZÜ
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="C3 Analiz",page_icon="🧲",layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp,[data-testid="stSidebar"]{background-color:#0a0a0f;color:#ddd}
h1,h2,h3{color:#00FF9D;font-family:'Courier New',monospace}
.stRadio>div{gap:4px}
.info-box{background:#111122;border:1px solid #2a2a4a;border-radius:8px;
          padding:10px 14px;margin:4px 0;font-family:'Courier New',monospace;font-size:0.82em}
.durum-badge{display:inline-block;border-radius:6px;padding:5px 14px;
             font-weight:bold;font-family:monospace;font-size:1.05em;margin:6px 0}
.tahmin-row{background:#111;border-left:3px solid;border-radius:4px;
            padding:6px 10px;margin:3px 0;font-family:monospace;font-size:0.80em}
div[data-testid="stMetricValue"]{color:#00FF9D;font-family:monospace}
.stButton>button{background:#1a237e;color:white;border:none;
                 font-family:monospace;border-radius:6px}
</style>
""",unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧲 C3 ANALİZ PRO")
    st.markdown("---")
    uploaded=st.file_uploader("CSV Dosyası",type=['csv'])
    adim_cm=st.selectbox("Adım (cm)",[20,25,50],index=2)
    adim_m=adim_cm/100.0
    st.markdown("---")
    st.markdown("**Görüntüleme Modu**")
    mod=st.radio("",['TFA','Sadece Z','Gradient','Analitik','FFT Derin','FFT Sig'],
                 label_visibility='collapsed')
    st.markdown("---")
    st.markdown("**Filtre**")
    gain      =st.slider("Kazanç",1,1000,100,5)
    esik      =st.slider("Eşik (nT)",0,500,25,1)
    blur      =st.slider("Yumuşat",0.0,5.0,0.5,0.1)
    noise     =st.slider("Parazit",1,9,3,2)
    sigma_esik=st.slider("σ Eşik",1,3,3,1)
    st.markdown("---")
    show_3d=st.checkbox("3D Görünüm",False)

# ── Ana sayfa ─────────────────────────────────────────────────────────────────
st.markdown("# C3 MANYETİK GRADİOMETRE ANALİZ")

if uploaded is None:
    st.info("📂 Sol panelden CSV dosyası yükleyin.")
    st.markdown("""
**Beklenen sütunlar:** `satir`, `sutun`, `s1_x`, `s1_y`, `s1_z`, `s2_x`, `s2_y`, `s2_z`

**Özellikler:** TFA/Z-fark · Dipol Fit (Metal/Boşluk R²) · 3-yöntem Derinlik · 
FFT Derin/Sig · Faz kayması · Obje tahmini (Çömlek, Küp, Boru, Tünel…) · TXT/CSV/JSON rapor
""")
    st.stop()

# Veri yükle
@st.cache_data(show_spinner="Veri işleniyor…")
def _yukle(fb,fn,am): return veri_yukle(fb,am)

try:
    file_bytes=uploaded.read()
    df,meta=_yukle(file_bytes,uploaded.name,adim_m)
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}"); st.stop()

# Grid & filtre
vcol='z_diff' if mod=='Sadece Z' else 'tfa_diff'
xi,yi,gX,gY,zi_raw=grid_olustur(df,vcol,meta['grid_res'])
zi,esik_val=filtrele(zi_raw,gain,noise,blur,sigma_esik,esik,meta['gurultu_std'],mod)
targets=hedef_tespit(zi,xi,yi,esik_val,meta['gurultu_std'],gain)

# Hedef seçimi
if 'sel_id' not in st.session_state or \
   st.session_state.get('sel_id') not in [t['id'] for t in targets]:
    st.session_state['sel_id'] = targets[0]['id'] if targets else None

# ── Üst şerit: alan / gürültü ─────────────────────────────────────────────────
m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Alan",f"{meta['alan_x']:.1f}m × {meta['alan_y']:.1f}m")
m2.metric("Ölçüm Noktası",f"{meta['n_sutun']}×{meta['n_satir']} = {meta['n_sutun']*meta['n_satir']}")
m3.metric("Hedef",len(targets))
m4.metric("Gürültü Std",f"{meta['gurultu_std']:.3f} nT")
m5.metric("Oto Eşik",f"{meta['gurultu_std']*sigma_esik:.2f} nT")

st.markdown("---")

# ── İki ana sütun: SOL harita | SAĞ bilgi ────────────────────────────────────
col_harita, col_bilgi = st.columns([3, 2], gap="medium")

with col_harita:
    fig_h=fig_harita(zi,xi,yi,targets,meta,mod,st.session_state['sel_id'])
    st.pyplot(fig_h,use_container_width=True)
    plt.close(fig_h)

    if show_3d:
        st.markdown("**3D Görünüm**")
        f3=fig_3d(zi,gX,gY,xi,yi,targets,gain)
        st.pyplot(f3,use_container_width=True)
        plt.close(f3)

with col_bilgi:
    # ── Hedef seç ─────────────────────────────────────────────────────────────
    if not targets:
        st.warning("⚠ Kayda değer anomali tespit edilmedi.")
        st.markdown(f"Gürültü std: `{meta['gurultu_std']:.3f} nT`  \n"
                    f"Eşik: `{esik_val:.2f} nT`  \nKazanç/sigma değerlerini deneyin.")
    else:
        hedef_seçenekler={f"H{t['id']}  ({t['amp']:+.0f} nT)":t['id'] for t in targets}
        seç=st.radio("**Hedef Seç**",list(hedef_seçenekler.keys()),
                     horizontal=True,label_visibility='visible')
        st.session_state['sel_id']=hedef_seçenekler[seç]
        sel_t=next(t for t in targets if t['id']==st.session_state['sel_id'])

        # analiz
        a=hedef_analiz(sel_t,zi,zi_raw,xi,yi,gain,meta['gurultu_std'],esik_val,mod)

        # Sütun/Satır konumu
        t_sut=round(sel_t['x']/meta['adim_m'])+meta['sutun_min']
        t_sat=round(sel_t['y']/meta['adim_m'])+meta['satir_min']

        # Teşhis badge
        bc=('#FF450033' if 'METAL' in a['durum'] else
            '#00FFFF22' if 'BOŞLUK' in a['durum'] or 'TÜNEL' in a['durum'] else
            '#FFD70033' if 'KARMA' in a['durum'] else '#ffffff11')
        st.markdown(
            f'<div class="durum-badge" style="background:{bc};border:1px solid {a["renk"]};color:{a["renk"]}">'
            f'{a["durum"]}</div>',unsafe_allow_html=True)
        st.markdown(f"*{a['aciklama']}*")

        st.markdown("---")

        # Konum + sayısal bilgiler
        r1,r2=st.columns(2)
        with r1:
            st.markdown('<div class="info-box">'
                f'📍 <b>Sütun {t_sut} / Satır {t_sat}</b><br>'
                f'X: {sel_t["x"]:.2f} m &nbsp; Y: {sel_t["y"]:.2f} m'
                '</div>',unsafe_allow_html=True)
            st.markdown('<div class="info-box">'
                f'🧲 Şiddet: <b>{a["val"]:.1f} nT</b><br>'
                f'SNR: {a["snr"]:.1f}×'
                '</div>',unsafe_allow_html=True)
        with r2:
            gv_str=f"%{a['guven']}" if a['guven'] else "?"
            st.markdown('<div class="info-box">'
                f'📏 Derinlik: <b>~{a["depth"]:.2f} m</b><br>'
                f'Güven: {gv_str}'
                '</div>',unsafe_allow_html=True)
            st.markdown('<div class="info-box">'
                f'🔩 Metal R²: <b>{f"{a[\"r2m\"]:.3f}" if a["r2m"] is not None else "—"}</b><br>'
                f'Boşluk R²: <b>{f"{a[\"r2b\"]:.3f}" if a["r2b"] is not None else "—"}</b>'
                '</div>',unsafe_allow_html=True)

        # Dipol yorumu
        if a['dyorum']:
            st.caption(f"🔬 Dipol: {a['dyorum']}")
        if a['ystr']:
            st.caption(f"📐 Yöntem: {a['ystr']}")
        if a['fwhm'] is not None:
            st.caption(f"↔ FWHM: {a['fwhm']:.3f}m — {a['siv_y']}")
        if a['ortusme']>0:
            st.caption(f"〰 Faz örtüşme: {a['ortusme']:.2f} — {a['faz_y']}")

        st.markdown("---")
        st.markdown("**🏺 Obje Tahmini**")
        for isim,puan,rk in a['tahminler']:
            st.markdown(
                f'<div class="tahmin-row" style="border-color:{rk}">'
                f'<span style="color:{rk}">{bar5(puan)} %{int(puan*100)}</span>'
                f'<br>{isim}</div>',unsafe_allow_html=True)

        st.markdown("---")
        # Profil grafikleri
        st.markdown("**📈 Kesit Profilleri**")
        fp=fig_profil(a)
        st.pyplot(fp,use_container_width=True)
        plt.close(fp)

        st.markdown("---")
        # Tüm hedefler özet tablosu
        st.markdown("**🎯 Tüm Hedefler**")
        rows=[]
        for t in targets:
            try:
                ri2=np.argmin(np.abs(yi-t['y']))
                pt=zi[ri2,:]/(gain or 1.0)
                d2,gv2,_=derinlik_pro(pt,xi,meta['gurultu_std'])
                _,_,rm2,rb2,_,tip2,_=dipol_fit(pt,xi)
            except:
                d2=derinlik_simple(t['amp']/(gain or 1.0)); gv2=rm2=rb2=tip2=None
            ts2=round(t['x']/meta['adim_m'])+meta['sutun_min']
            tr2=round(t['y']/meta['adim_m'])+meta['satir_min']
            rows.append({'ID':f"H{t['id']}",
                         'Konum':f"S{ts2}/R{tr2}",
                         'nT':f"{t['amp']:+.0f}",
                         '~m':f"{float(d2) if d2 else 0:.2f}",
                         'Güven':f"%{gv2}" if gv2 else '?',
                         'Tip':tip2 or '?'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
                     height=min(200,40+len(rows)*35))

# ── Rapor ─────────────────────────────────────────────────────────────────────
st.markdown("---")
rc1,rc2,rc3,rc4=st.columns(4)
with rc1:
    if st.button("📋 Rapor Oluştur"):
        txt_r,csv_r,jsn_r=rapor_uret(meta,mod,targets,zi,xi,yi,
                                      gain,meta['gurultu_std'],uploaded.name,esik_val)
        st.session_state.update(rapor_txt=txt_r,rapor_csv=csv_r,rapor_jsn=jsn_r)
        st.success("Hazır!")
if 'rapor_txt' in st.session_state:
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    with rc2:
        st.download_button("⬇ TXT",st.session_state['rapor_txt'],
                           f"C3_{ts}.txt",'text/plain')
    with rc3:
        st.download_button("⬇ CSV",st.session_state['rapor_csv'],
                           f"C3_{ts}.csv",'text/csv')
    with rc4:
        st.download_button("⬇ JSON",st.session_state['rapor_jsn'],
                           f"C3_{ts}.json",'application/json')
    with st.expander("Rapor Önizleme"):
        st.text(st.session_state['rapor_txt'])

st.markdown("---")
st.caption(f"C3 ANALİZ PRO v3 | {meta['n_sutun']}×{meta['n_satir']} nokta | "
           f"{meta['alan_x']:.1f}m×{meta['alan_y']:.1f}m | "
           f"Grid {meta['grid_res']}px | Adım {adim_cm}cm")
