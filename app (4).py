"""Dashboard interaktif EDA Penjualan Batik Pasir Putih 2025 (satu file).

Jalankan: streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


from scipy import stats

PRODUK = [
    "Kemeja Katun", "Hem Katun", "Kemeja Furing", "Hem Furing",
    "Tunik Katun", "Blus", "Tunik Furing", "Sarung", "Udeng",
]
BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
    "Agustus", "September", "Oktober", "November", "Desember",
]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
KOLOM_WAKTU = ["Bulan", "Minggu Ke", "Hari", "Jenis Hari"]
KOLOM_WAJIB = KOLOM_WAKTU + PRODUK
TOTAL = "Total"          # total unit dari produk yang sedang dipilih
BARIS_HEADER = 3         # header berada di baris ke-4 file sumber


# --------------------------------------------------------------------------- #
# Pemuatan dan pembersihan
# --------------------------------------------------------------------------- #
def kolom_hilang(raw: pd.DataFrame) -> list[str]:
    """Kembalikan kolom wajib yang tidak ada pada data mentah."""
    return [c for c in KOLOM_WAJIB if c not in raw.columns]


def periksa_kualitas(raw: pd.DataFrame) -> dict:
    """Ringkasan kualitas data mentah. Kompleksitas O(n * k)."""
    r = raw.copy()
    r.columns = r.columns.astype(str).str.strip()
    nilai = r[PRODUK].apply(pd.to_numeric, errors="coerce")
    hasil = {
        "Jumlah baris": len(r),
        "Missing value": int(r[KOLOM_WAJIB].isna().sum().sum()),
        "Baris duplikat": int(r.duplicated().sum()),
        "Nilai negatif": int((nilai < 0).sum().sum()),
        "Hari tanpa penjualan": int((nilai.sum(axis=1) == 0).sum()),
    }
    if "Total Harian" in r.columns:
        total_asli = pd.to_numeric(r["Total Harian"], errors="coerce")
        hasil["Total tidak cocok dengan jumlah produk"] = int(
            (nilai.sum(axis=1) != total_asli).sum()
        )
    return hasil


def bersihkan(raw: pd.DataFrame) -> pd.DataFrame:
    """Rapikan tipe data dan tambah indeks urutan hari.

    Baris dengan nilai kosong pada kolom wajib dibuang. Urutan baris file
    dipertahankan sebagai ``Hari_ke`` (1..n), bukan tanggal kalender.
    """
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(subset=KOLOM_WAJIB).reset_index(drop=True)
    df[PRODUK] = df[PRODUK].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df["Bulan"] = pd.Categorical(df["Bulan"], categories=BULAN, ordered=True)
    df["Hari"] = pd.Categorical(df["Hari"], categories=HARI, ordered=True)
    df["Minggu Ke"] = pd.to_numeric(df["Minggu Ke"], errors="coerce").astype(int)
    df["Jenis Hari"] = df["Jenis Hari"].astype(str).str.strip()
    df["Hari_ke"] = np.arange(1, len(df) + 1)
    return df


def tambah_total(df: pd.DataFrame, produk: list[str]) -> pd.DataFrame:
    """Hitung total unit harian dari produk terpilih."""
    out = df.copy()
    out[TOTAL] = out[produk].sum(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Statistik
# --------------------------------------------------------------------------- #
def batas_outlier(s: pd.Series, k: float = 1.5) -> dict:
    """Batas outlier IQR. O(n log n) akibat pengurutan kuantil."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {"q1": q1, "q3": q3, "iqr": iqr, "bawah": q1 - k * iqr, "atas": q3 + k * iqr}


def deskriptif(df: pd.DataFrame, produk: list[str]) -> pd.DataFrame:
    kolom = produk + [TOTAL]
    d = pd.DataFrame({
        "mean": df[kolom].mean(),
        "median": df[kolom].median(),
        "modus": df[kolom].mode().iloc[0],
        "min": df[kolom].min(),
        "max": df[kolom].max(),
        "std": df[kolom].std(),
        "skewness": df[kolom].skew(),
    })
    d["CV (std/mean)"] = d["std"] / d["mean"].replace(0, np.nan)
    d["total unit"] = df[kolom].sum()
    return d.round(2)


def kontribusi(df: pd.DataFrame, produk: list[str]) -> pd.DataFrame:
    unit = df[produk].sum().sort_values(ascending=False)
    total = unit.sum()
    pct = unit / total * 100 if total else unit * 0
    return pd.DataFrame({
        "unit": unit,
        "kontribusi (%)": pct.round(1),
        "kumulatif (%)": pct.cumsum().round(1),
    })


def uji_akhir_pekan(df: pd.DataFrame) -> dict | None:
    """Uji Mann-Whitney (akhir pekan > hari kerja). None bila salah satu grup kosong."""
    kerja = df.loc[df["Jenis Hari"] == "Hari Kerja", TOTAL]
    akhir = df.loc[df["Jenis Hari"] == "Akhir Pekan", TOTAL]
    if kerja.empty or akhir.empty:
        return None
    _, p = stats.mannwhitneyu(akhir, kerja, alternative="greater")
    total = df[TOTAL].sum()
    return {
        "rata_kerja": kerja.mean(), "median_kerja": kerja.median(),
        "rata_akhir": akhir.mean(), "median_akhir": akhir.median(),
        "rasio": akhir.mean() / kerja.mean() if kerja.mean() else np.nan,
        "kontribusi_akhir": akhir.sum() / total * 100 if total else np.nan,
        "porsi_hari": len(akhir) / len(df) * 100,
        "p_value": p,
    }


def total_bulanan(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Bulan", observed=True)[TOTAL].sum()


def indeks_bulanan(df: pd.DataFrame) -> pd.Series:
    tb = total_bulanan(df)
    return (tb / tb.mean() * 100).round(0).astype(int)


def lonjakan_akhir_pekan(df: pd.DataFrame, produk: list[str]) -> pd.Series | None:
    kerja = df[df["Jenis Hari"] == "Hari Kerja"][produk].mean()
    akhir = df[df["Jenis Hari"] == "Akhir Pekan"][produk].mean()
    if df["Jenis Hari"].nunique() < 2:
        return None
    return (akhir / kerja.replace(0, np.nan)).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Keputusan
# --------------------------------------------------------------------------- #
def kelas_produk(pct: float, prioritas: float, pendukung: float) -> str:
    if pct >= prioritas:
        return "Prioritas"
    return "Pendukung" if pct >= pendukung else "Lambat"


def target_stok(df: pd.DataFrame, produk: list[str], persen: int,
                prioritas: float, pendukung: float) -> pd.DataFrame:
    """Target stok harian berbasis persentil. O(n log n) per produk."""
    q = persen / 100
    kerja = df[df["Jenis Hari"] == "Hari Kerja"][produk]
    akhir = df[df["Jenis Hari"] == "Akhir Pekan"][produk]
    kon = kontribusi(df, produk)
    out = pd.DataFrame({
        "Rata-rata hari kerja": kerja.mean().round(1),
        "Rata-rata akhir pekan": akhir.mean().round(1),
        f"Stok hari kerja (P{persen})": np.ceil(kerja.quantile(q)),
        f"Stok akhir pekan (P{persen})": np.ceil(akhir.quantile(q)),
        "Kontribusi (%)": kon["kontribusi (%)"],
    }).loc[kon.index]
    out["Kelas"] = out["Kontribusi (%)"].map(lambda v: kelas_produk(v, prioritas, pendukung))
    return out


def peta_musim(df: pd.DataFrame, puncak: int, rendah: int) -> pd.DataFrame:
    tb = total_bulanan(df)
    idx = indeks_bulanan(df)
    out = pd.DataFrame({"Total unit": tb, "Indeks (rata-rata=100)": idx})
    out["Kategori"] = np.where(idx >= puncak, "Puncak",
                               np.where(idx <= rendah, "Rendah", "Normal"))
    out.index = out.index.astype(str)
    return out


# ===========================================================================
# ANTARMUKA STREAMLIT
# ===========================================================================



BIRU, ORANYE, ABU = "#2a78d6", "#eb6834", "#9a9a96"
WARNA_JENIS = {"Hari Kerja": BIRU, "Akhir Pekan": ORANYE}
FILE_DEFAULT = Path("nyoh_data.xlsx")

st.set_page_config(page_title="Dashboard Batik Pasir Putih 2025", layout="wide")


# --------------------------------------------------------------------------- #
# Helper tampilan (kompatibel lintas versi Streamlit)
# --------------------------------------------------------------------------- #
def tampil(fig: go.Figure) -> None:
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), title_x=0)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def tabel(data: pd.DataFrame | pd.io.formats.style.Styler) -> None:
    try:
        st.dataframe(data, width="stretch")
    except TypeError:
        st.dataframe(data, use_container_width=True)


def unduh(data: pd.DataFrame, nama: str, label: str = "Unduh CSV") -> None:
    st.download_button(label, data.to_csv().encode("utf-8"), file_name=nama, mime="text/csv")


def caption(teks: str) -> None:
    st.info(teks)


def heatmap(data: pd.DataFrame, judul: str, skala: str = "Blues",
            fmt: str = ".0f", zmin=None, zmax=None) -> go.Figure:
    d = data.copy()
    d.index = d.index.astype(str)
    fig = px.imshow(d, text_auto=fmt, aspect="auto", color_continuous_scale=skala,
                    zmin=zmin, zmax=zmax, title=judul)
    fig.update_xaxes(side="bottom")
    return fig


# --------------------------------------------------------------------------- #
# Pemuatan data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Membaca data...")
def baca_excel(isi: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(isi), header=BARIS_HEADER)


def muat_sumber() -> pd.DataFrame | None:
    st.sidebar.header("Data")
    unggahan = st.sidebar.file_uploader("Unggah nyoh_data.xlsx", type=["xlsx"])
    if unggahan is not None:
        return baca_excel(unggahan.getvalue())
    if FILE_DEFAULT.exists():
        st.sidebar.caption(f"Memakai file lokal: {FILE_DEFAULT.name}")
        return baca_excel(FILE_DEFAULT.read_bytes())
    return None


raw = muat_sumber()
st.title("Dashboard Penjualan Batik Pasir Putih 2025")
if raw is None:
    st.info("Unggah file nyoh_data.xlsx pada sidebar untuk memulai.")
    st.stop()

raw.columns = raw.columns.astype(str).str.strip()
hilang = kolom_hilang(raw)
if hilang:
    st.error(f"Kolom wajib tidak ditemukan: {', '.join(hilang)}. "
             "Pastikan header berada di baris ke-4 file.")
    st.stop()

kualitas = periksa_kualitas(raw)
data_penuh = bersihkan(raw)

# --------------------------------------------------------------------------- #
# Filter sidebar
# --------------------------------------------------------------------------- #
st.sidebar.header("Filter")
bulan_pilih = st.sidebar.multiselect("Bulan", BULAN, default=BULAN)
hari_pilih = st.sidebar.multiselect("Hari", HARI, default=HARI)
jenis_opsi = sorted(data_penuh["Jenis Hari"].unique())
jenis_pilih = st.sidebar.multiselect("Jenis hari", jenis_opsi, default=jenis_opsi)
minggu_opsi = sorted(data_penuh["Minggu Ke"].unique())
minggu_pilih = st.sidebar.multiselect("Minggu ke", minggu_opsi, default=minggu_opsi)
produk_pilih = st.sidebar.multiselect("Produk", PRODUK, default=PRODUK)

if not produk_pilih:
    st.warning("Pilih minimal satu produk.")
    st.stop()

mask = (
    data_penuh["Bulan"].isin(bulan_pilih)
    & data_penuh["Hari"].isin(hari_pilih)
    & data_penuh["Jenis Hari"].isin(jenis_pilih)
    & data_penuh["Minggu Ke"].isin(minggu_pilih)
)
df = tambah_total(data_penuh[mask], produk_pilih)
if df.empty:
    st.warning("Tidak ada data yang cocok dengan filter.")
    st.stop()

st.sidebar.caption(f"{len(df)} dari {len(data_penuh)} hari ditampilkan.")
st.caption(
    "Satuan adalah unit terjual, bukan rupiah. "
    f"Total dihitung dari {len(produk_pilih)} produk terpilih."
)

total_unit = int(df[TOTAL].sum())
kon = kontribusi(df, produk_pilih)
uji = uji_akhir_pekan(df)
tb = total_bulanan(df)

tabs = st.tabs([
    "Ringkasan", "Kualitas Data", "Deskriptif", "Univariat",
    "Bivariat", "Multivariat", "Pola", "Keputusan",
])

# --------------------------------------------------------------------------- #
# 1. Ringkasan
# --------------------------------------------------------------------------- #
with tabs[0]:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total unit", f"{total_unit:,}")
    k2.metric("Rata-rata per hari", f"{df[TOTAL].mean():.1f}", f"median {df[TOTAL].median():.0f}", delta_color="off")
    k3.metric("Produk teratas", kon.index[0], f"{kon['kontribusi (%)'].iloc[0]:.1f}%", delta_color="off")
    k4.metric("Rasio akhir pekan / hari kerja",
              f"{uji['rasio']:.1f}x" if uji else "n/a")

    rata_hari = df.groupby("Hari", observed=True)[TOTAL].mean()
    baris = [
        ("Q1", "Total unit dan produk teratas",
         f"{total_unit:,} unit; {kon.index[0]} {kon['kontribusi (%)'].iloc[0]:.1f}%; "
         f"4 teratas {kon['kontribusi (%)'].head(4).sum():.1f}%"),
        ("Q2", "Akhir pekan vs hari kerja",
         (f"{uji['rata_akhir']:.1f} vs {uji['rata_kerja']:.1f} unit ({uji['rasio']:.1f}x), "
          f"kontribusi {uji['kontribusi_akhir']:.0f}% total, p = {uji['p_value']:.1e}")
         if uji else "Butuh data hari kerja dan akhir pekan sekaligus"),
        ("Q2", "Hari puncak", f"{rata_hari.idxmax()} ({rata_hari.max():.1f} unit rata-rata)"),
        ("Q3", "Bulan puncak / terendah",
         f"{tb.idxmax()} {tb.max():,} / {tb.idxmin()} {tb.min():,}"),
        ("Q4", "Produk lambat",
         ", ".join(f"{k} ({v:.1f}%)" for k, v in kon["kontribusi (%)"].tail(2).items())),
    ]
    st.subheader("Jawaban pertanyaan analitik")
    tabel(pd.DataFrame(baris, columns=["Pertanyaan", "Temuan", "Hasil"]))
    with st.expander("Daftar pertanyaan analitik"):
        st.markdown(
            "- **Q1 Deskriptif:** total unit, rata-rata per hari, produk berkontribusi terbesar.\n"
            "- **Q2 Diagnostik:** selisih akhir pekan dan hari kerja.\n"
            "- **Q3 Diagnostik:** bulan puncak, terendah, dan pola musiman.\n"
            "- **Q4 Diagnostik:** produk yang bergerak bersamaan dan produk lambat.\n"
            "- **Q5 Preskriptif:** alokasi stok dan tenaga kerja (tab Keputusan)."
        )

# --------------------------------------------------------------------------- #
# 2. Kualitas data
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Pemeriksaan kualitas (data mentah, tanpa filter)")
    cols = st.columns(len(kualitas))
    for c, (k, v) in zip(cols, kualitas.items()):
        c.metric(k, v)

    st.subheader("Outlier (metode IQR)")
    k_iqr = st.slider("Pengali IQR", 1.0, 3.0, 1.5, 0.1)
    b = batas_outlier(df[TOTAL], k_iqr)
    outlier = df[(df[TOTAL] > b["atas"]) | (df[TOTAL] < b["bawah"])]
    st.write(f"Q1 = {b['q1']:.0f}, Q3 = {b['q3']:.0f}, IQR = {b['iqr']:.0f}, "
             f"batas atas = {b['atas']:.1f}. "
             f"Outlier: **{len(outlier)}** dari {len(df)} hari.")
    o1, o2 = st.columns(2)
    with o1:
        sb = outlier["Bulan"].value_counts().reindex(BULAN).dropna().astype(int)
        if not sb.empty:
            tampil(px.bar(sb, title="Outlier per bulan", labels={"value": "Hari", "index": "Bulan"},
                          color_discrete_sequence=[ORANYE]).update_layout(showlegend=False))
    with o2:
        sh = outlier["Hari"].value_counts().reindex(HARI).dropna().astype(int)
        if not sh.empty:
            tampil(px.bar(sh, title="Outlier per hari", labels={"value": "Hari", "index": "Hari"},
                          color_discrete_sequence=[ORANYE]).update_layout(showlegend=False))
    with st.expander("Daftar hari outlier"):
        tabel(outlier[["Bulan", "Minggu Ke", "Hari", "Jenis Hari", TOTAL]])
    caption("Outlier dipertahankan: nilainya wajar, terkonsentrasi pada akhir pekan dan "
            "bulan tertentu, sehingga mencerminkan lonjakan permintaan nyata. "
            "Catatan: Jumat berlabel Akhir Pekan oleh pencatat, tiap bulan tercatat 28 hari, "
            "dan data hanya satu tahun.")

# --------------------------------------------------------------------------- #
# 3. Deskriptif
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Statistik deskriptif")
    desk = deskriptif(df, produk_pilih)
    tabel(desk)
    unduh(desk, "statistik_deskriptif.csv")

    st.subheader("Kontribusi produk")
    tabel(kon)
    pareto = go.Figure()
    pareto.add_bar(x=kon.index, y=kon["unit"], name="Unit", marker_color=BIRU)
    pareto.add_scatter(x=kon.index, y=kon["kumulatif (%)"], name="Kumulatif (%)",
                       yaxis="y2", mode="lines+markers", line=dict(color=ORANYE))
    pareto.update_layout(title="Pareto kontribusi produk",
                         yaxis=dict(title="Unit"),
                         yaxis2=dict(title="Kumulatif (%)", overlaying="y", side="right", range=[0, 105]),
                         legend=dict(orientation="h", y=-0.2))
    tampil(pareto)

# --------------------------------------------------------------------------- #
# 4. Univariat
# --------------------------------------------------------------------------- #
with tabs[3]:
    lebar = st.slider("Lebar bin histogram (unit)", 1, 20, 5)
    u1, u2 = st.columns([2, 1])
    with u1:
        h = px.histogram(df, x=TOTAL, nbins=max(1, int(df[TOTAL].max() // lebar)),
                         color_discrete_sequence=[BIRU], title="Distribusi unit terjual per hari")
        h.add_vline(x=df[TOTAL].mean(), line_color=ORANYE, line_width=2,
                    annotation_text=f"mean {df[TOTAL].mean():.1f}")
        h.add_vline(x=df[TOTAL].median(), line_color="#2b2b2a", line_dash="dash",
                    annotation_text=f"median {df[TOTAL].median():.0f}", annotation_position="bottom right")
        h.update_layout(xaxis_title="Unit per hari", yaxis_title="Jumlah hari", bargap=0.05)
        tampil(h)
    with u2:
        tampil(px.box(df, y=TOTAL, points="outliers", color_discrete_sequence=[BIRU],
                      title="Boxplot dan outlier"))
    urut = kon["unit"].sort_values()
    bar = px.bar(urut, orientation="h", title="Total unit per produk",
                 labels={"value": "Unit", "index": ""}, color_discrete_sequence=[BIRU],
                 text=[f"{v:,} ({v / max(total_unit, 1) * 100:.1f}%)" for v in urut.values])
    bar.update_layout(showlegend=False)
    tampil(bar)
    skew = df[TOTAL].skew()
    caption(f"Skewness Total = {skew:.2f}. "
            + ("Distribusi menceng kanan: mayoritas hari rendah, ekor panjang pada hari ramai."
               if skew > 0.5 else "Distribusi relatif simetris pada filter ini."))

# --------------------------------------------------------------------------- #
# 5. Bivariat
# --------------------------------------------------------------------------- #
with tabs[4]:
    b1, b2 = st.columns(2)
    dasar = b1.radio("Kelompokkan berdasarkan", ["Hari", "Jenis Hari", "Bulan", "Minggu Ke"], horizontal=True)
    ukuran = b2.radio("Ukuran", ["Rata-rata", "Total", "Median"], horizontal=True)
    fn = {"Rata-rata": "mean", "Total": "sum", "Median": "median"}[ukuran]
    g = df.groupby(dasar, observed=True)[TOTAL].agg(fn).reset_index()
    g[dasar] = g[dasar].astype(str)
    if dasar == "Hari":
        jenis = df.groupby("Hari", observed=True)["Jenis Hari"].agg(lambda s: s.mode().iat[0])
        jenis.index = jenis.index.astype(str)
        g["Jenis Hari"] = g["Hari"].map(jenis)
        fig = px.bar(g, x=dasar, y=TOTAL, color="Jenis Hari", color_discrete_map=WARNA_JENIS, text_auto=".1f")
    elif dasar == "Jenis Hari":
        fig = px.bar(g, x=dasar, y=TOTAL, color="Jenis Hari", color_discrete_map=WARNA_JENIS, text_auto=".1f")
    else:
        fig = px.bar(g, x=dasar, y=TOTAL, color_discrete_sequence=[BIRU], text_auto=".1f")
    fig.update_layout(title=f"{ukuran} unit per {dasar.lower()}", yaxis_title="Unit")
    tampil(fig)

    st.subheader("Akhir pekan vs hari kerja")
    if uji:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rata-rata hari kerja", f"{uji['rata_kerja']:.1f}")
        m2.metric("Rata-rata akhir pekan", f"{uji['rata_akhir']:.1f}")
        m3.metric("Rasio", f"{uji['rasio']:.2f}x")
        m4.metric("p-value Mann-Whitney", f"{uji['p_value']:.1e}")
        tampil(px.box(df, x="Jenis Hari", y=TOTAL, color="Jenis Hari", color_discrete_map=WARNA_JENIS,
                      category_orders={"Jenis Hari": ["Hari Kerja", "Akhir Pekan"]},
                      title="Sebaran per jenis hari").update_layout(showlegend=False))
        caption(f"Akhir pekan menyumbang {uji['kontribusi_akhir']:.1f}% unit dari "
                f"{uji['porsi_hari']:.0f}% jumlah hari. Uji Mann-Whitney dipakai karena "
                "distribusi menceng dan tidak normal.")
    else:
        st.info("Butuh data hari kerja dan akhir pekan sekaligus. Longgarkan filter jenis hari.")

    st.subheader("Hubungan dua produk")
    if len(produk_pilih) >= 2:
        s1, s2, s3 = st.columns(3)
        pa = s1.selectbox("Produk sumbu X", produk_pilih, index=0)
        pb = s2.selectbox("Produk sumbu Y", [p for p in produk_pilih if p != pa], index=0)
        warna = s3.checkbox("Warnai menurut jenis hari", value=True)
        sc = px.scatter(df, x=pa, y=pb, opacity=0.55,
                        color="Jenis Hari" if warna else None,
                        color_discrete_map=WARNA_JENIS if warna else None,
                        color_discrete_sequence=None if warna else [BIRU],
                        hover_data=["Bulan", "Hari"])
        if df[pa].std() > 0:
            m, c0 = np.polyfit(df[pa], df[pb], 1)
            xs = np.linspace(df[pa].min(), df[pa].max(), 50)
            sc.add_scatter(x=xs, y=m * xs + c0, mode="lines", line=dict(color="#2b2b2a", width=2),
                           name="Regresi linear")
        r = df[pa].corr(df[pb])
        sc.update_layout(title=f"{pa} vs {pb} (r = {r:.2f})")
        tampil(sc)
    else:
        st.info("Pilih minimal dua produk pada sidebar.")

# --------------------------------------------------------------------------- #
# 6. Multivariat
# --------------------------------------------------------------------------- #
with tabs[5]:
    if len(produk_pilih) >= 2:
        subset = st.radio("Subset data untuk korelasi", ["Semua hari", "Hari Kerja", "Akhir Pekan"], horizontal=True)
        sub = df if subset == "Semua hari" else df[df["Jenis Hari"] == subset]
        if len(sub) > 2:
            kor = sub[produk_pilih].corr()
            tampil(heatmap(kor, f"Korelasi antar produk ({subset})", "RdBu_r", ".2f", -1, 1))
            pasangan = (kor.where(np.triu(np.ones(kor.shape, dtype=bool), 1)).stack().dropna()
                        .sort_values(ascending=False))
            n_top = st.slider("Jumlah pasangan teratas", 1, min(10, len(pasangan)), min(4, len(pasangan)))
            baris = []
            for (a, b_), r_all in pasangan.head(n_top).items():
                kerja = df[df["Jenis Hari"] == "Hari Kerja"]
                akhir = df[df["Jenis Hari"] == "Akhir Pekan"]
                baris.append({"Pasangan": f"{a} - {b_}", "r semua hari": r_all,
                              "r hari kerja": kerja[a].corr(kerja[b_]),
                              "r akhir pekan": akhir[a].corr(akhir[b_])})
            tabel(pd.DataFrame(baris).round(2))
            caption("Korelasi dengan Total sengaja tidak dipakai karena Total adalah jumlah produk "
                    "itu sendiri (part-whole). Korelasi yang turun setelah dipisah jenis hari "
                    "menandakan faktor bersama (hari ramai).")
        else:
            st.info("Data pada subset terlalu sedikit.")
    else:
        st.info("Pilih minimal dua produk pada sidebar.")

    agg = st.radio("Ukuran heatmap Bulan x Hari", ["mean", "sum", "median"], horizontal=True)
    pv = df.pivot_table(index="Bulan", columns="Hari", values=TOTAL, aggfunc=agg, observed=True)
    tampil(heatmap(pv, f"Unit terjual per Bulan x Hari ({agg})"))

    bauran = df.groupby("Bulan", observed=True)[produk_pilih].sum()
    bauran_pct = bauran.div(bauran.sum(axis=1).replace(0, np.nan), axis=0) * 100
    tampil(heatmap(bauran_pct, "Bauran produk per bulan (% dari total unit bulan)"))

# --------------------------------------------------------------------------- #
# 7. Pola
# --------------------------------------------------------------------------- #
with tabs[6]:
    jendela = st.slider("Jendela rata-rata bergerak (hari)", 1, 28, 7)
    tren = df.sort_values("Hari_ke").copy()
    tren["MA"] = tren[TOTAL].rolling(jendela, center=True, min_periods=max(1, jendela // 2)).mean()
    f = go.Figure()
    f.add_scatter(x=tren["Hari_ke"], y=tren[TOTAL], mode="lines", name="Harian",
                  line=dict(color=BIRU, width=1), opacity=0.3)
    f.add_scatter(x=tren["Hari_ke"], y=tren["MA"], mode="lines", name=f"Rata-rata bergerak {jendela} hari",
                  line=dict(color=BIRU, width=3))
    pos = tren.groupby("Bulan", observed=True)["Hari_ke"].mean()
    f.update_layout(title="Tren penjualan harian sepanjang 2025", yaxis_title="Unit per hari",
                    xaxis=dict(tickmode="array", tickvals=pos.values, ticktext=[str(b)[:3] for b in pos.index]),
                    legend=dict(orientation="h", y=-0.15))
    tampil(f)

    idx = indeks_bulanan(df)
    warna_idx = [ORANYE if v >= 120 else BIRU for v in idx.values]
    fi = go.Figure(go.Bar(x=idx.index.astype(str), y=idx.values, marker_color=warna_idx,
                          text=idx.values, textposition="outside"))
    fi.add_hline(y=100, line_dash="dash", line_color="#2b2b2a")
    fi.update_layout(title="Indeks bulanan (rata-rata bulanan = 100, oranye >= 120)")
    tampil(fi)

    lon = lonjakan_akhir_pekan(df, produk_pilih)
    if lon is not None:
        lon = lon.dropna().sort_values()
        fl = px.bar(lon, orientation="h", labels={"value": "Kelipatan", "index": ""},
                    title="Lonjakan akhir pekan per produk (rata-rata akhir pekan / hari kerja)",
                    color_discrete_sequence=[BIRU], text=[f"{v:.1f}x" for v in lon.values])
        fl.add_vline(x=1, line_dash="dash", line_color="#2b2b2a")
        fl.update_layout(showlegend=False)
        tampil(fl)
    caption("Produk dengan volume sangat kecil dapat menampilkan lonjakan tinggi dari basis rendah. "
            "Karena hanya satu tahun data, sifat musiman bulanan masih berupa dugaan.")

# --------------------------------------------------------------------------- #
# 8. Keputusan
# --------------------------------------------------------------------------- #
with tabs[7]:
    st.subheader("Target stok harian per produk")
    d1, d2, d3 = st.columns(3)
    persen = d1.slider("Persentil stok (P)", 50, 99, 90, help="P90 berarti stok cukup pada 9 dari 10 hari.")
    t_prior = d2.slider("Batas kelas Prioritas (% kontribusi)", 5.0, 30.0, 10.0, 0.5)
    t_dukung = d3.slider("Batas kelas Pendukung (% kontribusi)", 1.0, 15.0, 5.0, 0.5)
    stok = target_stok(df, produk_pilih, persen, t_prior, t_dukung)
    tabel(stok)
    unduh(stok, f"target_stok_P{persen}.csv")

    st.subheader("Peta musim")
    e1, e2 = st.columns(2)
    t_puncak = e1.slider("Indeks batas Puncak", 100, 200, 120)
    t_rendah = e2.slider("Indeks batas Rendah", 30, 100, 70)
    musim = peta_musim(df, t_puncak, t_rendah)
    tabel(musim)
    unduh(musim, "peta_musim.csv")

    st.subheader("Rekomendasi tindakan")
    prior = stok.index[stok["Kelas"] == "Prioritas"].tolist()
    lambat = stok.index[stok["Kelas"] == "Lambat"].tolist()
    puncak = musim.index[musim["Kategori"] == "Puncak"].tolist()
    rendah = musim.index[musim["Kategori"] == "Rendah"].tolist()
    rata_hari = df.groupby("Hari", observed=True)[TOTAL].mean()
    rekom = pd.DataFrame([
        [1, "Pusatkan stok pada produk prioritas",
         ", ".join(prior) or "-", f"{stok.loc[prior, 'Kontribusi (%)'].sum():.0f}% volume" if prior else "-"],
        [2, "Siapkan stok dan staf pada Kamis untuk Jumat sampai Minggu (target P" + str(persen) + ")",
         f"Hari puncak: {rata_hari.idxmax()}",
         f"Akhir pekan {uji['rasio']:.1f}x hari kerja" if uji else "-"],
        [3, "Siapkan stok 4 sampai 6 minggu sebelum bulan puncak",
         ", ".join(puncak) or "-", "Indeks >= " + str(t_puncak)],
        [4, "Promosi pada hari kerja dan bulan rendah",
         f"Hari terendah: {rata_hari.idxmin()}; bulan: {', '.join(rendah) or '-'}",
         "Indeks <= " + str(t_rendah)],
        [5, "Tinjau lini lambat: kurangi stok, tawarkan bundling",
         ", ".join(lambat) or "-", f"Kontribusi < {t_dukung:g}%"],
        [6, "Catat harga jual, HPP, dan event/libur harian",
         "Analisis saat ini hanya volume", "Analisis omzet dan margin"],
    ], columns=["Prioritas", "Keputusan", "Objek", "Dasar data"])
    tabel(rekom)
    unduh(rekom.set_index("Prioritas"), "rekomendasi.csv")
    caption("Langkah lanjutan: gabungkan data 2024 untuk menguji pola musiman, tambahkan harga dan "
            "margin agar prioritas berbasis laba, lalu bangun model peramalan setelah data cukup.")
