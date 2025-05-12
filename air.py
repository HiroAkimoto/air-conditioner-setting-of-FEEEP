import math
import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
from datetime import datetime, timedelta, time
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/1WcO_36bZ53qUFa1YR5PyJvYyL9jGxABU5PP4OiiS-Sw/export?gid=0&format=csv"
)
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
TZ = pytz.timezone("Asia/Tokyo")
HOURS = range(10, 18)  # 10 – 17 時台（平均 10–18 時分）

# ---- 冷房チューニング係数 ----
# T_set = A  - B*(Tavg-30) + C*(RH-55) + extra
A_COOL = 25.320513  # 基本オフセット
B_COOL = 0.320513   # 温度係数 (30°C を基点に逆比例)
C_COOL = -0.0384615 # 湿度係数 (55% を基点に負方向で下げる)

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def millis_until_midnight() -> int:
    """Return milliseconds until the next midnight in Asia/Tokyo."""
    now = datetime.now(TZ)
    nxt = datetime.combine(now.date() + timedelta(days=1), time(0), TZ)
    return int((nxt - now).total_seconds() * 1000)


def round_half_down(x: float) -> float:
    """丸め規則: 0.5 °C 刻みで *切り捨て*。例: 23.9 → 23.5"""
    return math.floor(x * 2) / 2

# -----------------------------------------------------------------------------
# GOOGLE SHEET (I/O)
# -----------------------------------------------------------------------------

def load_stores() -> pd.DataFrame:
    """Load store name & lat/lon from Google Sheet."""

    try:
        raw = pd.read_csv(SHEET_CSV, header=None)
    except Exception as exc:
        st.error(f"Google Sheet 読み込み失敗: {exc}")
        return pd.DataFrame(columns=["store", "lat", "lon"])

    header_idx = None
    for i, row in raw.iterrows():
        txt = " ".join(str(c).strip().lower() for c in row if pd.notna(c))
        if any(x in txt for x in ["store", "店舗", "店名"]):
            header_idx = i
            break
    if header_idx is None:
        st.error("ヘッダ行が見つかりません (store/lat/lon)")
        return pd.DataFrame(columns=["store", "lat", "lon"])

    df = pd.read_csv(SHEET_CSV, skiprows=header_idx, header=0)
    df.columns = [c.strip().lower() for c in df.columns]

    map_store = next((c for c in df.columns if c in {"store", "店舗", "店名"}), None)
    map_lat   = next((c for c in df.columns if c in {"lat", "latitude", "緯度"}), None)
    map_lon   = next((c for c in df.columns if c in {"lon", "lng", "long", "longitude", "経度"}), None)

    if not (map_store and map_lat and map_lon):
        st.error(f"必須列不足。現在列: {list(df.columns)}")
        return pd.DataFrame(columns=["store", "lat", "lon"])

    return (
        df[[map_store, map_lat, map_lon]]
        .rename(columns={map_store: "store", map_lat: "lat", map_lon: "lon"})
        .dropna()
    )

# -----------------------------------------------------------------------------
# WEATHER
# -----------------------------------------------------------------------------

def fetch_weather(lat: float, lon: float, date: str):
    """Fetch 24‑hour hourly weather data for *date* (ISO‑YYYY‑MM‑DD)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,weathercode",
        "timezone": "Asia/Tokyo",
        "start_date": date,
        "end_date": date,
    }
    try:
        resp = requests.get(WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        if "application/json" not in resp.headers.get("Content-Type", ""):
            st.warning("Open‑Meteo が JSON を返さなかったためスキップします。")
            return None
        return resp.json().get("hourly", {})
    except Exception as exc:
        st.error(f"Open‑Meteo 取得失敗: {exc}")
        return None


def summarize(hourly):
    """Return mean temperature & humidity for the HOURS window."""
    idx = [i for i, t in enumerate(hourly.get("time", [])) if int(t[11:13]) in HOURS]
    temps = np.array(hourly["temperature_2m"])[idx]
    hums  = np.array(hourly["relativehumidity_2m"])[idx]
    return temps.mean(), hums.mean()

# -----------------------------------------------------------------------------
# CONTROL LOGIC
# -----------------------------------------------------------------------------

def choose_mode(t_avg: float, rh_avg: float) -> str:
    """Determine HVAC mode by temperature & humidity.

    • 湿度 65 %以上 → 『除湿』を優先。
    • それ以外は気温ベース。
    """
    if rh_avg >= 65:
        return "除湿（ドライ）"
    if t_avg >= 25:
        return "冷房"
    if t_avg <= 15:
        return "暖房"
    if 21 < t_avg < 25:
        return "自動"
    return "送風"


# ---- 温度設定 ----

def _cooling_target(t_avg: float, rh_avg: float) -> float:
    """Calculate cooling target temperature (°C)."""
    base = A_COOL - B_COOL * (t_avg - 30) + C_COOL * (rh_avg - 55)

    # 追加湿度補正
    if rh_avg >= 60:
        base -= 0.5  # 除湿寄りに下げる
    if rh_avg <= 40:
        base += 0.5  # 乾燥時は上げる

    # 22–28 °C に収め、0.5 °C 刻みで *切り捨て*
    return round_half_down(np.clip(base, 22, 28))


def _heating_target(t_avg: float, rh_avg: float) -> float:
    base = 22 + 0.15 * (18 - t_avg) + 0.04 * (55 - rh_avg)

    if rh_avg >= 60:
        base -= 0.5
    if rh_avg <= 40:
        base += 0.5
    return round_half_down(np.clip(base, 20, 24))


def set_temp(t_avg: float, rh_avg: float, mode: str):
    """Return target temperature (°C) by mode or None (送風)."""
    if mode == "冷房":
        return _cooling_target(t_avg, rh_avg)
    if mode == "暖房":
        return _heating_target(t_avg, rh_avg)
    if mode == "自動":
        return 22.0 if rh_avg <= 40 else 23.5
    return None  # 送風

# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

def _inject_css():
    st.markdown(
        """
        <style>
        .big-bold {
            font-size: 32px;
            font-weight: bold;
            color: #c3d600;
        }
        .header-style {
            font-size: 40px;
            font-weight: bold;
            color: #c3d600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="FEEEP AC Settings", layout="centered")
    st_autorefresh(interval=millis_until_midnight(), key="midnight")

    today = datetime.now(TZ).date()
    st.title("FEEEP エアコン設定ガイド")
    st.caption(f"{today:%Y-%m-%d} の 10–17 時平均データを基に算出 (Open‑Meteo)")

    stores = load_stores()
    if stores.empty:
        st.stop()

    selected = st.selectbox("店舗を選択", stores["store"].tolist(), index=0)
    lat, lon = stores.loc[stores["store"] == selected, ["lat", "lon"]].iloc[0]

    hourly = fetch_weather(float(lat), float(lon), today.isoformat())
    if not hourly:
        st.warning("気象データ取得に失敗しました。")
        st.stop()

    t_avg, rh_avg = summarize(hourly)
    mode = choose_mode(t_avg, rh_avg)
    t_set = set_temp(t_avg, rh_avg, mode)

    _inject_css()

    st.subheader(selected)
    st.markdown(f"**平均気温**: {t_avg:.1f}°C")
    st.markdown(f"**平均湿度**: {rh_avg:.0f}%")
    st.markdown(
        f"**最適モード**: <span class='big-bold'>{mode}</span>",
        unsafe_allow_html=True,
    )

    temp_display = "設定不要" if t_set is None else f"<span class='big-bold'>{t_set:.1f}°C</span>"
    st.markdown(f"**設定温度**: {temp_display}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
