import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
from datetime import datetime, timedelta, time
from streamlit_autorefresh import st_autorefresh

SHEET_CSV = "https://docs.google.com/spreadsheets/d/1WcO_36bZ53qUFa1YR5PyJvYyL9jGxABU5PP4OiiS-Sw/export?gid=0&format=csv"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
TZ = pytz.timezone("Asia/Tokyo")
HOURS = range(10, 19)  # 10–18

# ----------------------------------
# Utils
# ----------------------------------

def millis_until_midnight():
    now = datetime.now(TZ)
    nxt = datetime.combine(now.date() + timedelta(days=1), time(0), TZ)
    return int((nxt - now).total_seconds() * 1000)

# ----------------------------------
# Load stores
# ----------------------------------

def load_stores() -> pd.DataFrame:
    try:
        raw = pd.read_csv(SHEET_CSV, header=None)
    except Exception as e:
        st.error(f"Google Sheet 読み込み失敗: {e}")
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

    df = df[[map_store, map_lat, map_lon]].rename(columns={map_store:"store", map_lat:"lat", map_lon:"lon"}).dropna()
    return df

# ----------------------------------
# Weather
# ----------------------------------

def fetch_weather(lat: float, lon: float, date: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,weathercode",
        "timezone": "Asia/Tokyo",
        "start_date": date,
        "end_date": date,
    }
    try:
        r = requests.get(WEATHER_API, params=params, timeout=10)
        r.raise_for_status()
        if "application/json" not in r.headers.get("Content-Type", ""):
            st.warning("Open‑Meteo が JSON を返さなかったためスキップします。")
            return None
        return r.json().get("hourly", {})
    except Exception as e:
        st.error(f"Open‑Meteo 取得失敗: {e}")
        return None


def summarize(hourly):
    idx = [i for i, t in enumerate(hourly.get("time", [])) if int(t[11:13]) in HOURS]
    temps = np.array(hourly["temperature_2m"])[idx]
    hums  = np.array(hourly["relativehumidity_2m"])[idx]
    return temps.mean(), hums.mean()

# ----------------------------------
# Logic
# ----------------------------------

def choose_mode(t_avg: float) -> str:
    if t_avg >= 25:
        return "冷房"
    if t_avg <= 15:
        return "暖房"
    if 21 < t_avg < 25:
        return "自動"
    return "送風"


def set_temp(t_avg: float, rh_avg: float, mode: str):
    if mode == "冷房":
        t = 26 - 0.15*(t_avg-30) - 0.04*(rh_avg-55)
        if rh_avg>=60: t-=0.5
        if rh_avg<=40: t+=0.5
        return round(float(np.clip(t,24,28))*2)/2
    if mode == "暖房":
        t = 22 + 0.15*(18-t_avg) + 0.04*(55-rh_avg)
        if rh_avg>=60: t-=0.5
        if rh_avg<=40: t+=0.5
        return round(float(np.clip(t,20,24))*2)/2
    if mode == "自動":
        return 24.0
    return None  # 送風

# ----------------------------------
# Streamlit
# ----------------------------------

def set_display_style():
    st.markdown(
        """
        <style>
        .big-bold {
            font-size: 32px;
            font-weight: bold;
            color: #c3d600;  /* 設定温度の強調カラーを #c3d600に設定 */
        }
        .header-style {
            font-size: 40px;
            font-weight: bold;
            color: #c3d600;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="FEEEP AC Settings", layout="centered")
    st_autorefresh(interval=millis_until_midnight(), key="midnight")

    today = datetime.now(TZ).date()
    st.title("FEEEP エアコン設定ガイド")
    st.caption(f"{today} の 10–18 時平均データを基に算出 (Open‑Meteo)")

    stores = load_stores()
    if stores.empty:
        st.stop()

    store_names = stores["store"].tolist()
    selected = st.selectbox("店舗を選択", store_names, index=0)

    row = stores[stores["store"] == selected].iloc[0]
    lat, lon = float(row["lat"]), float(row["lon"])

    hourly = fetch_weather(lat, lon, today.isoformat())
    if not hourly:
        st.warning("気象データ取得に失敗しました。")
        st.stop()

    t_avg, rh_avg = summarize(hourly)
    mode = choose_mode(t_avg)
    t_set = set_temp(t_avg, rh_avg, mode)

    set_display_style()

    st.subheader(selected)
    st.markdown(f"**平均気温**: {t_avg:.1f}°C", unsafe_allow_html=True)
    st.markdown(f"**平均湿度**: {rh_avg:.0f}%", unsafe_allow_html=True)
    
    # 運転モードと設定温度の強調表示
    st.markdown(f"**最適モード**: <span class='big-bold'>{mode}</span>", unsafe_allow_html=True)
    
    # 設定温度の強調表示
    temp_display = "設定不要 (送風)" if t_set is None else f"<span class='big-bold'>{t_set:.1f}°C</span>"
    st.markdown(f"**設定温度**: {temp_display}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
