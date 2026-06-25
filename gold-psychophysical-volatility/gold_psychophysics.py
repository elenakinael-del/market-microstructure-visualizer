import pandas as pd
import numpy as np
import yfinance as yf

from fredapi import Fred

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import zscore

# ======================================================
# SETTINGS
# ======================================================

FRED_KEY = "f4cc710ee914042b0631cfbd3a4a6c7d"

START = "2008-01-01"

# ======================================================
# DOWNLOAD GOLD
# ======================================================

gold = yf.download(
    "GC=F",
    start=START,
    auto_adjust=True
)

gold.columns = gold.columns.get_level_values(0)

gold["Return"] = np.log(
    gold["Close"] /
    gold["Close"].shift(1)
)

# ======================================================
# REALIZED VOLATILITY
# ======================================================

gold["RV20"] = (
    gold["Return"]
    .rolling(20)
    .std()
    * np.sqrt(252)
)

gold["RV5_FWD"] = (
    gold["Return"]
    .rolling(5)
    .std()
    .shift(-5)
    * np.sqrt(252)
)

# ======================================================
# PSYCHOPHYSICAL VOLATILITY
# ======================================================

gold["VolRef"] = (
    gold["RV20"]
    .rolling(252)
    .mean()
)

gold["PerceivedVol"] = np.log(
    gold["RV20"] /
    gold["VolRef"]
)

gold["VolSurprise"] = np.log(
    gold["RV20"] /
    gold["RV20"].ewm(span=63).mean()
)

# ======================================================
# FRED DATA
# ======================================================

fred = Fred(api_key=FRED_KEY)

dxy = fred.get_series("DTWEXBGS")
us10 = fred.get_series("DGS10")

dxy = pd.DataFrame(dxy, columns=["DXY"])
us10 = pd.DataFrame(us10, columns=["US10Y"])

gold = gold.merge(
    dxy,
    left_index=True,
    right_index=True,
    how="left"
)

gold = gold.merge(
    us10,
    left_index=True,
    right_index=True,
    how="left"
)

gold = gold.ffill()

# ======================================================
# MANUAL COT INPUT
# ======================================================

managed_long = 128528
managed_short = 15610

swap_long = 29056
swap_short = 213010

producer_long = 17066
producer_short = 35714

other_long = 75636
other_short = 14169

gold["ManagedMoneyNet"] = (
    managed_long
    - managed_short
)

gold["SwapNet"] = (
    swap_long
    - swap_short
)

gold["ProducerNet"] = (
    producer_long
    - producer_short
)

gold["OtherNet"] = (
    other_long
    - other_short
)

# ======================================================
# POSITIONING PRESSURE INDEX
# ======================================================

gold["PositionPressure"] = (
    gold["ManagedMoneyNet"]
    - gold["SwapNet"]
)

# ======================================================
# PERCEPTION INDEX
# ======================================================

gold["PsychophysicalIndex"] = gold["PerceivedVol"]

# ======================================================
# MACHINE LEARNING DATASET
# ======================================================

features = [
    "RV20",
    "PerceivedVol",
    "VolSurprise",
    "PsychophysicalIndex",
    "DXY",
    "US10Y"
]

df = gold.copy()

print("\n========== DEBUG ==========")
print("Rows before dropna:", len(df))
print(df.isna().sum())

needed_cols = [
    "RV20",
    "PerceivedVol",
    "VolSurprise",
    "PsychophysicalIndex",
    "RV5_FWD"
]

df = df.dropna(subset=needed_cols)
print("Final rows:", len(df))

print("Rows after dropna:", len(df))
print("===========================\n")

print("After dropna:", len(df))

X = df[features]

y = df["RV5_FWD"]

# ======================================================
# TRAIN TEST
# ======================================================
print("Final dataframe shape:", df.shape)

if len(df) == 0:
    raise Exception("DATAFRAME EMPTY AFTER DROPNA")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=0.2
)

# ======================================================
# RANDOM FOREST
# ======================================================

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print(
    "RF R²:",
    r2_score(
        y_test,
        pred_rf
    )
)


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

imp = pd.DataFrame(
    {
        "Feature": features,
        "Importance": rf.feature_importances_
    }
)

imp = imp.sort_values(
    "Importance",
    ascending=False
)

print(imp)

# ======================================================
# INTERACTIVE CHART
# ======================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RV20"],
        name="Realized Vol"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["PerceivedVol"],
        name="Perceived Vol"
    )
)

fig.update_layout(
    title="Psychophysical Gold Volatility"
)

fig.write_html(
    "PsychophysicalVolatility.html"
)

# ======================================================
# 3D SURFACE
# ======================================================

sample = df.tail(500)

fig3d = go.Figure(
    data=[
        go.Scatter3d(
            x=sample["RV20"],
            y=sample["PerceivedVol"],
            z=sample["RV5_FWD"],
            mode="markers"
        )
    ]
)

fig3d.update_layout(
    title="3D Psychophysical Surface"
)

fig3d.write_html(
    "3D_Surface.html"
)

# ======================================================
# ANIMATION
# ======================================================

animation_df = df.tail(500)

fig_anim = px.scatter(
    animation_df,
    x="PerceivedVol",
    y="RV20",
    animation_frame=animation_df.index.astype(str),
    size=np.abs(
        animation_df["VolSurprise"]
    )
)

fig_anim.write_html(
    "PsychophysicalAnimation.html"
)

print(
    "Research completed"
)