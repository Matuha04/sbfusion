import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

errors = []

# ─── CONFIG ───────────────────────────────────────────────────────────────
BAZAAR_API    = "https://api.hypixel.net/skyblock/bazaar"
META_FILE     = "shard_metadata.csv"
HISTORY_FILE  = "fusion_history.csv"
METRIC_COLS   = [
    "date", "result", "inputs", "output_qty",
    "cost", "revenue", "profit",
    "demand_history", "demand_live", "score"
]
RARITY_ORDER  = ["common","uncommon","rare","epic","legendary"]

# ─── LOAD & NORMALIZE METADATA ────────────────────────────────────────────
meta = pd.read_csv(META_FILE).rename(columns=str.lower)
meta["rarity"]   = meta["rarity"].str.lower().str.strip()
meta["category"] = meta["category"].str.lower().str.strip()
meta["families"] = (
    meta["families"]
      .fillna("")
      .apply(lambda s: [f.strip().lower() for f in s.split(",") if f.strip()])
)
meta["fusion_qty"] = meta.get("fusion_qty", 5).fillna(5).astype(int)
qty_map = dict(zip(meta["item_id"], meta["fusion_qty"]))

# ─── FETCH BAZAAR ──────────────────────────────────────────────────────────
def fetch_bazaar():
    r = requests.get(BAZAAR_API); r.raise_for_status()
    out = r.json()
    if not out.get("success"):
        raise RuntimeError("Bazaar API error")
    return out["products"]

bazaar = {}

def get_price(item, mode, kind):
    data = bazaar[str(item)]
    if kind=="buy":
        bucket = data["sell_summary"] if mode=="order" else data["buy_summary"]
        return max(x["pricePerUnit"] for x in bucket) if mode=="order" else min(x["pricePerUnit"] for x in bucket)
    if kind=="sell":
        return min(x["pricePerUnit"] for x in data["buy_summary"])
    raise ValueError(kind)

# ─── LOAD RECIPES CSV ─────────────────────────────────────────────────────
def load_recipes(path):
    df = pd.read_csv(path)
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    for c in df.select_dtypes("object"):
        df[c] = df[c].str.strip()
    return df.rename(columns=str.lower)

# ─── PICK WILDCARDED SHARD ────────────────────────────────────────────────
def pick_shard(rarity=None, category=None, family=None, exclude_ids=None):
    df2 = meta.copy()
    if isinstance(rarity, str):
        r = rarity.lower().strip()
        if r.endswith("+"):
            idx = RARITY_ORDER.index(r[:-1])
            df2 = df2[df2["rarity"].apply(lambda x: RARITY_ORDER.index(x) >= idx)]
        else:
            df2 = df2[df2["rarity"]==r]
    if isinstance(category, str):
        df2 = df2[df2["category"]==category.lower().strip()]
    if isinstance(family, str):
        f = family.lower().strip()
        df2 = df2[df2["families"].apply(lambda fams: f in fams)]
    if exclude_ids:
        df2 = df2[~df2["item_id"].isin(exclude_ids)]

    # keep only bazaar‐listed
    df2 = df2[df2["item_id"].isin(bazaar)]
    # find lowest‐cost order buy
    candidates = []
    for iid in df2["item_id"]:
        try:
            p = get_price(iid, "order", "buy")
            candidates.append((iid,p))
        except:
            pass
    if not candidates:
        raise KeyError("No matching shards for pick_shard()")
    return min(candidates, key=lambda x: x[1])[0]


# ─── COMPUTE METRICS ──────────────────────────────────────────────────────
def compute_metrics(df, buy_mode="order"):
    """
    Walk through each recipe row in df, resolve inputs (including wildcards),
    compute cost, revenue, profit, and return a DataFrame with the core columns.
    """
    global errors
    errors = []
    rows   = []
    today  = datetime.date.today()

    for _, r in df.iterrows():
        result   = r["result"]
        # output_qty (default to 1) and min_profit (default to 0)
        out_q    = int(r["output_qty"]) if pd.notna(r["output_qty"]) else 1
        prof_min = float(r["min_profit"]) if pd.notna(r["min_profit"]) else 0.0

        total_cost = 0
        inputs     = []
        skip       = False

        # resolve the two inputs
        for i in (1, 2):
            try:
                # either explicit ID or wildcard
                raw_id = r.get(f"input{i}")
                if pd.notna(raw_id):
                    sid = raw_id
                else:
                    sid = pick_shard(
                        r.get(f"input{i}_rarity"),
                        r.get(f"input{i}_category"),
                        r.get(f"input{i}_family"),
                        exclude_ids=(
                            [x.strip() for x in str(r.get(f"input{i}_exclude")).split(",")
                             if x.strip()]
                            if pd.notna(r.get(f"input{i}_exclude")) else None
                        )
                    )

                # safe quantity parsing
                raw_qty = r.get(f"qty{i}")
                if pd.notna(raw_qty) and int(raw_qty) > 0:
                    qty_i = int(raw_qty)
                else:
                    qty_i = qty_map.get(sid, 1)

                inputs.append((sid, qty_i))
                total_cost += get_price(sid, buy_mode, "buy") * qty_i

            except Exception as e:
                errors.append(f"Recipe '{result}', input {i}: {e}")
                skip = True
                break

        if skip:
            continue

        # compute revenue & profit
        try:
            revenue = get_price(result, buy_mode, "sell") * out_q
        except Exception as e:
            errors.append(f"Recipe '{result}', revenue fetch: {e}")
            continue

        profit = revenue - total_cost
        if profit < prof_min:
            continue

        # build row dict
        rows.append({
            "date":          today,
            "result":        result,
            "inputs":        ", ".join(f"{q}×{sid}" for sid, q in inputs),
            "output_qty":    out_q,
            "cost":          total_cost,
            "revenue":       revenue,
            "profit":        profit,
            # demand_history, demand_live, score added in UI()
        })

    return pd.DataFrame(rows, columns=METRIC_COLS)


# ─── HISTORY LOGGING ──────────────────────────────────────────────────────
def append_history(df):
    header = not os.path.isfile(HISTORY_FILE)
    df.to_csv(HISTORY_FILE, mode="a", header=header, index=False)

# ─── CACHED BAZAAR FETCH ────────────────────────────────────────────
@st.cache_data(ttl=10)
def fetch_bazaar_cached():
    return fetch_bazaar()

def ui():
    global bazaar

    # ─── USER-SELECTABLE REFRESH INTERVAL ─────────────────────────────
    REFRESH_SEC = st.sidebar.selectbox(
        "Refresh every…",
        [10, 30, 60],
        index=0   # default to 30s
    )

    # 1) soft rerun every second for countdown
    # 1) rerun the script at your chosen interval
    tick = st_autorefresh(interval=REFRESH_SEC * 1000, limit=None, key="fusion_timer")

    # 2) track when we last actually fetched
    now = time.time()
    if "last_fetch" not in st.session_state:
        # on first run, mark last_fetch = now so it doesn't immediately refetch
        st.session_state["last_fetch"] = now

    # if it's been >= REFRESH_SEC since our last real fetch, pull again
    if now - st.session_state["last_fetch"] >= REFRESH_SEC:
        bazaar = fetch_bazaar_cached()    # do your fetch here
        st.session_state["last_fetch"] = time.time()
    
    # 3) fetch bazaar (cached to 60s)
    bazaar = fetch_bazaar_cached()

    # ─── BUILD A MASTER-STOCK SNAPSHOT FROM THE WHOLE BAZAAR ─────────────
    master_stock = {
        item: sum(e["amount"] for e in data["buy_summary"])
        for item, data in bazaar.items()
    }

    # 4) page layout & filters
    st.set_page_config(page_title="Shard Fusion Profits", layout="wide")
    st.title("Shard Fusion Profits")
    search = st.text_input("🔍 Filter for specific shard/result", "")
    RECIPES_FILE = "recipes.csv"
    if not os.path.exists(RECIPES_FILE):
        st.error(f"⚠️ Can't find '{RECIPES_FILE}'")
        return
    df = load_recipes(RECIPES_FILE)
    if search:
        df = df[df["result"].str.contains(search, case=False, na=False)]

    min_profit = st.sidebar.number_input("Min profit", 0.0, step=1000000.0)
    sort_by = st.sidebar.selectbox("Sort by",
            ["profit", "demand_live", "score", "demand_change"],)
    top_n      = st.sidebar.slider("Top N", 1, 50, 50)
    buy_mode   = st.sidebar.selectbox("Buy Mode", ["order", "instabuy"])

    # 5) compute base metrics & initial display set
    mets = compute_metrics(df, buy_mode=buy_mode)
    mets = mets[mets["profit"] >= min_profit]
    # hold all candidates for now, we'll sort *after* filling all cols
    disp = mets.copy()    

    # 6) demand_history = delta from last snapshot
    if "last_stock" not in st.session_state:
        st.session_state.last_stock = {}

    # … after you build new_stock = {} but before using session_state.initial_stock …
    new_stock = {}
    
    # ── BOOTSTRAP “SINCE START” & “LAST” SNAPSHOTS ─────────────
    if "initial_stock" not in st.session_state:
        # capture the very first state of the bazaar
        st.session_state["initial_stock"] = master_stock.copy()

    # now you can safely read/write them

    for idx, row in disp.iterrows():
        item     = str(row["result"])
        curr_amt = master_stock[item]
        init_amt = st.session_state["initial_stock"].get(item, curr_amt)
        # how many units have disappeared since script start
        disp.at[idx, "demand_history"] = init_amt - curr_amt
        new_stock[item] = curr_amt

    # bootstrap both first-run baselines
    if "initial_stock" not in st.session_state:
       st.session_state.initial_stock = new_stock.copy()


    # 7) fill remaining columns & score
    # 1) keep a numeric version for sorting
    disp["demand_change_num"] = disp["demand_history"]
    # 2) build the pretty string for display only
    disp["demand_change"] = disp["demand_change_num"].apply(
        lambda x: f"🔻 {abs(x)}" if x < 0 else f"🟢 {x}"    
    )
    disp["demand_live"]    = disp["result"].map(
        lambda it: sum(e["amount"] for e in bazaar[str(it)]["buy_summary"])
    ).astype(int)
    disp["score"] = disp.apply(
    lambda r: r["profit"] * np.log1p(r["demand_history"])
    if r["demand_history"] > -1 else None,
    axis=1
    )


    # 8) sort & slice
    # if sorting by demand_change, use the numeric column
    sort_col = "demand_change_num" if sort_by == "demand_change" else sort_by
    disp = disp.sort_values(by=sort_col, ascending=False).head(top_n)

    # 9) REORDER COLUMNS (hard-code your desired order)
    disp = disp[
      ["result","inputs","output_qty",
       "cost","revenue","profit",
       "demand_change",
       "demand_live","score"]
    ]

    # 10) render
    with st.container():
        st.markdown("### 📊 Top Fusion Opportunities")
        st.dataframe(
            disp,
            use_container_width=True,
            height=2000
        )

    st.subheader("Errors")
    if errors:
        for e in errors:
            st.text(f"• {e}")
    else:
        st.text("No errors encountered.")
    append_history(disp)


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui()
