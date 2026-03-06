import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ページ設定
st.set_page_config(page_title="鋼材曲げ3D：高精度ソリッド＆解析版", layout="wide")
st.title("🏗️ 鋼材曲げ3D：高精度ソリッド＆解析版")

# --- 1. 基本設定 ---
st.sidebar.header("🛠️ 1. 基本設定")
st.sidebar.info("💡 スマホ操作：指1本で回転、指2本で拡大縮小")
shape_type = st.sidebar.selectbox("曲げ形状", ["L型", "コの字型", "Z型 (段曲げ)", "ハット型"])
mat_type = st.sidebar.selectbox("材質", ["鉄 (7.85)", "ステンレス (7.93)", "アルミ (2.70)"])
densities = {"鉄 (7.85)": 7.85, "ステンレス (7.93)": 7.93, "アルミ (2.70)": 2.70}

t = st.sidebar.number_input("板厚 (t) [mm]", value=3.2, min_value=0.1, step=0.1)
width_plate = st.sidebar.number_input("横幅 (W) [mm]", value=100.0, step=10.0)
r = st.sidebar.number_input("内R (r) [mm]", value=2.0, min_value=0.1, step=0.1)
k = st.sidebar.slider("中立軸係数 (k)", 0.2, 0.6, 0.35, 0.01)

st.sidebar.divider()
st.sidebar.header("📐 2. 各辺と角度")

if shape_type == "L型":
    b_vals, a_vals, labels, dirs, b_idx = [50.0, 60.0], [90.0], ["B1", "B2"], [1], 0
elif shape_type == "コの字型":
    b_vals, a_vals, labels, dirs, b_idx = [50.0, 150.0, 50.0], [90.0, 90.0], ["B1", "底辺 (A)", "B2"], [1, 1], 1
elif shape_type == "Z型 (段曲げ)":
    b_vals, a_vals, labels, dirs, b_idx = [50.0, 50.0, 50.0], [45.0, 45.0], ["B1", "立ち上がり (A)", "B2"], [1, -1], 0
elif shape_type == "ハット型":
    b_vals, a_vals, labels, dirs, b_idx = [30.0, 50.0, 100.0, 50.0, 30.0], [90.0, 90.0, 90.0, 90.0], ["B1", "B2", "A", "B3", "B4"], [1, 1, 1, 1], 2

for i in range(len(b_vals)): b_vals[i] = st.sidebar.number_input(f"{labels[i]}", value=float(b_vals[i]))
for i in range(len(a_vals)): a_vals[i] = st.sidebar.number_input(f"角度{i+1}", value=float(a_vals[i]))

# --- 2. 座標計算ロジック ---
def get_solid_profiles(t_val, r_val, b_list, a_list, dirs_list, base_index):
    div = 40
    def gen_path():
        pts, angs = [], []
        cp, ca = np.array([0.0, 0.0]), 0.0
        for i in range(len(b_list)):
            op = (r_val + t_val) * np.tan(np.radians(a_list[i-1])/2) if i > 0 else 0
            on = (r_val + t_val) * np.tan(np.radians(a_list[i])/2) if i < len(a_list) else 0
            slen = max(0.1, b_list[i] - op - on)
            pts.append(cp.copy()); angs.append(ca)
            cp += [slen * np.cos(ca), slen * np.sin(ca)]
            pts.append(cp.copy()); angs.append(ca)
            if i < len(a_list):
                d, ar = dirs_list[i], np.radians(a_list[i])
                center = cp + [(r_val + t_val) * np.cos(ca + d * np.pi/2), (r_val + t_val) * np.sin(ca + d * np.pi/2)]
                s_phi = ca - d * np.pi/2
                for phi in np.linspace(s_phi, s_phi + d * ar, div):
                    pts.append(center + [(r_val + t_val) * np.cos(phi), (r_val + t_val) * np.sin(phi)])
                    angs.append(phi + d * np.pi/2)
                ca += d * ar; cp = pts[-1]
        return np.array(pts), np.array(angs)

    bp, ba = gen_path()
    po, pi = [], []
    for p, ang in zip(bp, ba):
        norm = np.array([-np.sin(ang), np.cos(ang)])
        po.append(p); pi.append(p + norm * t_val)
    po, pi = np.array(po), np.array(pi)
    idx = base_index * (2 + div)
    rot = -np.arctan2(po[idx+1,1]-po[idx,1], po[idx+1,0]-po[idx,0])
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    po, pi = np.dot(po, R.T), np.dot(pi, R.T)
    z_min = np.min(po[:,1])
    return po - [po[idx,0], z_min], pi - [po[idx,0], z_min]

p_o, p_i = get_solid_profiles(t, r, b_vals, a_vals, dirs, b_idx)

# --- 3. 寸法解析・重量表示 ---
analysis = []
total_slen = 0
for i, val in enumerate(b_vals):
    op = (r+t)*np.tan(np.radians(a_vals[i-1])/2) if i > 0 else 0
    on = (r+t)*np.tan(np.radians(a_vals[i])/2) if i < len(a_vals) else 0
    slen = max(0.1, val - op - on)
    total_slen += slen
    analysis.append({"部位": labels[i], "入力外寸": f"{val:.2f}", "直線部長さ": f"{slen:.2f}"})

total_ba = sum([2 * np.pi * (r + k * t) * (ang / 360) for ang in a_vals])
total_L = total_slen + total_ba
weight = total_L * width_plate * t * densities[mat_type] / 1_000_000

with st.expander("📊 寸法解析結果・重量", expanded=True):
    col_a, col_b = st.columns([2, 1])
    col_a.table(pd.DataFrame(analysis))
    col_b.metric("総展開長 (L)", f"{total_L:.2f} mm")
    col_b.metric("概算重量", f"{weight:.2f} kg")

# --- 4. 視点ボタン ---
st.write("### 📸 視点切り替え")
c = st.columns(4)
if 'cam' not in st.session_state: st.session_state.cam = dict(eye=dict(x=1.5, y=1.5, z=1.2))

if c[0].button("📐 アイソメ"): st.session_state.cam = dict(eye=dict(x=1.5, y=1.5, z=1.2))
if c[1].button("🔝 平面"): st.session_state.cam = dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
if c[2].button("正面"): st.session_state.cam = dict(eye=dict(x=0, y=-2.5, z=0))
if c[3].button("側面"): st.session_state.cam = dict(eye=dict(x=2.5, y=0, z=0))

# --- 5. 3D描画 ---
munsell_out = '#747A4A'
munsell_in  = '#8B9165'
munsell_edge = '#4B4F30'

fig = go.Figure()
y_v = [0, width_plate]

# 表面・裏面
for p, color, name in [(p_o, munsell_out, '外面'), (p_i, munsell_in, '内面')]:
    X, Y = np.meshgrid(p[:,0], y_v)
    Z, _ = np.meshgrid(p[:,1], y_v)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,color],[1,color]], showscale=False, name=name, lighting=dict(ambient=0.6)))

# 側面（厚み部分）を密集線で塗りつぶす
for y in y_v:
    for i in range(len(p_o)):
        fig.add_trace(go.Scatter3d(
            x=[p_o[i,0], p_i[i,0]], y=[y, y], z=[p_o[i,1], p_i[i,1]],
            mode='lines', line=dict(color=munsell_edge, width=4), showlegend=False
        ))
    fig.add_trace(go.Scatter3d(x=p_o[:,0], y=[y]*len(p_o), z=p_o[:,1], mode='lines', line=dict(color='black', width=3), showlegend=False))
    fig.add_trace(go.Scatter3d(x=p_i[:,0], y=[y]*len(p_i), z=p_i[:,1], mode='lines', line=dict(color='black', width=3), showlegend=False))

# 前後の断面エッジ
for i in [0, -1]:
    for p in [p_o, p_i]:
        fig.add_trace(go.Scatter3d(x=[p[i,0]]*2, y=y_v, z=[p[i,1]]*2, mode='lines', line=dict(color='black', width=3), showlegend=False))

# レイアウト設定
fig.update_layout(
    scene=dict(
        xaxis_title="長さ", yaxis_title="幅(W)", zaxis_title="高さ",
        aspectmode='data', 
        camera=st.session_state.cam,
        camera_projection=dict(type='perspective')
    ),
    modebar=dict(
        orientation='v',
        bgcolor='rgba(255,255,255,0.7)',
        activecolor='#747A4A'
    ),
    dragmode="orbit", 
    doubleClick="reset",
    showlegend=False, 
    height=700, 
    margin=dict(l=0, r=0, b=0, t=0),
    hovermode=False
)

# Plotlyのコンフィグ設定
config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['zoomIn3d', 'zoomOut3d', 'resetCameraLastSave3d']
}

st.plotly_chart(fig, use_container_width=True, config=config)
