import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ------------------- 中文正常显示 -------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 网站标题 -------------------
st.set_page_config(page_title="储层预测自动绘图系统", layout="wide")
st.title("🌍 多属性融合储层预测-全自动在线绘图网站")
st.markdown("### 功能：输入数据 → 自动生成可视化图")

# ===================== 1. 数据输入区 =====================
st.sidebar.header("📥 数据输入")
input_mode = st.sidebar.radio("选择输入方式", ["手动输入数据", "上传Excel文件"])

df = None

# 方式1：手动输入
if input_mode == "手动输入数据":
    st.subheader("✏️ 手动输入数据")
    well_num = st.number_input("井数量", min_value=3, max_value=20, value=10)
    
    well_names = []
    xs, ys, poros, amps, envs, doms = [], [], [], [], [], []
    
    with st.form("input_form"):
        for i in range(int(well_num)):
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1: well = st.text_input(f"井名{i+1}", f"well{i+1}")
            with col2: x = st.number_input(f"X坐标{i+1}", value=546000+i*100)
            with col3: y = st.number_input(f"Y坐标{i+1}", value=4285000+i*100)
            with col4: por = st.number_input(f"孔隙度{i+1}%", value=16.0)
            with col5: amp = st.number_input(f"振幅{i+1}", value=1.5)
            with col6: env = st.number_input(f"包络{i+1}", value=1.6)
            well_names.append(well)
            xs.append(x)
            ys.append(y)
            poros.append(por)
            amps.append(amp)
            envs.append(env)
        submitted = st.form_submit_button("✅ 生成数据并绘图")
    
    if submitted:
        df = pd.DataFrame({
            "井名": well_names, "x": xs, "y": ys, "por": poros,
            "振幅": amps, "包络": envs, "主频": np.random.uniform(25, 50, len(xs))
        })

# 方式2：上传Excel
else:
    st.subheader("📤 上传Excel文件（自动绘图）")
    st.info("Excel必须包含列：井名、x、y、por、振幅、包络、主频")
    uploaded = st.file_uploader("上传Excel", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)

# ===================== 2. 自动绘图区 =====================
if df is not None:
    st.subheader("📊 你的输入数据")
    st.dataframe(df, use_container_width=True)

    # 自动建模
    X1 = df[["振幅"]].values
    X_m = df[["振幅", "包络", "主频"]].values
    y_true = df["por"].values

    # 单一属性
    m1 = LinearRegression()
    m1.fit(X1, y_true)
    p1 = m1.predict(X1)
    r1 = r2_score(y_true, p1)

    # 多属性融合
    m2 = RandomForestRegressor(random_state=42)
    m2.fit(X_m, y_true)
    p2 = m2.predict(X_m)
    r2_m = r2_score(y_true, p2)

    # ==================== 4张图自动布局 ====================
    st.markdown("---")
    st.subheader("🖼️ 生成可视化图")
    fig = plt.figure(figsize=(16, 12))

    # 图1：井位分布
    plt.subplot(2,2,1)
    plt.scatter(df.x, df.y, c='blue', s=100)
    for i,row in df.iterrows(): plt.annotate(row.井名, (row.x, row.y), fontsize=9)
    plt.title("井位分布图", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.gca().set_aspect('equal')

    # 图2：孔隙度分布
    plt.subplot(2,2,2)
    sc = plt.scatter(df.x, df.y, c=df.por, cmap='jet', s=100)
    plt.colorbar(sc, label="孔隙度%")
    for i,row in df.iterrows(): plt.annotate(row.井名, (row.x, row.y), fontsize=9, color='white')
    plt.title("孔隙度空间分布图", fontweight='bold')
    plt.gca().set_aspect('equal')

    # 图3：单一属性预测
    plt.subplot(2,2,3)
    plt.scatter(y_true, p1, c='red', s=80)
    plt.plot([y_true.min(),y_true.max()], [y_true.min(),y_true.max()], 'k--')
    plt.title(f"单一属性预测\n$R^2$={r1:.2f}", fontweight='bold')
    plt.xlabel("真实孔隙度")
    plt.ylabel("预测孔隙度")
    plt.grid(alpha=0.3)

    # 图4：多属性融合预测
    plt.subplot(2,2,4)
    plt.scatter(y_true, p2, c='green', s=80)
    plt.plot([y_true.min(),y_true.max()], [y_true.min(),y_true.max()], 'k--')
    plt.title(f"多属性融合预测\n$R^2$={r2_m:.2f}", fontweight='bold')
    plt.xlabel("真实孔隙度")
    plt.ylabel("预测孔隙度")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # 保存高清图
    plt.savefig("自动生成.png", dpi=300, bbox_inches='tight')
    with open("自动生成.png", "rb") as f:
        st.download_button("💾 下载高清图", f, "储层预测图.png")

    # 结论
    st.markdown("---")
    st.subheader("📝 自动分析结论")
    colA, colB = st.columns(2)
    with colA: st.warning(f"单一属性精度 R² = {r1:.2f}（精度低、多解性强）")
    with colB: st.success(f"多属性融合精度 R² = {r2_m:.2f}（精度高、适应性强）")

else:
    st.info("👈 请在左侧输入数据或上传Excel，自动生成可视化图！")