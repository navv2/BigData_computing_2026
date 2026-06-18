
"""
app.py - 기대수명 예측 대시보드
조건 2~4: 모델 성능 비교 + 실시간 예측 UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="기대수명 예측 대시보드",
    page_icon="🏥",
    layout="wide",
)

# ── 한글 폰트 설정 (matplotlib) ────────────────────────────────────────────────
def set_korean_font():
    font_candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "DejaVu Sans"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
# Modified to use os.getcwd() as __file__ is not defined in Colab notebook cells
BASE_DIR   = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "Linear": "linear_pipeline.pkl",
    "Poly":   "poly_pipeline.pkl",
    "Ridge":  "ridge_pipeline.pkl",
}

# ── 데이터 및 모델 로드 (캐시) ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            st.error(f"모델 파일이 없습니다: {path}")
            st.stop()
        loaded[name] = joblib.load(path)
    return loaded

@st.cache_data
def load_data():
    try:
        X_test_path  = os.path.join(MODELS_DIR, "X_test.npy")
        y_test_path  = os.path.join(MODELS_DIR, "y_test.npy")
        X_train_path = os.path.join(MODELS_DIR, "X_train.npy")
        y_train_path = os.path.join(MODELS_DIR, "y_train.npy")
        meta_path    = os.path.join(MODELS_DIR, "feature_stats.json")

        X_test  = np.load(X_test_path)
        y_test  = np.load(y_test_path)
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return X_test, y_test, X_train, y_train, meta
    except FileNotFoundError as e:
        st.error(f"데이터 파일이 없습니다: {e}")
        st.stop()

models = load_models()
X_test, y_test, X_train, y_train, meta = load_data()
FEATURES     = meta["features"]
feature_stats = meta["stats"]

# ── 성능 데이터프레임 생성 ─────────────────────────────────────────────────────
@st.cache_data
def build_perf_df():
    rows = []
    for name, pipe in models.items():
        tr2  = r2_score(y_train,  pipe.predict(X_train))
        te2  = r2_score(y_test,   pipe.predict(X_test))
        tmse = mean_squared_error(y_train, pipe.predict(X_train))
        emse = mean_squared_error(y_test,  pipe.predict(X_test))
        if "poly" in pipe.named_steps:
            complexity = pipe.named_steps["poly"].n_output_features_
        else:
            complexity = len(FEATURES)
        rows.append({
            "모델":          name,
            "Train R²":     round(tr2,  4),
            "Test R²":      round(te2,  4),
            "Train MSE":    round(tmse, 4),
            "Test MSE":     round(emse, 4),
            "복잡도 (특성 수)": complexity,
        })
    return pd.DataFrame(rows)

perf_df = build_perf_df()

# ══════════════════════════════════════════════════════════════════════════════
# ── 헤더 ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg,#1a6b3a,#2196a6);
            padding:28px 32px; border-radius:14px; margin-bottom:24px;'>
  <h1 style='color:white; margin:0; font-size:2rem;'>🏥 기대수명 예측 대시보드</h1>
  <p style='color:#d0f0ff; margin:8px 0 0;'>
    WHO 기대수명 데이터 기반 · Linear / Poly / Ridge 회귀 모델 비교 & 실시간 예측
  </p>
</div>
""", unsafe_allow_html=True)

# ── 데이터 요약 배지 ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 데이터",   "1,649 행")
c2.metric("훈련 샘플 (50개)", f"{len(X_train)}개")
c3.metric("테스트 샘플",   f"{len(X_test)}개")
c4.metric("독립변수",       f"{len(FEATURES)}개")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── 섹션 1 : 모델 성능 비교 (조건 3) ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 모델 성능 비교")

tab1, tab2 = st.tabs(["📋 성능 지표 테이블", "📈 Test R² 막대 그래프"])

with tab1:
    st.markdown("### 훈련 / 테스트 R² · MSE · 복잡도 비교")

    # 색상 하이라이트: Test R² 기준
    def highlight_row(row):
        best  = perf_df["Test R²"].max()
        worst = perf_df["Test R²"].min()
        if row["Test R²"] == best:
            return ["background-color:#d4edda; color:#155724; font-weight:bold"] * len(row)
        elif row["Test R²"] == worst:
            return ["background-color:#f8d7da; color:#721c24"] * len(row)
        return [""] * len(row)

    styled = (
        perf_df.style
        .apply(highlight_row, axis=1)
        .format({
            "Train R²": "{:.4f}", "Test R²": "{:.4f}",
            "Train MSE": "{:.4f}", "Test MSE": "{:.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=160)

    # 과대적합 진단 설명
    poly_row = perf_df[perf_df["모델"] == "Poly"].iloc[0]
    st.info(
        f"**과대적합 진단** — Poly 모델의 Train R²={poly_row['Train R²']:.4f} (거의 1.0)이지만 "
        f"Test R²={poly_row['Test R²']:.4f}로 극단적으로 낮아 심각한 과대적합이 발생했습니다. "
        f"Ridge 규제를 적용하면 과대적합이 완화됩니다."
    )

with tab2:
    st.markdown("### 3종 모델 Test R² 비교 막대 그래프")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Linear": "#2196F3", "Poly": "#F44336", "Ridge": "#4CAF50"}
    model_names = perf_df["모델"].tolist()
    bar_colors  = [colors[m] for m in model_names]

    # ── 왼쪽: Test R² Bar Chart ──
    ax = axes[0]
    bars = ax.bar(model_names, perf_df["Test R²"], color=bar_colors,
                  edgecolor="white", linewidth=1.5, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, perf_df["Test R²"]):
        ypos = val + 0.03 if val >= 0 else val - 0.05
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Test R² Score 비교", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("R² (결정계수)", fontsize=11)
    ax.set_ylim(min(perf_df["Test R²"].min() - 0.1, -0.3), 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)

    # ── 오른쪽: Train vs Test R² 그룹 Bar Chart ──
    ax2 = axes[1]
    x    = np.arange(len(model_names))
    w    = 0.35
    b1   = ax2.bar(x - w/2, perf_df["Train R²"], w, label="Train R²",
                   color=["#90CAF9","#EF9A9A","#A5D6A7"], edgecolor="white")
    b2   = ax2.bar(x + w/2, perf_df["Test R²"],  w, label="Test R²",
                   color=["#1565C0","#B71C1C","#1B5E20"], edgecolor="white")
    for bar, val in zip(list(b1)+list(b2),
                        list(perf_df["Train R²"])+list(perf_df["Test R²"])):
        ypos = val + 0.02 if val >= 0 else val - 0.05
        ax2.text(bar.get_x()+bar.get_width()/2, ypos,
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(model_names)
    ax2.set_title("Train vs Test R² 비교", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("R² (결정계수)", fontsize=11)
    ax2.set_ylim(min(perf_df["Test R²"].min() - 0.1, -0.3), 1.15)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Test MSE 비교 (참고) ──
    st.markdown("#### Test MSE 비교 (낮을수록 좋음)")
    fig2, ax3 = plt.subplots(figsize=(7, 3.5))
    test_mse_vals = perf_df["Test MSE"].tolist()
    bars2 = ax3.bar(model_names, test_mse_vals, color=bar_colors,
                    edgecolor="white", linewidth=1.5, width=0.45)
    for bar, val in zip(bars2, test_mse_vals):
        ax3.text(bar.get_x()+bar.get_width()/2, val*1.02,
                 f"{val:,.1f}", ha="center", fontsize=11, fontweight="bold")
    ax3.set_title("Test MSE 비교", fontsize=13, fontweight="bold")
    ax3.set_ylabel("MSE")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── 섹션 2 : 실시간 예측 UI (조건 4) ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 실시간 기대수명 예측")

# ── 사이드바: 슬라이더 + 모델 선택 ────────────────────────────────────────────
st.sidebar.markdown("##  예측 입력값 설정")
st.sidebar.markdown("---")
st.sidebar.markdown("###  특성값 조절")

slider_values = {}
slider_config = {
    "Adult mortality": ("성인 사망률 (명/1000명)", 1.0,  723.0, 0),
    "BMI":             ("평균 BMI",               2.0,   77.1, 1),
    "GDP":             ("1인당 GDP (USD)",         1.0,119172.0, 0),
    "Alcohol":         ("1인당 알코올 소비 (리터)", 0.01, 17.87, 2),
    "Polio":           ("폴리오 예방접종률 (%)",    3.0,   99.0, 0),
}

for feat in FEATURES:
    label, mn, mx, decimals = slider_config[feat]
    mean_val = float(feature_stats[feat]["mean"])
    step     = 1.0 if decimals == 0 else (0.1 if decimals == 1 else 0.01)
    val = st.sidebar.slider(
        label,
        min_value=float(mn),
        max_value=float(mx),
        value=round(mean_val, decimals),
        step=step,
        format=f"%.{decimals}f",
    )
    slider_values[feat] = val
    st.sidebar.caption(f"평균값: {mean_val:.{decimals}f}  |  범위: {mn} ~ {mx}")
    st.sidebar.markdown("")

st.sidebar.markdown("---")
st.sidebar.markdown("###  모델 선택")
selected_model_name = st.sidebar.selectbox(
    "예측에 사용할 모델",
    options=list(MODEL_FILES.keys()),
    index=0,
    help="Linear: 선형 회귀 / Poly: 3차 다항 (과대적합) / Ridge: 3차 다항 + 릿지 규제",
)

model_desc = {
    "Linear": " 1차 선형 회귀 — 단순하고 안정적",
    "Poly":   " 3차 다항 회귀 — 훈련 데이터 과대적합 (참고용)",
    "Ridge":  " 3차 다항 + Ridge 규제 (α=1.0) — 과대적합 억제",
}
st.sidebar.info(model_desc[selected_model_name])

# ── 예측 수행 ──────────────────────────────────────────────────────────────────
input_arr   = np.array([[slider_values[f] for f in FEATURES]])
selected_pipe = models[selected_model_name]
pred_value  = float(selected_pipe.predict(input_arr)[0])
pred_value  = max(0.0, min(100.0, pred_value))   # 0~100 클리핑

# ── 예측 결과 출력 영역 ─────────────────────────────────────────────────────────
col_pred, col_info = st.columns([1, 1])

with col_pred:
    # 기대수명 구간 색상
    if pred_value >= 75:
        badge_color = "#2ecc71"; badge_label = "높음 "
    elif pred_value >= 65:
        badge_color = "#f39c12"; badge_label = "보통 "
    else:
        badge_color = "#e74c3c"; badge_label = "낮음 "

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                border-radius:18px; padding:36px 28px; text-align:center;
                box-shadow:0 8px 32px rgba(0,0,0,0.3); margin-bottom:16px;'>
      <p style='color:#a0c4ff; font-size:1.0rem; margin:0 0 8px;'>
         {selected_model_name} 모델 예측 결과
      </p>
      <h1 style='color:white; font-size:5rem; margin:0; font-weight:900;
                 text-shadow: 0 0 20px {badge_color};'>
        {pred_value:.1f}
      </h1>
      <p style='color:#a0c4ff; font-size:1.2rem; margin:8px 0 16px;'>세 (years)</p>
      <span style='background:{badge_color}; color:white; padding:6px 18px;
                   border-radius:20px; font-size:1rem; font-weight:bold;'>
        기대수명 수준: {badge_label}
      </span>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown("####  입력값 요약")
    input_df = pd.DataFrame({
        "특성":   [slider_config[f][0] for f in FEATURES],
        "입력값": [f"{slider_values[f]:.2f}" for f in FEATURES],
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True, height=215)

    # 선택 모델 성능 요약
    m_row = perf_df[perf_df["모델"] == selected_model_name].iloc[0]
    st.markdown(f"""
    **선택 모델 성능 요약**
    - Train R² : `{m_row['Train R²']:.4f}`  |  Test R² : `{m_row['Test R²']:.4f}`
    - Train MSE: `{m_row['Train MSE']:.4f}` |  Test MSE: `{m_row['Test MSE']:.4f}`
    - 변환 후 특성 수: `{m_row['복잡도 (특성 수)']}개`
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ── 섹션 3 : 모델 복잡도 이론 그래프 ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("##  모델 복잡도 vs 오차 이론")

complexity_vals = np.linspace(1, 10, 100)
train_err = 10 * np.exp(-0.4 * complexity_vals) + 0.5
test_err  = 8  * np.exp(-0.6 * complexity_vals) + 0.1 * (complexity_vals - 4)**2 + 2
opt_idx   = np.argmin(test_err)

fig3, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(complexity_vals, train_err, label="Training Error", color="#27ae60", lw=2.5)
ax4.plot(complexity_vals, test_err,  label="Test Error",    color="#2980b9", lw=2.5)
ax4.scatter(complexity_vals[opt_idx], test_err[opt_idx],
            color="#e74c3c", s=120, zorder=5)
ax4.annotate("Optimal Point",
             xy=(complexity_vals[opt_idx], test_err[opt_idx]),
             xytext=(complexity_vals[opt_idx]+0.6, test_err[opt_idx]+1.2),
             arrowprops=dict(arrowstyle="->", color="#e74c3c"),
             color="#e74c3c", fontweight="bold", fontsize=11)
ax4.fill_between(complexity_vals, train_err, test_err,
                 where=(complexity_vals > complexity_vals[opt_idx]),
                 alpha=0.12, color="#e74c3c", label="과대적합 구간")
ax4.fill_between(complexity_vals, train_err, test_err,
                 where=(complexity_vals < complexity_vals[opt_idx]),
                 alpha=0.12, color="#27ae60", label="과소적합 구간")
ax4.set_title("Theory: Model Complexity vs. Error", fontsize=14, fontweight="bold")
ax4.set_xlabel("Model Complexity (Low ← → High)")
ax4.set_ylabel("Error")
ax4.set_xticks(np.arange(1, 11))
ax4.legend(fontsize=10)
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.spines[["top","right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.caption(
    "Linear 모델은 과소적합 가능성이 있고, "
    "Poly 모델은 훈련 50개 샘플에서 극단적 과대적합이 발생합니다. "
    "Ridge는 규제를 통해 분산을 줄여 균형점에 가까워집니다."
)
