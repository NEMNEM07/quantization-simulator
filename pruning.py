import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. 페이지 설정 및 커스텀 CSS
# ==========================================
st.set_page_config(page_title="AI Pruning Simulator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    body { font-family: 'JetBrains Mono', monospace; }
    
    .info-box {
        background-color: #1e1e1e; padding: 20px; border-radius: 12px;
        border-left: 6px solid #1f77b4; margin-top: 10px; margin-bottom: 20px;
    }
    .highlight { color: #FF4B4B; font-weight: bold; }
    .wanda-color { color: #2ca02c; font-weight: bold; }
    .sparse-color { color: #ff7f0e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 난수 시드 고정 및 데이터 생성 (일관된 시각화를 위해)
# ==========================================
np.random.seed(42)

# 가상의 가중치 행렬 (10x10)
W_orig = np.random.randn(10, 10) * 0.5

# 가상의 입력 활성화(Activation) 벡터 (크기 10)
# 특정 채널(3, 7번)에 활성화 값이 극단적으로 쏠려있다고 가정 (LLM의 특징인 Outlier Feature)
X_act = np.random.uniform(0.5, 1.5, 10)
X_act[3] = 15.0
X_act[7] = 12.0

# ==========================================
# 3. 사이드바 UI 컨트롤
# ==========================================
st.sidebar.title("Pruning 컨트롤 패널")
st.sidebar.markdown("가지치기 시뮬레이터")

st.sidebar.markdown("---")
sparsity = st.sidebar.slider("희소성 (Sparsity) 비율", min_value=10, max_value=90, value=50, step=10, format="%d%%")
sparsity_ratio = sparsity / 100.0

prune_method = st.sidebar.radio(
    "가지치기 기법 선택",
    ["Magnitude (크기 기반)", "Wanda (가중치 × 활성화)", "SparseGPT (오차 보상)"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**데이터 특성 가정:**
현재 입력 데이터의 4번째, 8번째 특징(Feature)에 활성화 값이 비정상적으로 높게 나타나는 **Outlier 현상**이 존재합니다. (LLM의 전형적 특징)
""")

# ==========================================
# 4. 가지치기 로직 구현
# ==========================================
W_pruned = np.copy(W_orig)
mask = np.ones_like(W_orig, dtype=bool)

if prune_method == "Magnitude (크기 기반)":
    # 단순히 가중치의 절댓값이 작은 것을 자름
    threshold = np.percentile(np.abs(W_orig), sparsity)
    mask = np.abs(W_orig) >= threshold
    W_pruned = W_orig * mask

elif prune_method == "Wanda (가중치 × 활성화)":
    # 가중치 절댓값과 해당 입력 활성화 값의 곱(Score)을 기준으로 자름
    # W_orig 행렬의 각 열(입력 채널)에 X_act를 곱함
    score = np.abs(W_orig) * X_act
    threshold = np.percentile(score, sparsity)
    mask = score >= threshold
    W_pruned = W_orig * mask

elif prune_method == "SparseGPT (오차 보상)":
    # 크기 기반으로 자르되, 잘려나간 가중치들의 오차를 남은 가중치에 분배 (시뮬레이션 단순화)
    threshold = np.percentile(np.abs(W_orig), sparsity)
    mask = np.abs(W_orig) >= threshold
    W_pruned = W_orig * mask
    
    # [시각화용 트릭] 잘려나간 값들의 합을 구해서, 살아남은 가중치들에 더해줌 (오차 보상 효과 시뮬레이션)
    for row in range(10):
        pruned_sum = np.sum(W_orig[row, ~mask[row]])
        survivors = np.sum(mask[row])
        if survivors > 0:
            compensation = pruned_sum / survivors
            W_pruned[row, mask[row]] += compensation

# ==========================================
# 5. 메인 화면 UI 및 행렬 시각화
# ==========================================
st.title("신경망 가지치기(Pruning) 시뮬레이터")

# 기법별 설명 박스
if prune_method == "Magnitude (크기 기반)":
    st.markdown(f"""
    <div class="info-box" style="border-left-color: #888888;">
        <h3>Magnitude Pruning (크기 기반)</h3>
        <p>가장 전통적인 방식입니다. 행렬 전체에서 <b>절댓값이 작은 순서대로 {sparsity}%의 가중치를 0으로</b> 만듭니다.<br>
        하지만 LLM에서는 가중치가 작아도 곱해지는 입력값(Activation)이 엄청나게 크면 결과에 큰 영향을 미치는데, 이 방식은 그걸 무시해버려 치명적인 성능 저하를 유발할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)
elif prune_method == "Wanda (가중치 × 활성화)":
    st.markdown(f"""
    <div class="info-box" style="border-left-color: #2ca02c;">
        <h3>🪄 Wanda (Weight AND Activation)</h3>
        <p><b>"가중치 크기 × 활성화 크기"</b>를 기준으로 자릅니다. <br>
        가중치 자체가 작더라도, 들어오는 입력 데이터(Outlier)가 거대하다면 그 가중치를 보호합니다. <br>
        현재 시뮬레이션에서는 <b>4번째, 8번째 열(큰 활성화 값)의 가중치들이 집중적으로 살아남는 것</b>을 확인할 수 있습니다!</p>
    </div>
    """, unsafe_allow_html=True)
elif prune_method == "SparseGPT (오차 보상)":
    st.markdown(f"""
    <div class="info-box" style="border-left-color: #ff7f0e;">
        <h3>SparseGPT (오차 분배 및 보상)</h3>
        <p>가지치기로 인해 발생하는 <b>출력 오차(Error)를 계산하여, 살아남은 가중치들의 값을 수정(업데이트)</b>하여 메꿔줍니다.<br>
        단순히 0으로 만드는 것을 넘어, 남은 가중치들이 더 무거운 짐을 나누어 지도록 하여 성능을 극도로 방어합니다. (원본 대비 살아남은 가중치의 색상/값이 미세하게 변한 것을 확인하세요)</p>
    </div>
    """, unsafe_allow_html=True)

# Heatmap 시각화
col1, col2 = st.columns(2)

with col1:
    fig_orig = px.imshow(W_orig, text_auto=".2f", aspect="auto", 
                         color_continuous_scale="RdBu", zmin=-1.5, zmax=1.5)
    fig_orig.update_layout(title="원본 가중치 (Original Weights)", template="plotly_dark", height=450, coloraxis_showscale=False)
    st.plotly_chart(fig_orig, use_container_width=True)

with col2:
    fig_pruned = px.imshow(W_pruned, text_auto=".2f", aspect="auto", 
                           color_continuous_scale="RdBu", zmin=-1.5, zmax=1.5)
    # 0이 된 부분을 검은색에 가깝게 보이도록 처리 효과는 RdBu 스케일에서 0은 흰색이므로 시각적 대비가 됨
    fig_pruned.update_layout(title=f"가지치기 후 ({prune_method}, {sparsity}%)", template="plotly_dark", height=450, coloraxis_showscale=False)
    st.plotly_chart(fig_pruned, use_container_width=True)

# ==========================================
# 6. 심화 분석: 왜 Wanda / SparseGPT 가 필요한가?
# ==========================================
st.markdown("---")
st.markdown("## 기법별 심화 동작 원리 분석")

tab1, tab2 = st.tabs(["특이점(Outlier) 보호 분석", "가중치 분포 변화"])

with tab1:
    st.markdown("#### 입력 데이터 활성화(Activation) 대비 생존율")
    
    # 생존한 가중치 개수를 열(Channel) 단위로 계산
    survival_per_col = np.sum(mask, axis=0) / 10.0 * 100 # 퍼센트
    
    fig_act = go.Figure()
    # 활성화 크기 바 차트 (배경)
    fig_act.add_trace(go.Bar(x=[f"입력 {i+1}" for i in range(10)], y=X_act, 
                             name="입력 활성화(Activation) 크기", marker_color='rgba(255, 255, 255, 0.2)', yaxis='y1'))
    
    # 생존율 꺾은선 그래프
    line_color = '#888888' if prune_method == "Magnitude (크기 기반)" else '#2ca02c' if prune_method == "Wanda (가중치 × 활성화)" else '#ff7f0e'
    fig_act.add_trace(go.Scatter(x=[f"입력 {i+1}" for i in range(10)], y=survival_per_col, 
                                 name="가중치 생존율 (%)", mode='lines+markers', line=dict(color=line_color, width=4), yaxis='y2'))
    
    fig_act.update_layout(
        template="plotly_dark", height=400,
        title="Wanda를 쓰면 활성화 값이 큰 '입력 4', '입력 8'의 가중치 생존율이 치솟습니다!",
        yaxis=dict(title="Activation 크기", side="left", showgrid=False),
        yaxis2=dict(title="생존율 (%)", side="right", overlaying="y", range=[0, 105], showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_act, use_container_width=True)

with tab2:
    st.markdown("#### 가지치기 전/후 가중치 값 변화 (Scatter)")
    
    # 1D로 펴서 산점도로 비교
    w_orig_flat = W_orig.flatten()
    w_pruned_flat = W_pruned.flatten()
    indices = np.arange(len(w_orig_flat))
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=indices, y=w_orig_flat, mode='markers', name='원본 (Original)', marker=dict(color='rgba(31, 119, 180, 0.5)', size=8)))
    fig_scatter.add_trace(go.Scatter(x=indices, y=w_pruned_flat, mode='markers', name='가지치기 후 (Pruned)', marker=dict(color=line_color, size=6, symbol='x')))
    
    scatter_title = "0으로 뚝 떨어지는 점들(잘린 가중치)을 확인하세요."
    if prune_method == "SparseGPT (오차 보상)":
        scatter_title = "SparseGPT: 살아남은 점(X)들이 원본(O)에서 살짝씩 위아래로 이동하여 오차를 보상합니다!"
        
    fig_scatter.update_layout(
        template="plotly_dark", height=400, title=scatter_title,
        xaxis=dict(title="가중치 인덱스 (1D Flatten)"), yaxis=dict(title="가중치 값"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 7. 실제 모델 예상 성능 방어율
# ==========================================
st.markdown("---")
st.markdown("##  예상 정확도 하락 (LLM 50% Pruning 기준)")

if prune_method == "Magnitude (크기 기반)":
    perf_drop = "≈ 10.0 ~ 20.0% 하락"
    perf_color = "#ff4b4b"
    perf_desc = "단순 크기만 보고 자르면 LLM의 중요한 문맥 정보가 끊겨, 언어 생성 능력이 심각하게 망가집니다."
elif prune_method == "Wanda (가중치 × 활성화)":
    perf_drop = "≈ 1.0 ~ 3.0% 하락"
    perf_color = "#2ca02c"
    perf_desc = "Activation Outlier를 보호하여, 연산량을 반으로 줄이면서도 LLM의 성능을 놀라울 정도로 유지합니다."
elif prune_method == "SparseGPT (오차 보상)":
    perf_drop = "≈ 0.5 ~ 2.0% 하락"
    perf_color = "#1f77b4"
    perf_desc = "잘려나간 정보까지 남은 가중치에 밀어 넣어 복원력이 가장 뛰어납니다. (단, 계산 복잡도가 매우 높음)"

st.markdown(f"""
<div style="background-color: #1e1e1e; padding: 25px; border-radius: 12px; border-left: 8px solid {perf_color}; text-align: center;">
    <p style="color: #888888; font-size: 18px; margin-bottom: 5px;">선택한 기법 <b>[{prune_method}]</b> 적용 시</p>
    <h2 style="margin-top: 0; color: #ffffff;">예상 성능 손실: <span style="color: {perf_color};">{perf_drop}</span></h2>
    <p style="font-size: 16px; margin-bottom: 0; color: #dddddd;">{perf_desc}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><div style='background:#2c2f33; padding:15px; border-radius:8px; text-align:center;'><b>made by song</b></div>", unsafe_allow_html=True)