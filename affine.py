import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 페이지 설정 및 CSS
# ==========================================
st.set_page_config(page_title="Linear Quantization Master", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    body { font-family: 'JetBrains Mono', monospace; }
    .info-box { background-color: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
    .box-green { border-left: 6px solid #2ca02c; }
    .box-blue { border-left: 6px solid #1f77b4; }
    .box-pink { border-left: 6px solid #e377c2; }
    .box-cyan { border-left: 6px solid #17becf; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 양자화 핵심 로직
# ==========================================
def symmetric_quantize(x, bits=8):
    q_max = 2**(bits - 1) - 1
    q_min = -q_max
    abs_max = np.max(np.abs(x))
    scale = abs_max / q_max if q_max > 0 else 1
    x_q = np.clip(np.round(x / scale), q_min, q_max)
    x_deq = x_q * scale
    return x_q, x_deq, scale

def affine_quantize(x, bits=8):
    q_min, q_max = 0, 2**bits - 1
    x_min, x_max = np.min(x), np.max(x)
    scale = (x_max - x_min) / (q_max - q_min) if (q_max - q_min) > 0 else 1
    zero_point = np.round(-x_min / scale)
    x_q = np.clip(np.round(x / scale + zero_point), q_min, q_max)
    x_deq = (x_q - zero_point) * scale
    return x_q, x_deq, scale, zero_point

# ==========================================
# 3. 메인 화면 구성
# ==========================================
st.title("선형 양자화(Linear Quantization) 완전 정복")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. 선형 양자화 일반식", 
    "2. 대칭 양자화 (Symmetric)", 
    "3. 아핀 양자화 및 대칭 비교", 
    "4. 아핀 변환의 기하학"
])

# ------------------------------------------
# 탭 1: 선형 양자화의 기본 개념 (수정됨)
# ------------------------------------------
with tab1:
    st.markdown("""
    <div class="info-box box-green">
        <h4 style="color: #2ca02c; margin-top: 0;">선형 양자화 (Linear Quantization)</h4>
        <p style="color: #dddddd;">실수(Float) 범위의 데이터를 주어진 비트(b) 수에 맞춰 정수(Integer) 범위로 일정하게 매핑하는 방식입니다.<br>
        데이터의 최솟값과 최댓값을 활용해 정해진 구간(0 부터 2^b - 1)에 비례적으로 분배합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**1. 실수 r을 b비트 정수 q로 변환 (Quantization)**")
    st.latex(r"q = \text{round}\left( \frac{r - r_{\min}}{r_{\max} - r_{\min}} \times (2^b - 1) \right)")
    
    st.markdown("**2. 역변환 (정수에서 실수 복원, Dequantization)**")
    st.latex(r"\hat{r} = \frac{q}{2^b - 1} \times (r_{\max} - r_{\min}) + r_{\min}")
    
    st.markdown("**3. 양자화 오차 (Quantization Error)**")
    st.latex(r"\varepsilon = r - \hat{r}, \quad \text{최대 오차} \le \frac{r_{\max} - r_{\min}}{2(2^b - 1)}")
    
    st.markdown("""
    **[변수 설명]**
    - r : 원본 실수 데이터
    - q : 양자화된 정수 데이터
    - b : 양자화에 사용할 비트 수 (예: 8비트의 경우 2^8 - 1 = 255)
    - r_min, r_max : 데이터의 최솟값 및 최댓값
    - r_hat : 정수 q를 다시 실수로 복원한 값
    - epsilon : 변환 과정(반올림)에서 발생하는 원본과 복원 값 사이의 오차
    """)

    st.markdown("---")
    st.markdown("**[핵심 포인트] 양자화 중심(Zero-point)과 스케일(Scale)은 어디에 있을까요?**")
    st.markdown("위 공식은 비례식의 형태로 직관적이지만, 실제 하드웨어 및 프레임워크 연산에서는 이 공식을 **Scale(S)**과 **Zero-point(Z)** 개념으로 분리하여 사용합니다. 두 수식은 완전히 동일합니다.")
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.markdown("**Scale (S) : 정수 1칸이 나타내는 실수의 크기**")
        st.latex(r"S = \frac{r_{\max} - r_{\min}}{2^b - 1}")
    with col_z2:
        st.markdown("**Zero-point (Z) : 실수 0이 매핑되는 정수 좌표**")
        st.latex(r"Z = \text{round}\left( \frac{0 - r_{\min}}{S} \right)")

    st.markdown("결과적으로 위 비례식 공식은 우리가 익히 알고 있는 **$q = \text{round}(r / S) + Z$** 형태로 치환됩니다. 데이터의 중심(실수 0)이 어디로 이동했는지 알고 싶다면 Z 공식을 통해 확인할 수 있습니다.")

# ------------------------------------------
# 탭 2: 대칭 양자화 (선형 양자화의 특수한 케이스)
# ------------------------------------------
with tab2:
    st.markdown("""
    <div class="info-box box-blue">
        <h4 style="color: #1f77b4; margin-top: 0;">대칭 양자화 (Symmetric Quantization)</h4>
        <p style="color: #dddddd;">양자화 중심(Zero-point)을 0으로 고정한 형태입니다. 실수의 0과 정수의 0이 항상 일치합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns([1, 1.5])
    with col_s1:
        st.markdown("**[ 대칭 양자화 수식 (Z=0) ]**")
        st.latex(r"x_q = \text{round} \left( \frac{x}{S} \right)")
        st.latex(r"S = \frac{\max(|x_{min}|, |x_{max}|)}{q_{max}}")
        st.markdown("""
        **특징:**
        - 중심을 맞추는 연산이 없어 연산 속도가 가장 빠름.
        - 가중치(Weight)처럼 양수와 음수가 0을 기준으로 대칭적인 데이터에 완벽함.
        """)
        test_bits = st.slider("대칭 양자화 비트 수", 2, 8, 3, key="sym_bit_only")
        
    with col_s2:
        x_sym = np.linspace(-10, 10, 200) 
        _, deq_sym, s_sym = symmetric_quantize(x_sym, test_bits)
        
        fig_sym = go.Figure()
        fig_sym.add_trace(go.Scatter(x=x_sym, y=x_sym, name="Original Data", line=dict(color="white", dash="dash")))
        fig_sym.add_trace(go.Scatter(x=x_sym, y=deq_sym, name="Symmetric Quant", line=dict(color="#1f77b4", shape='hv', width=3)))
        fig_sym.update_layout(title=f"대칭 데이터는 대칭 양자화가 효율적입니다 (Scale: {s_sym:.3f})", template="plotly_dark", height=400)
        st.plotly_chart(fig_sym, use_container_width=True)

# ------------------------------------------
# 탭 3: 아핀 양자화 및 대칭 양자화와의 완벽 비교
# ------------------------------------------
with tab3:
    st.markdown("""
    <div class="info-box box-pink">
        <h4 style="color: #e377c2; margin-top: 0;">아핀 양자화 (Affine/Asymmetric Quantization) 및 성능 비교</h4>
        <p style="color: #dddddd;">데이터가 한쪽으로 쏠려있을 때(예: ReLU 통과 후), 대칭 양자화는 빈 공간을 낭비합니다.<br>
        아핀 양자화는 양자화 구간을 데이터에 딱 맞게 평행 이동시켜 공간을 100% 활용합니다.</p>
    </div>
    """, unsafe_allow_html=True)

    np.random.seed(42)
    x_skewed = np.random.normal(loc=7.0, scale=2.0, size=500)
    x_skewed = np.clip(x_skewed, 2, 12) 
    
    bits = st.slider("비교용 양자화 비트 수 (해상도 차이를 보려면 3~4bit 권장)", 2, 8, 3, key="aff_bit")
    
    _, deq_s, s_s = symmetric_quantize(x_skewed, bits)
    _, deq_a, s_a, zp_a = affine_quantize(x_skewed, bits)
    
    st.markdown("#### 1. 비트 공간(Integer Space) 활용도 비교")
    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("대칭 (음수 공간 낭비)", "아핀 (공간 100% 활용)"))
    
    q_max_sym, q_min_sym = 2**(bits - 1) - 1, -(2**(bits - 1) - 1)
    fig_hist.add_trace(go.Histogram(x=np.clip(np.round(x_skewed/s_s), q_min_sym, q_max_sym), marker_color="#1f77b4", nbinsx=20), row=1, col=1)
    fig_hist.add_vrect(x0=q_min_sym, x1=0, fillcolor="red", opacity=0.1, annotation_text="낭비됨", row=1, col=1)
    
    q_max_aff, q_min_aff = 2**bits - 1, 0
    fig_hist.add_trace(go.Histogram(x=np.clip(np.round(x_skewed/s_a + zp_a), q_min_aff, q_max_aff), marker_color="#e377c2", nbinsx=20), row=1, col=2)
    
    fig_hist.update_layout(template="plotly_dark", height=300, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### 2. 복원된 데이터 해상도(계단 현상) 및 오차 비교")
    
    col_err1, col_err2 = st.columns(2)
    with col_err1: st.error(f"대칭 양자화 오차 (MSE): {np.mean((x_skewed - deq_s)**2):.4f}")
    with col_err2: st.success(f"아핀 양자화 오차 (MSE): {np.mean((x_skewed - deq_a)**2):.4f}")

    sample_idx = np.argsort(x_skewed[:100])
    x_plot = np.arange(100)
    
    fig_step = go.Figure()
    fig_step.add_trace(go.Scatter(x=x_plot, y=x_skewed[:100][sample_idx], mode='lines', name='Original (Float)', line=dict(color='white', dash='dash')))
    fig_step.add_trace(go.Scatter(x=x_plot, y=deq_s[:100][sample_idx], mode='lines+markers', name='Symmetric (계단 큼)', line=dict(color='#1f77b4', shape='hv')))
    fig_step.add_trace(go.Scatter(x=x_plot, y=deq_a[:100][sample_idx], mode='lines+markers', name='Affine (계단 촘촘함)', line=dict(color='#e377c2', shape='hv')))
    
    fig_step.update_layout(template="plotly_dark", height=450, xaxis_title="데이터 인덱스", yaxis_title="복원된 Float 값", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_step, use_container_width=True)

# ------------------------------------------
# 탭 4: 기하학적 아핀 변환
# ------------------------------------------
with tab4:
    st.markdown("""
    <div class="info-box box-cyan">
        <h4 style="color: #17becf; margin-top: 0;">선형 변환(Ax) + 평행 이동(b) = 아핀 변환</h4>
        <p style="color: #dddddd;">양자화에서 <b>Scale 조절이 선형 변환</b>, <b>데이터 범위 이동이 평행 이동</b>에 해당합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_geo_ctrl, col_geo_plot = st.columns([1, 2])
    
    with col_geo_ctrl:
        st.markdown("##### 1. 선형 변환 행렬 (Matrix A)")
        a11 = st.slider("X축 확대/축소 (a11)", -3.0, 3.0, 1.5, 0.1, key="a11")
        a12 = st.slider("X축 기울임 (a12)", -3.0, 3.0, 0.0, 0.1, key="a12")
        a21 = st.slider("Y축 기울임 (a21)", -3.0, 3.0, 0.0, 0.1, key="a21")
        a22 = st.slider("Y축 확대/축소 (a22)", -3.0, 3.0, 1.5, 0.1, key="a22")
        
        st.markdown("##### 2. 평행 이동 벡터 (Vector b)")
        tx = st.slider("X축 이동 (tx)", -5.0, 5.0, 3.0, 0.5, key="tx")
        ty = st.slider("Y축 이동 (ty)", -5.0, 5.0, 2.0, 0.5, key="ty")

        st.markdown("##### 수학적 표현")
        st.latex(r"\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}")
        
    with col_geo_plot:
        points = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [0, 0]]).T
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([[tx], [ty]])
        
        linear_points = A @ points
        affine_points = linear_points + b
        
        fig_geo = go.Figure()
        fig_geo.add_vline(x=0, line_width=2, line_color='white', opacity=0.3)
        fig_geo.add_hline(y=0, line_width=2, line_color='white', opacity=0.3)
        
        fig_geo.add_trace(go.Scatter(x=points[0, :], y=points[1, :], mode='lines+markers', name='Original', fill="toself", line=dict(color='gray', dash='dash')))
        fig_geo.add_trace(go.Scatter(x=linear_points[0, :], y=linear_points[1, :], mode='lines+markers', name='Linear (Ax)', fill="toself", line=dict(color='#1f77b4')))
        fig_geo.add_trace(go.Scatter(x=affine_points[0, :], y=affine_points[1, :], mode='lines+markers', name='Affine (Ax + b)', fill="toself", line=dict(color='#e377c2')))
        
        fig_geo.update_layout(template="plotly_dark", height=450, xaxis=dict(range=[-10, 10]), yaxis=dict(range=[-10, 10], scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("<br><div style='text-align:center; color:#888;'><b>made by song</b></div>", unsafe_allow_html=True)