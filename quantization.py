import streamlit as st
import struct
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 설정 및 커스텀 CSS (간지 & 애니메이션)
# ==========================================
st.set_page_config(page_title="AI Quantization Simulator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    .bit-box {
        display: inline-block;
        width: 25px;
        height: 35px;
        margin: 3px;
        line-height: 35px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in-out forwards;
        opacity: 0;
        transform: translateY(-10px);
    }
    
    @keyframes fadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .sign { background: linear-gradient(135deg, #ff4b4b, #c21807); }
    .exponent { background: linear-gradient(135deg, #1f77b4, #0b4a7d); }
    .mantissa { background: linear-gradient(135deg, #2ca02c, #145e14); }
    .integer { background: linear-gradient(135deg, #9467bd, #54337a); }
    
    .container {
        display: flex; flex-wrap: wrap; justify-content: center; margin: 20px 0;
        background-color: #1e1e1e; padding: 20px; border-radius: 12px;
        border: 1px solid #333;
    }
    .legend {
        display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace; font-size: 14px;
    }
    .legend span { display: flex; align-items: center; gap: 5px; }
    .color-box { width: 15px; height: 15px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 비트 변환 로직
# ==========================================
def get_fp32_bits(f):
    s = struct.pack('>f', f)
    i = struct.unpack('>I', s)[0]
    return f"{i:032b}"

def get_fp16_bits(f):
    f16 = np.float16(f)
    i = f16.view('H')
    return f"{i:016b}"

# ==========================================
# 3. 사이드바 UI (입력 컨트롤)
# ==========================================
st.sidebar.title("양자화 컨트롤 패널")
st.sidebar.markdown("인터랙티브 시뮬레이터")

input_val = st.sidebar.slider("FP32 입력 값", min_value=-10.0, max_value=10.0, value=3.1415, step=0.0001, format="%.4f")
precision = st.sidebar.select_slider("데이터 타입 (정밀도) 선택", options=["FP32", "FP16", "INT8", "INT4"])

st.sidebar.markdown("---")
st.sidebar.markdown("**양자화 스케일 설정 (INT용)**")
max_range = st.sidebar.number_input("최대 표현 범위 (Max Range)", value=10.0, disabled=True)

# ==========================================
# 4. 양자화 계산 로직
# ==========================================
original_val = input_val
dequantized_val = 0.0
bits = ""

if precision == "FP32":
    bits = get_fp32_bits(original_val)
    dequantized_val = original_val
    
elif precision == "FP16":
    bits = get_fp16_bits(original_val)
    dequantized_val = float(np.float16(original_val))
    
elif precision == "INT8":
    scale = 127.0 / max_range
    q_val = np.clip(np.round(original_val * scale), -128, 127)
    dequantized_val = q_val / scale
    bits = f"{int(q_val) & 0xFF:08b}"
    
elif precision == "INT4":
    scale = 7.0 / max_range
    q_val = np.clip(np.round(original_val * scale), -8, 7)
    dequantized_val = q_val / scale
    bits = f"{int(q_val) & 0x0F:04b}"

# 오차 및 퍼센트 계산
error = abs(original_val - dequantized_val)
if original_val != 0:
    error_percentage = (error / abs(original_val)) * 100
else:
    error_percentage = 0.0

# ==========================================
# 5. 메인 화면 출력 (시각화)
# ==========================================
st.title("🚀 신경망 양자화(Quantization) 시뮬레이터")
st.markdown(f"### 현재 선택된 정밀도: **{precision}**")

st.markdown("""
<div class="legend">
    <span><div class="color-box" style="background:#ff4b4b;"></div> Sign (부호)</span>
    <span><div class="color-box" style="background:#1f77b4;"></div> Exponent (지수부)</span>
    <span><div class="color-box" style="background:#2ca02c;"></div> Mantissa (가수부)</span>
    <span><div class="color-box" style="background:#9467bd;"></div> Integer (정수부 - 양자화된 값)</span>
</div>
""", unsafe_allow_html=True)

html_bits = '<div class="container">'
for idx, b in enumerate(bits):
    delay = f"animation-delay: {idx * 0.03}s;"
    
    if precision == "FP32":
        if idx == 0: cls = "sign"
        elif idx < 9: cls = "exponent"
        else: cls = "mantissa"
    elif precision == "FP16":
        if idx == 0: cls = "sign"
        elif idx < 6: cls = "exponent"
        else: cls = "mantissa"
    else:
        if idx == 0: cls = "sign"
        else: cls = "integer"
        
    html_bits += f'<div class="bit-box {cls}" style="{delay}">{b}</div>'
html_bits += '</div>'

st.markdown(html_bits, unsafe_allow_html=True)

# ==========================================
# 6. 오차 분석 대시보드 (퍼센트 추가)
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric(label="원본 값 (FP32)", value=f"{original_val:.6f}")
col2.metric(label=f"역양자화 값 ({precision})", value=f"{dequantized_val:.6f}")

# 오차 퍼센트를 빨간색 화살표와 함께 눈에 띄게 표시
col3.metric(
    label="양자화 오차 (Error)", 
    value=f"{error:.6f}", 
    delta=f"{error_percentage:.2f}% 오차율", 
    delta_color="inverse"
)

# Plotly 차트
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Original (FP32)', f'Dequantized ({precision})'],
    y=[original_val, dequantized_val],
    marker_color=['#1F77B4', '#FF4B4B'],
    text=[f"{original_val:.4f}", f"{dequantized_val:.4f}"],
    textposition='auto'
))

fig.update_layout(
    title=f"원본 값 vs 양자화 복원 값 비교 (오차율: {error_percentage:.2f}%)",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Value"),
    height=400
)

st.plotly_chart(fig, use_container_width=True)
st.info("MADE BY SONG_JG")