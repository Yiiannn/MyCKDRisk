# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 15:19:10 2026

@author: lenovo
"""

# 导入包
import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go

# 设置页面配置
st.set_page_config(
    page_title='慢性肾脏疾病风险评估工具',
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 更简洁的版本
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem !important;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }

    /* 副标题样式 */
    .sub-header {
        font-size: 1.2rem !important;
        color: #7f8c8d;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* 步骤标题样式 */
    .step-header {
        font-size: 1.8rem !important;
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
        padding: 0.5rem;
        background: linear-gradient(90deg, #e3f2fd, #fff);
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }

    /* 信息框样式 */
    .info-box {
        background-color: #e8f4fc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }

    /* 卡片样式 */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }

    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1) !important;
    }

    /* 主要按钮 */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(90deg, #3498db, #2980b9) !important;
        color: white !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(90deg, #2980b9, #21618c) !important;
    }

    /* 次要按钮 */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: #f8f9fa !important;
        color: #2c3e50 !important;
        border: 1px solid #dee2e6 !important;
    }

    /* 进度指示器样式 */
    .progress-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }

    .progress-step {
        text-align: center;
        z-index: 2;
        flex: 1;
    }

    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #ecf0f1;
        color: #7f8c8d;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: 0 auto 0.5rem;
        border: 3px solid #ecf0f1;
    }

    .step-circle.active {
        background: #3498db;
        color: white;
        border-color: #3498db;
    }

    .step-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }

    .step-label.active {
        color: #2c3e50;
        font-weight: 600;
    }

    /* 分割线 */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #3498db, transparent);
        margin: 2rem 0;
    }

    /* 结果卡片样式 */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        animation: fadeIn 1s ease-in;
        margin: 2rem 0;
    }

    .result-high {
        background: linear-gradient(135deg, #ffeaa7, #fab1a0);
        border-left: 8px solid #e74c3c;
    }

    .result-low {
        background: linear-gradient(135deg, #81ecec, #55efc4);
        border-left: 8px solid #00b894;
    }

    /* 特征标签样式 */
    .feature-label {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }

    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* 滑块样式 */
    .stSlider > div > div > div {
        background: #3498db !important;
    }

    /* 单选按钮样式 */
    .stRadio > div {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "form_data" not in st.session_state:
    st.session_state.form_data = {}

# 模型配置（保持不变）
MODEL_CONFIG = {
    ("中国 China", "高血压 Hypertension"): {
        "model_path": "CHyp_LR_model.sav",
        "features": ["rgender", "be001", "age_cul", "qm002", "mean_pulse", "max_respiration", "mean_handgrip_left",
                     "da033", "da081", "bmi", "dyslipidemia", "diabetes"],
    },
    ("中国 China", "糖尿病 Diabetes"): {
        "model_path": "CDia_LR_model.sav",
        "features": ["rgender", "be001", "age_cul", "mean_pulse", "max_respiration", "mean_handgrip_left",
                     "mean_handgrip_right", "depression", "da033", "da041", "da081", "bmi", "hypertension",
                     "dyslipidemia"],
    },
    ("中国 China", "血脂异常 Dyslipidemia"): {
        "model_path": "CDys_LR_model.sav",
        "features": ["rgender", "be001", "age_cul", "qm002", "mean_pulse", "max_respiration", "mean_handgrip_left",
                     "mean_handgrip_right", "da069", "depression", "da033", "da041", "da081", "bmi", "hypertension",
                     "diabetes"],
    },
    ("美国 USA", "高血压 Hypertension"): {
        "model_path": "UHyp_XGBoost_best_model.sav",
        "features": ["be001", "depression", "da041", "da069", "age_cul", "qm002", "mean_pulse", "max_respiration",
                     "mean_handgrip_left", "mean_handgrip_right", "da081", "bmi", "dyslipidemia", "diabetes"],
    },
    ("美国 USA", "糖尿病 Diabetes"): {
        "model_path": "UDia_XGBoost_best_model.sav",
        "features": ["be001", "depression", "da033", "da041", "da069", "age_cul", "qm002", "max_respiration",
                     "mean_handgrip_right", "da081", "dyslipidemia", "hypertension"],
    },
    ("美国 USA", "血脂异常 Dyslipidemia"): {
        "model_path": "UDys_LR_model.sav",
        "features": ["be001", "depression", "da033", "da069", "age_cul", "qm002", "mean_pulse", "max_respiration",
                     "mean_handgrip_right", "da081", "bmi", "diabetes", "hypertension"],
    }
}

SCALER_STATS = {
    ("中国 China", "高血压 Hypertension"): {
        "age_cul": (45, 102),
        "qm002": (10.2, 142.9),
        "mean_pulse": (38, 151),
        "max_respiration": (30, 890),
        "mean_handgrip_left": (1.5, 68),
        "bmi": (20, 72),
    },
    ("中国 China", "糖尿病 Diabetes"): {
        "age_cul": (45, 97),
        "mean_pulse": (41, 151),
        "max_respiration": (30, 800),
        "mean_handgrip_left": (2, 57.2),
        "mean_handgrip_right": (0.2, 65),
        "bmi": (10, 61),
    },
    ("中国 China", "血脂异常 Dyslipidemia"): {
        "age_cul": (45, 102),
        "qm002": (22, 150),
        "mean_pulse": (43, 151),
        "max_respiration": (30, 890),
        "mean_handgrip_left": (2, 64.2),
        "mean_handgrip_right": (0.2, 70.8),
        "bmi": (19.6, 83.6),
    },
    ("美国 USA", "高血压 Hypertension"): {
        "age_cul": (45, 101),
        "qm002": (58.42, 89.4),
        "mean_pulse": (40.5, 117),
        "max_respiration": (30, 793),
        "mean_handgrip_left": (5, 65.75),
        "mean_handgrip_right": (5, 65.75),
        "bmi": (17.85, 50),
    },
    ("美国 USA", "糖尿病 Diabetes"): {
        "age_cul": (45, 95),
        "qm002": (58.42, 149.86),
        "max_respiration": (30, 793),
        "mean_handgrip_right": (5.25, 63.75),
    },
    ("美国 USA", "血脂异常 Dyslipidemia"): {
        "age_cul": (45, 95),
        "qm002": (64.77, 150),
        "mean_pulse": (42, 107),
        "max_respiration": (30, 999),
        "mean_handgrip_right": (5.25, 63),
        "bmi": (17.85, 50),
    },
}


def coding_fun(df):
    df = df.copy()

    if 'rgender' in df.columns:
        df['rgender'] = df['rgender'].replace(
            ['男性(Male)', '女性(Female)'], [1, 2]
        )

    if 'be001' in df.columns:
        df['be001'] = df['be001'].replace(
            ['已婚(Married)',
             '分居/离异/丧偶(Separated/Divorced/Widowed)',
             '未婚(Never married)'],
            [1, 2, 3]
        )

    if 'depression' in df.columns:
        df['depression'] = df['depression'].replace(
            ['是(Yes)', '否(No)'], [1, 0]
        )

    if 'da033' in df.columns:
        df['da033'] = df['da033'].replace(
            ['极好(Excellent)', '很好(Very good)', '好(Good)', '一般(Fair)', '不好(Poor)'],
            [5, 4, 3, 2, 1]
        )

    if 'da081' in df.columns:
        df['da081'] = df['da081'].replace(
            ['几乎不可能(Almost impossible)',
             '不太可能(Not very likely)',
             '可能(Maybe)',
             '非常可能(Very likely)',
             '几乎肯定(Almost certain)'],
            [1, 2, 3, 4, 5]
        )

    yes_no_cols = ['da069', 'da041', 'diabetes', 'hypertension', 'dyslipidemia']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace(['是(Yes)', '否(No)'], [1, 0])

    return df


def create_progress_bar():
    """创建进度条"""
    steps = ["选择国家", "选择疾病", "填写信息"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="progress-step">
            <div class="step-circle {'active' if st.session_state.step >= 1 else ''}">1</div>
            <div class="step-label {'active' if st.session_state.step >= 1 else ''}">{steps[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="progress-step">
            <div class="step-circle {'active' if st.session_state.step >= 2 else ''}">2</div>
            <div class="step-label {'active' if st.session_state.step >= 2 else ''}">{steps[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="progress-step">
            <div class="step-circle {'active' if st.session_state.step >= 3 else ''}">3</div>
            <div class="step-label {'active' if st.session_state.step >= 3 else ''}">{steps[2]}</div>
        </div>
        """, unsafe_allow_html=True)


# 侧边栏
#with st.sidebar:
#    st.markdown("## 🏥 CKD风险评估工具")
#    st.markdown("---")

#    st.markdown("### ℹ️ 使用说明")
    #with st.expander("点击查看详细信息"):
    
#    st.markdown("""
#        📊 本工具可以帮助您：\nOur tool can help you:\n\n
#        - 评估慢性肾脏疾病风险\nAssess the risk of chronic kidney disease\n\n
#        - 获得个性化建议\nReceive personalized suggestions\n\n

#        📞 注意事项：\nNotes:\n\n
#        - 结果仅供参考\nThe result is for reference only\n\n
#        - 不能替代专业医疗建议\nCannot replace professional medical advice\n\n
#        - 如有不适请及时就医\nIf you feel unwell, please seek medical attention immediately\n\n
#        """)

    #st.markdown("### 📊 关于模型")
    #st.markdown("基于机器学习算法开发，经过临床数据验证")

    #st.markdown("### 📞 紧急联系")
    #st.markdown("如有紧急情况，请立即联系：")
    #st.markdown("- 🚑 急救电话: 120")
    #st.markdown("- 📱 健康热线: 12320")

#    st.markdown("---")
#    st.markdown("*版本 1.0 | 2026*")

# 侧边栏
# 侧边栏
# 侧边栏
with st.sidebar:
    # 顶部标题 - 学术期刊专业配色方案
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; 
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                color: white; border-radius: 0 0 15px 15px; margin-bottom: 2rem; 
                box-shadow: 0 4px 12px rgba(44, 62, 80, 0.15);
                border-bottom: 3px solid #3498db;">
        <div style="background: rgba(52, 152, 219, 0.2); width: 60px; height: 60px; 
                    border-radius: 50%; display: flex; align-items: center; 
                    justify-content: center; margin: 0 auto 1rem; 
                    border: 2px solid rgba(52, 152, 219, 0.4);">
            <span style="font-size: 1.8rem; color: white;">🏥</span>
        </div>
        <h2 style="color: white; margin-bottom: 0.3rem; font-size: 1.3rem; font-weight: 700; 
                   letter-spacing: 0.5px;">CKD风险评估工具</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.85rem; 
                  font-weight: 400;">Chronic Kidney Disease Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # 使用说明部分
    st.markdown("### ℹ️ 使用说明")

    # 学术风格卡片容器
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;">
    """, unsafe_allow_html=True)

    # 工具功能部分 - 使用更学术的排版
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <p style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem; font-size: 0.95rem;">
            📊 本工具可以帮助您 / Our tool can help you:
        </p>
        <ul style="margin: 0; padding-left: 1.2rem; color: #34495e; font-size: 0.9rem;">
            <li style="margin-bottom: 0.3rem;">评估慢性肾脏疾病风险<br>
                <span style="color: #7f8c8d; font-size: 0.85rem;">Assess the risk of chronic kidney disease</span>
            </li>
            <li>获得个性化建议<br>
                <span style="color: #7f8c8d; font-size: 0.85rem;">Receive personalized suggestions</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 注意事项部分 - 学术风格的警告框
    st.markdown("""
    <div style="background: #f9f9f9; padding: 1rem; border-radius: 6px; border-left: 4px solid #e74c3c;">
        <p style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem; font-size: 0.95rem;">
            ⚠️ 注意事项 / Important Notes:
        </p>
        <ul style="margin: 0; padding-left: 1rem; color: #34495e; font-size: 0.9rem;">
            <li style="margin-bottom: 0.3rem;">结果仅供参考<br>
                <span style="color: #e74c3c; font-size: 0.85rem;">The result is for reference only</span>
            </li>
            <li style="margin-bottom: 0.3rem;">不能替代专业医疗建议<br>
                <span style="color: #e74c3c; font-size: 0.85rem;">Cannot replace professional medical advice</span>
            </li>
            <li>如有不适请及时就医<br>
                <span style="color: #e74c3c; font-size: 0.85rem;">If you feel unwell, please seek medical attention immediately</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 版本信息 - 简洁学术风格
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem; 
                background: #f8f9fa; border-radius: 8px; margin-top: 1rem;
                color: #2c3e50; border: 1px solid #e9ecef;">
        <div style="display: inline-block; background: #3498db; 
                    color: white; padding: 0.4rem 1.2rem; border-radius: 20px; 
                    font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem;">
            🔬 v1.0 | 2026
        </div>
        <p style="margin: 0; color: #2c3e50; font-size: 0.9rem; 
                  font-weight: 500;">© 2026 CKD风险评估系统</p>
        <p style="margin: 0.3rem 0 0 0; color: #7f8c8d; font-size: 0.8rem;">
            仅供科研参考使用 | For Research Reference Only
        </p>
        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e9ecef;">
            <p style="color: #95a5a6; font-size: 0.75rem; margin: 0.2rem 0;">
                📧 yi_xiaohann@163.com
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">🏥 中老年三高患者慢性肾脏疾病风险评估工具</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Web-based tool for assessing the risk of chronic kidney disease in middle-aged and elderly patients with hypertension, diabetes, and hyperlipidemia</p>',
    unsafe_allow_html=True)

# 进度条
create_progress_bar()

# 主内容区域
if st.session_state.step == 1:
    st.markdown('<h2 class="step-header">📋 第一步：选择国家 First step:Select country</h2>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">🌍 请选择您所在的国家，系统将为您匹配相应的评估模型\nPlease select your country, and the system will match the corresponding assessment model for you.</div>', unsafe_allow_html=True)

    # 使用列布局
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 创建卡片效果
        with st.container():
            st.markdown(
                '<div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">',
                unsafe_allow_html=True)
            country = st.radio(
                "### 请选择您的国家 Please select your country",
                ["中国 China", "美国 USA"],
                index=None
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("下一步 ➔\nNext", type="primary", use_container_width=True):
                if country is None:
                    st.warning("⚠️ 请先选择您的国家 Please select your country first")
                else:
                    st.session_state.country = country
                    st.session_state.step = 2
                    st.rerun()

elif st.session_state.step == 2:
    st.markdown('<h2 class="step-header">💊 第二步：选择疾病 Step 2: Select the disease</h2>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box">👤 您选择的国家The country you have selected: <strong>{st.session_state.country}</strong></div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown(
                '<div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">',
                unsafe_allow_html=True)
            group = st.radio(
                "### 请选择您的主要疾病Please select your primary disease",
                ["高血压 Hypertension", "糖尿病 Diabetes", "血脂异常 Dyslipidemia"],
                index=None,
                help="选择您患有的主要疾病类型 Select the main type of disease you are suffering from"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("← 上一步\nBack", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
            with col2:
                if st.button("下一步 ➔\nNext", type="primary", use_container_width=True):
                    if group is None:
                        st.warning("⚠️ 请先选择疾病 Please select the disease first")
                    else:
                        st.session_state.group = group
                        st.session_state.step = 3
                        st.rerun()

elif st.session_state.step == 3:
    country = st.session_state.country
    group = st.session_state.group

    st.markdown('<h2 class="step-header">📝 第三步：填写健康信息 Step 3: Fill in health information</h2>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        🎯 当前评估人群 Current assessment population:<strong>{country.split()[0]}</strong> - <strong>{group.split()[0]}</strong><br>
        📋 请根据实际情况填写以下信息，所有字段均为必填项\n\nPlease fill in the following information according to the actual situation. All fields are mandatory.
    </div>
    """, unsafe_allow_html=True)

    config = MODEL_CONFIG[(country, group)]
    need_features = config["features"]

    # 创建表单
    input_dict = {}

    # 分两列布局
    col1, col2 = st.columns(2)

    with col1:
        #st.markdown("### 基本信息 Basic information")
        st.markdown('<h3 class="section-header">👤 基本信息 Basic information</h3>', unsafe_allow_html=True)
        if "rgender" in need_features:
            st.markdown('<div class="feature-label">请选择您的性别</div>', unsafe_allow_html=True)
            input_dict["rgender"] = st.radio(
                "Please select your gender",
                ['男性(Male)', '女性(Female)'],
                index=None,
                horizontal=True,
                key="gender_radio"
            )

        if "be001" in need_features:
            st.markdown('<div class="feature-label">请选择您的婚姻状态</div>', unsafe_allow_html=True)
            input_dict["be001"] = st.radio(
                "Please select your marital status",
                ['已婚(Married)', '分居/离异/丧偶(Separated/Divorced/Widowed)', '未婚(Never married)'],
                index=None,
                key="marital_radio"
            )

        if "age_cul" in need_features:
            st.markdown('<div class="feature-label">请选择您的年龄</div>', unsafe_allow_html=True)
            age = st.slider("Please select your age", 45, 120, 60, key="age_slider")
            input_dict["age_cul"] = age
            st.caption(f"当前年龄 Current age: {age} ")


        if "bmi" in need_features:
            st.markdown('<div class="feature-label">请选择您的BMI</div>', unsafe_allow_html=True)
            bmi = st.slider("Please select your BMI", 12.0, 60.0, 24.0, 0.1, key="bmi_slider")
            input_dict["bmi"] = bmi
            st.caption(f"当前BMI Current BMI: {bmi}")

        if "qm002" in need_features:
            st.markdown('<div class="feature-label">请选择您的腰围 (cm)</div>', unsafe_allow_html=True)
            waist = st.slider("Please select your waist(cm)", 40, 150, 85, key="waist_slider")
            input_dict["qm002"] = waist
            st.caption(f"当前腰围 Current waist: {waist} cm")

    with col2:
        #st.markdown("### 生理指标 Physical signs")
        st.markdown('<h3 class="section-header">📊 生理指标 Physical signs</h3>', unsafe_allow_html=True)
        if "mean_pulse" in need_features:
            st.markdown('<div class="feature-label">请选择您的平均脉搏 (次/分钟)</div>', unsafe_allow_html=True)
            pulse = st.slider("Please select your average pulse rate (beats per minute)", 40, 220, 72, key="pulse_slider")
            input_dict["mean_pulse"] = pulse
            st.caption(f"当前脉搏 Current pulse: {pulse} ")

        if "max_respiration" in need_features:
            st.markdown('<div class="feature-label">请选择您的最大呼气流速</div>', unsafe_allow_html=True)
            resp = st.slider("Please select your peak expiratory velocity maximum", 50, 800, 400, key="resp_slider")
            input_dict["max_respiration"] = resp
            st.caption(f"当前最大呼气流速 Current peak expiratory velocity maximum: {resp}")

        if "mean_handgrip_left" in need_features:
            st.markdown('<div class="feature-label">请选择您的左手平均握力 (kg)</div>', unsafe_allow_html=True)
            grip_left = st.slider("Please select your mean left hand grip strength", 0, 80, 30, key="grip_left_slider")
            input_dict["mean_handgrip_left"] = grip_left
            st.caption(f"当前左手握力 Current mean left hand grip strength: {grip_left} kg")

        if "mean_handgrip_right" in need_features:
            st.markdown('<div class="feature-label">请选择您的右手平均握力 (kg)</div>', unsafe_allow_html=True)
            grip_right = st.slider("Please select your mean right hand grip strength", 0, 80, 32, key="grip_right_slider")
            input_dict["mean_handgrip_right"] = grip_right
            st.caption(f"当前右手握力 Current mean right hand grip strength: {grip_right} kg")

    # 健康问题部分
    # 健康问题部分
    st.markdown('<h3 class="section-header">🩺 健康状况 Health status</h3>', unsafe_allow_html=True)
    cols = st.columns(2)
    col_idx = 0

    health_questions = [
        ("depression", "您患有抑郁相关疾病吗？", "Do you have depression?", ['是(Yes)', '否(No)']),
        ("da033", "您的远距离视力怎么样？", "How is your eyesight for seeing things at a distance?",
         ['极好(Excellent)', '很好(Very good)', '好(Good)', '一般(Fair)', '不好(Poor)']),
        ("da081", "您认为自己活到预期年龄的可能性怎么样?", "How do you think your chances of living to the expected age are?",
         ['几乎不可能(Almost impossible)', '不太可能(Not very likely)', '可能(Maybe)', '非常可能(Very likely)',
          '几乎肯定(Almost certain)']),
        ("da069", "您现在喝酒吗？", "Do you drink alcohol now?", ['是(Yes)', '否(No)']),
        ("da041", "您现在身体有疼痛吗？", "Do you have body pain now?", ['是(Yes)', '否(No)']),
        ("diabetes", "您患有糖尿病吗？", "Do you have diabetes?", ['是(Yes)', '否(No)']),
        ("hypertension", "您患有高血压吗？", "Do you have hypertension?", ['是(Yes)', '否(No)']),
        ("dyslipidemia", "您患有血脂异常吗？", "Do you have dyslipidemia?", ['是(Yes)', '否(No)']),
    ]

    for feature, chinese_question, english_question, options in health_questions:
        if feature in need_features:
            with cols[col_idx % 2]:
                # 中文问题
                st.markdown(f'<div class="feature-label">{chinese_question.split("？")[0]}</div>',
                            unsafe_allow_html=True)
                # 英文问题
                st.markdown(
                    f'<div style="color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">{english_question}</div>',
                    unsafe_allow_html=True)
                # 选项
                input_dict[feature] = st.radio(
                    "",  # 空字符串，因为我们已经在上面显示了问题
                    options,
                    index=None,
                    key=f"{feature}_radio",
                    label_visibility="collapsed"  # 隐藏radio的默认标签
                )
            col_idx += 1

    # 按钮区域
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← 重新选择\nReselect", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with col2:
        predict_button = st.button("🔍 开始评估 Start the assessment", type="primary", use_container_width=True)

    # 处理预测
    if predict_button:
        input_df = pd.DataFrame([input_dict])

        # 检查完整性
        missing_fields = [field for field, value in input_dict.items() if value is None]
        if missing_fields:
            st.error(f"⚠️ 请完成以下必填项 Please complete the following mandatory fields: {', '.join(missing_fields)}")
        else:
            with st.spinner("正在评估中，请稍候...\nEvaluation in progress. Please wait..."):
                # 编码处理
                input_encoded = coding_fun(input_df)
                X = input_encoded[need_features].copy()
                X = X.apply(pd.to_numeric)

                # 标准化
                if (country, group) in SCALER_STATS:
                    scaler_dict = SCALER_STATS[(country, group)]
                    for col, (vmin, vmax) in scaler_dict.items():
                        if col in X.columns and vmax > vmin:
                            X[col] = (X[col] - vmin) / (vmax - vmin)

                # 加载模型并预测
                try:
                    with open(config["model_path"], "rb") as f:
                        model = pickle.load(f)

                    prob = model.predict_proba(X)[0][1]
                    result = int(prob >= 0.5)

                    # 显示结果
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                    if result == 1:
                        st.markdown("""
                        <div class="result-card result-high">
                            <h2>⚠️ 高风险预警 High-risk warning</h2>
                            <h3>您可能属于CKD高危人群</h3>
                            <p><em>You might fall into the high-risk category of CKD.</em></p>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">
                            <strong>建议 Suggestion：</strong>
                            </p>
                            <ul style="text-align: left; margin: 1rem 2rem;">
                                <li>立即进行肾功能相关检查 Have immediate tests related to kidney function conducted</li>
                                <li>尽快咨询专业肾病医生 Consult a professional nephrologist as soon as possible</li>
                                <li>定期监测血压、血糖等指标 Regularly monitor indicators such as blood pressure, blood sugar, and blood lipids</li>
                                <li>调整饮食结构，控制蛋白质摄入 Adjust the diet structure and control protein intake</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        # 创建风险指示器
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "CKD风险评分\nCKD Risk score", 'font': {'size': 24}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "red"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            },
                            number={'font': {'size': 40}}
                        ))
                        fig.update_layout(height=300, margin=dict(t=50, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.markdown("""
                        <div class="result-card result-low">
                            <h2>✅ 低风险评估 Low-risk assessment</h2>
                            <h3>您可能属于CKD低危人群</h3>
                            <p><em>You might fall into the low-risk category of CKD.</em></p>
                            <p style="font-size: 1.2rem; margin: 1rem 0;">
                            <strong>建议 Suggestion：</strong>
                            </p>
                            <ul style="text-align: left; margin: 1rem 2rem;">
                                <li>保持健康生活方式 Maintain a healthy lifestyle</li>
                                <li>定期进行健康体检 Regular health check-ups</li>
                                <li>控制血压、血糖、血脂 Control blood pressure, blood sugar and blood lipids</li>
                                <li>保持适当运动 Maintain appropriate exercise</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        # 创建风险指示器
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "CKD风险评分\nCKD Risk score", 'font': {'size': 24}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "green"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            },
                            number={'font': {'size': 40}}
                        ))
                        fig.update_layout(height=300, margin=dict(t=50, b=0))
                        st.plotly_chart(fig, use_container_width=True)

                    # 免责声明
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 2rem; border: 1px solid #dee2e6;">
                        <p style="margin: 0; font-size: 0.9rem;">
                        <strong>免责声明：</strong>本评估结果基于统计模型计算，仅供参考，不能替代专业医疗诊断。如有任何健康问题，请咨询专业医生。<br>
                        <strong>Disclaimer:</strong> This assessment is based on statistical models and is for reference only. It cannot replace professional medical diagnosis. Please consult a healthcare professional for any health concerns.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 重新评估按钮
                    if st.button("🔄 重新评估 Transvaluation", type="secondary", use_container_width=True):
                        st.session_state.step = 1
                        st.session_state.form_data = {}
                        st.rerun()

                except Exception as e:
                    st.error(f"评估过程中出现错误: {str(e)}")
                    st.info("请检查模型是否正确")

# 页脚
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #7f8c8d; border-top: 1px solid #e9ecef; background-color: #f8f9fa;">
    <p style="margin: 0.5rem 0;">
        <strong>© 2026 慢性肾脏疾病风险评估系统</strong><br>
        <small>仅供科研参考使用 | For Research Reference Only</small>
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        联系邮箱 E-mail: yi_xiaohann@163.com
    </p>
</div>
""", unsafe_allow_html=True)