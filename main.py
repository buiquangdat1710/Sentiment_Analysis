import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Ứng dụng Phân tích Cảm xúc",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #1E88E5, #8E24AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #1e2130;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.2rem;
        border-top: 4px solid #1E88E5;
        color: #e0e0e0;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 0.8rem;
    }
    .emotion-label {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #1E88E5, #8E24AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .insight-text {
        font-size: 1rem;
        color: #424242;
        line-height: 1.5;
    }
    .stButton>button {
        background: linear-gradient(to right, #1E88E5, #8E24AA);
        border: none;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.6rem 2.5rem;
        border-radius: 50px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(30, 136, 229, 0.3);
    }
    /* Custom styling for sidebar */
    .css-1d391kg, .css-163ttbj {
        background-image: linear-gradient(to bottom, #f5f7fa, #e4e8f0);
    }
    /* Styling for text area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        padding: 10px;
        font-size: 1.1rem;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTextArea textarea:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30,136,229,0.2);
    }
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background-color: #1e2130;
        border-radius: 10px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
        color: #e0e0e0;
    }
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 1rem;
        border-bottom: 2px solid #424242;
        padding-bottom: 0.5rem;
    }
    /* Insight bullets */
    .insight-bullet {
        background-color: #263238;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        border-left: 4px solid #1E88E5;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Function to get emotion insights
vietnamese_labels = {
                "joy": "Vui vẻ",
                "sadness": "Buồn bã",
                "anger": "Tức giận",
                "fear": "Sợ hãi",
                "love": "Yêu thương",
                "surprise": "Ngạc nhiên"
            }
def get_emotion_insights(emotion, score):

    insights = {
        "joy": [
            "Thể hiện sự hạnh phúc, hài lòng hoặc phấn khích tích cực.",
            "Điểm cao cho thấy nội dung tích cực hoặc mang tính kỷ niệm.",
            "Cần xem xét ngữ cảnh - niềm vui có thể thực sự hoặc mang tính mỉa mai."
        ],
        "sadness": [
            "Phản ánh sự thất vọng, đau buồn hoặc mất mát.",
            "Điểm cao cho thấy có thể đang gặp khó khăn cá nhân hoặc trải nghiệm tiêu cực.",
            "Văn bản thường chứa ngôn ngữ u ám hoặc đề cập đến khó khăn."
        ],
        "anger": [
            "Thể hiện sự tức giận, khó chịu hoặc bực bội.",
            "Điểm cao thường đi kèm ngôn ngữ tiêu cực mạnh hoặc chỉ trích.",
            "Có thể cho thấy sự bất mãn với dịch vụ, trải nghiệm hoặc con người."
        ],
        "fear": [
            "Thể hiện sự lo lắng, sợ hãi về điều gì đó.",
            "Điểm cao cho thấy lo ngại về các sự kiện tương lai hoặc không chắc chắn.",
            "Thường chứa ngôn ngữ về rủi ro, nguy hiểm hoặc hậu quả tiêu cực."
        ],
        "love": [
            "Thể hiện tình cảm, sự trìu mến hoặc cảm xúc lãng mạn.",
            "Điểm cao thường liên quan đến biểu hiện tình cảm tích cực với người khác.",
            "Có thể chứa các từ ngữ trìu mến hoặc lòng biết ơn."
        ],
        "surprise": [
            "Thể hiện sự ngạc nhiên, bất ngờ.",
            "Điểm cao cho thấy gặp phải điều gì đó không mong đợi.",
            "Có thể là tích cực (ngạc nhiên thú vị) hoặc tiêu cực (sốc)."
        ]
    }
    
    # Thêm đánh giá cường độ
    if score > 0.8:
        intensity = f"Văn bản thể hiện sự {vietnamese_labels[emotion]} rất mạnh mẽ."
    elif score > 0.6:
        intensity = f"Văn bản thể hiện rõ ràng sự {vietnamese_labels[emotion]}."
    elif score > 0.4:
        intensity = f"Văn bản cho thấy dấu hiệu vừa phải của sự {vietnamese_labels[emotion]}."
    elif score > 0.2:
        intensity = f"Văn bản có gợi ý tinh tế về sự {vietnamese_labels[emotion]}."
    else:
        intensity = f"Văn bản chỉ có dấu vết rất nhỏ của sự {vietnamese_labels[emotion]}."
        
    return [intensity] + insights.get(emotion, ["Không có thông tin chi tiết cho cảm xúc này."])

# Function to get color map for emotions
def get_emotion_colors():
    return {
        "joy": "#FFD700",      # Gold
        "sadness": "#4682B4",  # Steel Blue
        "anger": "#DC143C",    # Crimson
        "fear": "#800080",     # Purple
        "love": "#FF1493",     # Deep Pink
        "surprise": "#FF8C00"  # Dark Orange
    }

# Function to get emoji for emotions
def get_emotion_emoji():
    return {
        "joy": "😄",
        "sadness": "😢",
        "anger": "😠",
        "fear": "😨",
        "love": "❤️",
        "surprise": "😲"
    }


genai.configure(api_key=API_KEY)

def analyze_emotion_with_gemini(user_input, top_emotion):
    try:
        # Khởi tạo model Gemini
        # model = genai.GenerativeModel('gemini-pro')
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Prompt gửi tới Gemini
        contents = f"Câu nói: '{user_input}' được model AI gắn nhãn là: '{vietnamese_labels[top_emotion]}'. Bạn hãy phân tích thật kỹ cảm xúc của câu nói trên và nhận xét về dự đoán của model, cố gắng khen model kệ cả nó dự đoán chưa được tốt"
        
        # Gọi API và nhận phản hồi
        response = model.generate_content(contents)

        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Title and description
st.markdown('<h1 class="main-header">🔍 Ứng dụng Phân tích Cảm xúc</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Phân tích cảm xúc ẩn sau bất kỳ văn bản nào bằng NLP</p>', unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    # Add animation
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_xt3zjzlv.json"  # More elegant emotion analysis animation
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=220, key="emotion_animation")
    else:
        # Fallback if animation doesn't load
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <span style="font-size: 100px;">🔍😀😢😠</span>
        </div>
        """, unsafe_allow_html=True)

with col1:
    # Input text
    user_input = st.text_area("Nhập văn bản tiếng anh để phân tích cảm xúc:", 
                            "I saw a movie today and it was really good.", 
                            height=120)
    
    # Analyze button with centered layout
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("Phân tích Cảm xúc")
    st.markdown('</div>', unsafe_allow_html=True)

# Load the model when the app starts to avoid reloading
@st.cache_resource
def load_model():
    model_id = "Dat1710/distilbert-base-uncased-finetuned-emotion"
    return pipeline("text-classification", model=model_id)

# Try to load the model
try:
    classifier = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Define emotion labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]




# When analyze button is clicked
if analyze_button and model_loaded:
    if user_input.strip():
        # Add a spinner while processing
        with st.spinner("Đang phân tích cảm xúc..."):
            # Get predictions
            preds = classifier(user_input, return_all_scores=True)
            preds_df = pd.DataFrame(preds[0])
            
            # Ensure we use the correct labels instead of indices
            # Map any numeric or incorrect labels to our emotion labels if needed
            if not all(label in emotion_labels for label in preds_df["label"]):
                # If labels are indices, map them to proper emotion names
                label_mapping = {i: emotion_labels[i] if i < len(emotion_labels) else f"unknown_{i}" 
                                for i in range(len(preds_df))}
                preds_df["label"] = preds_df.index.map(lambda x: label_mapping.get(x, f"unknown_{x}"))
            
            # Sort predictions by score
            preds_df = preds_df.sort_values(by="score", ascending=False)
            
            # Get top emotion
            top_emotion = preds_df.iloc[0]["label"]
            top_score = preds_df.iloc[0]["score"]
            # Get emoji and color map
            emotion_emoji = get_emotion_emoji()
            emotion_colors = get_emotion_colors()
            
            # Display results
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                st.markdown(f'<p class="emotion-label">{emotion_emoji.get(top_emotion, "")}{top_emotion.capitalize()}</p>', unsafe_allow_html=True)
                
                # Create a radial gauge for confidence
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = top_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': emotion_colors.get(top_emotion, "#1E88E5")},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': '#EFF3F6'},
                            {'range': [25, 50], 'color': '#DEE5ED'},
                            {'range': [50, 75], 'color': '#CEDBE3'},
                            {'range': [75, 100], 'color': '#BDD1DA'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': top_score * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=30, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#424242", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Create a horizontal bar chart using Plotly
            preds_df["color"] = preds_df["label"].map(lambda x: emotion_colors.get(x, "#CCCCCC"))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y= preds_df["label"].map(vietnamese_labels),
                x=preds_df["score"] * 100,
                orientation='h',
                marker_color=preds_df["color"],
                text=[f"{score:.1f}%" for score in preds_df["score"] * 100],
                textposition='auto',
                hoverinfo="text",
                hovertext=[f"{label.capitalize()}: {score:.2%}" for label, score in zip(preds_df["label"], preds_df["score"])]
            ))
            
            fig.update_layout(
                title={
                    'text': "Emotion Distribution",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': '#424242'}
                },
                xaxis_title="Probability (%)",
                yaxis_title="Emotion",
                height=350,
                margin=dict(l=20, r=20, t=50, b=30),
                xaxis=dict(range=[0, 100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'size': 14, 'color': '#424242'},
                shapes=[
                    # Add reference lines
                    go.layout.Shape(
                        type="line",
                        x0=25, y0=-0.5,
                        x1=25, y1=len(preds_df)-0.5,
                        line=dict(color="#E0E0E0", width=1, dash="dash")
                    ),
                    go.layout.Shape(
                        type="line",
                        x0=50, y0=-0.5,
                        x1=50, y1=len(preds_df)-0.5,
                        line=dict(color="#E0E0E0", width=1, dash="dash")
                    ),
                    go.layout.Shape(
                        type="line",
                        x0=75, y0=-0.5,
                        x1=75, y1=len(preds_df)-0.5,
                        line=dict(color="#E0E0E0", width=1, dash="dash")
                    )
                ]
            )
            
            # Display the chart
            with col_result2:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display insights
            st.markdown('<h3 class="section-header">💡 Phân Tích Cảm Xúc </h3>', unsafe_allow_html=True)
            with st.spinner("Đang phân tích chi tiết cảm xúc ..."):
                gemini_insight = analyze_emotion_with_gemini(user_input, top_emotion)
                st.markdown(f'<div class="insight-bullet">{gemini_insight}</div>', unsafe_allow_html=True)
            
            # Additional insights on emotion distribution
            second_emotion = preds_df.iloc[1]["label"]
            second_score = preds_df.iloc[1]["score"]
            

            # Phần Sentiment Direction sửa thành biểu đồ 6 cảm xúc
            st.markdown('<h3 class="section-header">🧭 Phân Bổ Cảm Xúc</h3>', unsafe_allow_html=True)
            threshold = 0.0001  # 1%
            preds_df = preds_df[preds_df["score"] > threshold]

            col_chart, col_legend = st.columns([3, 1])

            with col_chart:
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=[vietnamese_labels[label] for label in preds_df["label"]],
                    values=preds_df["score"] * 100,
                    marker=dict(colors=[emotion_colors[label] for label in preds_df["label"]]),
                    textinfo="label",
                    insidetextorientation="radial",
                    hole=0.4,
                    hoverinfo="label+percent+value"
                ))
                
                fig.update_layout(
                    showlegend=False,  # Tắt legend mặc định
                    height=400,
                    margin=dict(l=20, r=20, t=100, b=10),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_legend:
                # Tạo bảng xác suất
                st.markdown("""
                <style>
                    .legend-table {
                        background: white;
                        border-radius: 10px;
                        padding: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .legend-item {
                        display: flex;
                        align-items: center;
                        margin: 8px 0;
                    }
                    .color-box {
                        width: 20px;
                        height: 20px;
                        border-radius: 4px;
                        margin-right: 10px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Hiển thị từng cảm xúc với màu và xác suất
                for _, row in preds_df.iterrows():
                    st.markdown(f"""
                    <div class="legend-item">
                        <div class="color-box" style="background: {emotion_colors[row['label']]};"></div>
                        <div>
                            <span style="font-weight:500; color:#424242;">{vietnamese_labels[row['label']]}</span><br>
                            <span style="font-size:0.9em; color:#757575;">{row['score']*100:.4f} %</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("""
<div class="footer">
    <h3>Giới Thiệu Ứng Dụng</h3>
    <p>Ứng dụng sử dụng mô hình DistilBERT đã được tinh chỉnh để phân tích 6 loại cảm xúc chính:</p>
    <ul style="list-style: none; padding: 0; text-align: center;">
        <li>😊 Vui vẻ - 🥺 Buồn bã - 😠 Tức giận</li>
        <li>😨 Sợ hãi - ❤️ Yêu thương - 😲 Ngạc nhiên</li>
    </ul>
    <p>Được xây dựng bằng Hugging Face Transformers và Streamlit</p>
    <p><i>Mô hình: Dat1710/distilbert-base-uncased-finetuned-emotion</i></p>
</div>
""", unsafe_allow_html=True)

# Add requirements information

