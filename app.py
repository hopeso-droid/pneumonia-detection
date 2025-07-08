#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
胸部X光肺炎检测在线平台 - Streamlit Cloud版本
专业肺炎检测模型，专为云部署优化
"""

import streamlit as st
import numpy as np
import os
import tempfile
import time
from PIL import Image
import io
import requests
from pathlib import Path

# 安全导入OpenCV
try:
    import cv2
except ImportError as e:
    st.error(f"❌ OpenCV导入失败: {str(e)}")
    st.info("🔄 正在尝试修复...")
    # 提供无OpenCV的备用方案
    cv2 = None

# 设置页面配置
st.set_page_config(
    page_title="🫁 胸部X光肺炎AI检测平台",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 中文标签映射
CHINESE_LABELS = {
    'high-pneumonia': '高度肺炎',
    'low-pneumonia': '轻度肺炎', 
    'no-pneumonia': '正常/无肺炎'
}

def download_model_from_github():
    """从GitHub Release下载模型文件或使用本地缓存"""
    local_model_path = "best.pt"
    cache_dir = Path.home() / ".cache" / "pneumonia-detection"
    cached_model_path = cache_dir / "best.pt"
    
    # 优先使用缓存文件
    if cached_model_path.exists():
        # st.info("📁 使用缓存的专业肺炎检测模型")
        return str(cached_model_path)
    
    # 其次使用本地文件（便于开发和测试）
    if os.path.exists(local_model_path):
        # st.info("📁 使用本地专业肺炎检测模型")
        return local_model_path
    
    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 从GitHub Release下载
    model_urls = [
        "https://github.com/hopeso-droid/pneumonia-detection/releases/download/v1.0/best.pt",  # Release方式（推荐）
        "https://github.com/hopeso-droid/pneumonia-detection/raw/main/run/weights/best.pt",  # 直接文件方式（备用）
    ]
    
    for i, model_url in enumerate(model_urls):
        try:
            method_name = "GitHub Release" if i == 0 else "直接文件"
            st.info(f"🔄 正在从{method_name}下载专业肺炎检测模型（仅首次需要）...")
            
            with st.spinner("下载中，请稍候..."):
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # 创建进度条
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with open(cached_model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                progress_bar.progress(progress)
                                status_text.text(f"已下载: {downloaded // 1024 // 1024:.1f}MB / {total_size // 1024 // 1024:.1f}MB")
                
                # 清除进度显示
                progress_container.empty()
                st.success(f"✅ 专业肺炎检测模型下载完成并已缓存！（来源：{method_name}）")
                st.info("💡 下次使用时将直接从缓存加载，无需重复下载")
                return str(cached_model_path)
                
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                st.warning(f"⚠️ {method_name}中未找到模型文件")
                if i < len(model_urls) - 1:
                    st.info("🔄 尝试其他下载方式...")
                    continue
            else:
                st.error(f"❌ {method_name}下载失败: {str(e)}")
        except Exception as e:
            st.error(f"❌ {method_name}下载失败: {str(e)}")
            if i < len(model_urls) - 1:
                st.info("🔄 尝试其他下载方式...")
                continue
    
    # 如果所有方式都失败
    st.error("❌ 所有下载方式都失败")
    st.info("💡 请确认Release已创建或将best.pt文件推送到main分支")
    st.info("🔄 现在使用备用通用模型...")
    return None

@st.cache_resource
def load_model():
    """缓存模型加载，避免重复加载"""
    try:
        from ultralytics import YOLO
        import torch
        
        # 设置PyTorch安全设置
        os.environ['TORCH_ALLOW_UNSAFE_PICKLES'] = '1'
        
        # 首先尝试加载自定义肺炎检测模型
        custom_model_path = download_model_from_github()
        
        if custom_model_path and os.path.exists(custom_model_path):
            try:
                # 静默加载，不显示任何信息
                model = YOLO(custom_model_path)
                return model, "custom"
            except Exception as e:
                st.warning(f"⚠️ 自定义模型加载失败: {str(e)}")
        
        # 备用方案：使用通用检测模型
        st.info("🔄 使用通用检测模型...")
        fallback_models = ["yolo11n.pt", "yolov8n.pt"]
        
        for model_name in fallback_models:
            try:
                model = YOLO(model_name)
                st.warning("⚠️ 使用通用检测模型，可能不如专业肺炎模型准确")
                return model, "general"
            except Exception as e:
                st.warning(f"⚠️ 无法加载 {model_name}: {str(e)}")
                continue
        
        st.error("❌ 所有模型加载失败")
        return None, None
        
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None, None

def get_severity_color(class_name):
    """根据检测结果返回对应的颜色"""
    color_map = {
        'high-pneumonia': '#FF4444',  # 红色 - 高风险
        'low-pneumonia': '#FFA500',   # 橙色 - 中风险
        'no-pneumonia': '#00AA00'     # 绿色 - 正常
    }
    return color_map.get(class_name, '#666666')

def get_risk_level(class_name):
    """获取风险等级"""
    risk_map = {
        'high-pneumonia': '🔴 高风险',
        'low-pneumonia': '🟡 中风险', 
        'no-pneumonia': '🟢 正常'
    }
    return risk_map.get(class_name, '⚪ 未知')

def process_image(image, model, model_type):
    """处理图像并返回检测结果"""
    if model is None:
        return None, None, "模型未加载"
    
    try:
        # 预处理图像
        if cv2 is not None and len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # 备用方案：直接使用RGB图像
            image_bgr = image
        
        # 进行预测
        with st.spinner("🤖 AI正在分析肺部影像..."):
            start_time = time.time()
            results = model(image_bgr, conf=0.25, iou=0.45)
            end_time = time.time()
        
        # 处理结果
        if len(results) > 0 and len(results[0].boxes) > 0:
            annotated_image = results[0].plot()
            
            boxes = results[0].boxes
            detections = []
            
            for box in boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                confidence = float(box.conf.cpu().numpy()[0])
                bbox = box.xyxy.cpu().numpy()[0]
                
                # 获取类别名称
                class_name = model.names[class_id] if hasattr(model, 'names') else f"Class_{class_id}"
                
                # 转换为中文标签（如果是自定义模型）
                if model_type == "custom":
                    chinese_name = CHINESE_LABELS.get(class_name, class_name)
                else:
                    chinese_name = class_name
                
                detections.append({
                    'class': class_name,
                    'chinese_class': chinese_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'risk_level': get_risk_level(class_name) if model_type == "custom" else "🔍 检测到"
                })
            
            process_time = f"{end_time - start_time:.2f}秒"
            return annotated_image, detections, process_time
        else:
            process_time = f"{end_time - start_time:.2f}秒"
            return image_bgr, [], process_time
            
    except Exception as e:
        return None, None, f"检测过程出错: {str(e)}"

def main():
    """主应用程序"""
    
    # 标题和介绍
    st.title("🫁 胸部X光肺炎AI检测平台")
    st.markdown("""
    > 🩺 **专业的肺炎AI检测工具**  
    > 上传胸部X光图片，AI将智能识别肺炎风险等级  
    > ⚠️ **重要提醒**: 此工具仅供参考，不能替代专业医学诊断，请及时就医
    """)
    
    # 侧边栏
    with st.sidebar:
        st.header("🛠️ 检测设置")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "📁 上传胸部X光图片",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="支持 JPG、PNG、BMP 格式"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 检测说明
        
        **专业肺炎模型可检测:**
        - 🔴 **高度肺炎**: 明显肺炎征象
        - 🟡 **轻度肺炎**: 轻微肺炎征象  
        - 🟢 **正常/无肺炎**: 未发现肺炎征象
        
        ### ⚠️ 医学免责声明
        - 本工具基于AI算法，仅供辅助参考
        - **不能替代专业医学诊断**
        - 如有症状请及时就医咨询专业医生
        - AI结果可能存在误差，请谨慎判断
        
        ### 📞 紧急情况
        如有急性症状，请立即就医！
        """)
    
    # 主界面
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 原始影像")
    
    with col2:
        st.subheader("🎯 AI检测结果")
    
    # 加载模型
    model, model_type = load_model()
    
    if uploaded_file is not None:
        try:
            # 读取图像
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # 显示原始图像
            with col1:
                st.image(image, caption=f"📁 {uploaded_file.name}", width=600)
                
                # 显示图像信息
                st.info(f"""
                📊 **影像信息**
                - 文件名: {uploaded_file.name}
                - 尺寸: {image_array.shape[1]} × {image_array.shape[0]} 像素
                - 格式: {image.format}
                - 大小: {len(uploaded_file.getvalue()) / 1024:.1f} KB
                """)
            
            # 处理图像
            if model is not None:
                result_image, detections, process_time = process_image(image_array, model, model_type)
                
                with col2:
                    if result_image is not None:
                        # 显示检测结果图像
                        if isinstance(result_image, np.ndarray):
                            if cv2 is not None:
                                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            else:
                                result_image_rgb = result_image
                            st.image(result_image_rgb, caption="🎯 AI检测结果", width=600)
                        
                        # 显示检测统计
                        if detections:
                            st.success(f"🎉 检测完成！发现 {len(detections)} 处异常区域")
                            
                            # 风险评估汇总
                            if model_type == "custom":
                                risk_counts = {}
                                for det in detections:
                                    risk = det['risk_level']
                                    risk_counts[risk] = risk_counts.get(risk, 0) + 1
                                
                                st.markdown("### 📋 风险评估汇总")
                                for risk, count in risk_counts.items():
                                    st.markdown(f"- {risk}: {count} 处")
                            
                            # 创建详细检测结果表格
                            import pandas as pd
                            df_data = []
                            for i, det in enumerate(detections):
                                df_data.append({
                                    '序号': i + 1,
                                    '检测结果': det['chinese_class'] if model_type == "custom" else det['class'],
                                    '风险等级': det['risk_level'],
                                    '置信度': f"{det['confidence']:.1%}",
                                    '位置': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f})"
                                })
                            
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # 显示处理时间
                            st.info(f"⏱️ 检测用时: {process_time}")
                            
                            # 专业建议
                            if model_type == "custom":
                                high_risk = any('high-pneumonia' in det['class'] for det in detections)
                                if high_risk:
                                    st.error("""
                                    🚨 **重要提醒**
                                    
                                    检测到高度肺炎征象，建议：
                                    - 立即咨询专业医生
                                    - 进行进一步检查
                                    - 密切关注症状变化
                                    """)
                                elif any('low-pneumonia' in det['class'] for det in detections):
                                    st.warning("""
                                    ⚠️ **注意事项**
                                    
                                    检测到轻度肺炎征象，建议：
                                    - 及时就医确诊
                                    - 注意休息和营养
                                    - 遵医嘱用药
                                    """)
                                else:
                                    st.success("""
                                    ✅ **检测结果良好**
                                    
                                    未发现明显肺炎征象，但仍建议：
                                    - 定期健康检查
                                    - 如有症状及时就医
                                    - 保持健康生活方式
                                    """)
                            
                        else:
                            st.info("✅ 未检测到明显异常区域")
                            st.info(f"⏱️ 检测用时: {process_time}")
                            if model_type == "custom":
                                st.success("""
                                ✅ **检测结果良好**
                                
                                AI未检测到肺炎征象，但建议：
                                - 如有症状仍需就医确诊
                                - 定期进行健康检查
                                - 保持良好生活习惯
                                """)
                    else:
                        st.error(f"❌ 检测失败: {process_time}")
            else:
                with col2:
                    st.error("❌ 模型加载失败，无法进行检测")
                    
        except Exception as e:
            st.error(f"❌ 图像处理失败: {str(e)}")
    
    else:
        # 显示默认内容
        with col1:
            st.info("👆 请在左侧上传胸部X光图片开始检测")
            
            # 显示样例说明
            st.markdown("""
            ### 📖 使用建议
            
            **适用图像类型:**
            - 胸部X光正位片
            - 清晰的医学影像
            - 标准的胸片格式
            
            **不适用:**
            - 模糊不清的图像
            - 非胸部X光图像
            - 过度曝光的照片
            """)
        
        with col2:
            st.info("🎯 AI检测结果将在这里显示")
            
            # 显示模型信息
            if model is not None:
                if model_type == "custom":
                    # st.success("""
                    # 🧠 **专业肺炎检测模型已就绪**
                    # 
                    # - 专门训练用于肺炎检测
                    # - 可识别不同严重程度
                    # - 提供风险等级评估
                    # """)
                    pass
                else:
                    st.warning("""
                    🧠 **通用检测模型已就绪**
                    
                    - 使用通用目标检测模型
                    - 准确性可能不如专业模型
                    - 建议谨慎解读结果
                    """)
    
    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚀 合溪生物科技 | 
        💻 部署于 <a href='https://streamlit.io/cloud'>Streamlit Cloud</a> | 
        🩺 专业肺炎检测模型 | 
        🔬 仅供医学研究使用
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 