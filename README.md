# 🫁 胸部X光AI检测平台

## 📋 项目介绍

这是一个基于深度学习的胸部X光图像检测系统，使用YOLOv8模型进行实时目标检测。用户可以上传胸部X光图片，AI将自动识别可能的异常区域。

**⚠️ 重要声明**: 此工具仅供学习和研究使用，检测结果不能作为医学诊断依据，如有健康问题请咨询专业医生。

## 🚀 在线体验

访问在线平台: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## ✨ 功能特点

- 🔍 **智能检测**: 基于YOLOv8的实时目标检测
- 🖼️ **多格式支持**: 支持JPG、PNG、BMP等常见图像格式
- 📊 **详细分析**: 提供检测置信度、位置信息等详细数据
- 🌐 **在线访问**: 无需安装，通过浏览器即可使用
- 📱 **响应式设计**: 支持桌面和移动设备

## 🛠️ 技术栈

- **前端**: Streamlit
- **AI模型**: YOLOv8 (Ultralytics)
- **图像处理**: OpenCV, PIL
- **数据处理**: NumPy, Pandas
- **部署平台**: Streamlit Cloud

## 📦 本地运行

### 环境要求

- Python 3.8+
- 至少2GB内存

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/your-username/chest-detection.git
cd chest-detection
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
streamlit run app.py
```

4. 打开浏览器访问 `http://localhost:8501`

## 🚀 部署到Streamlit Cloud

### 步骤说明

1. **创建GitHub仓库**
   - 将代码推送到GitHub公开仓库
   - 确保包含 `app.py`、`requirements.txt` 和 `.streamlit/config.toml`

2. **部署到Streamlit Cloud**
   - 访问 [share.streamlit.io](https://share.streamlit.io)
   - 使用GitHub账号登录
   - 选择仓库和分支
   - 主文件设置为 `app.py`
   - 点击Deploy

3. **配置完成**
   - 等待部署完成（通常3-5分钟）
   - 获得专属的 `.streamlit.app` 域名

## 📁 项目结构

```
chest-detection/
├── app.py                 # 主应用文件（云部署版本）
├── web.py                 # 完整版本（本地使用）
├── model.py               # 模型处理模块
├── requirements.txt       # 依赖包列表
├── .streamlit/
│   └── config.toml       # Streamlit配置
├── QtFusion_replacement.py # 工具函数
├── chinese_name_list.py   # 中文标签
└── README.md             # 项目说明
```

## 🔧 配置说明

### 模型自动下载
应用会自动下载以下模型之一：
- `yolo11n.pt` (推荐，体积小，速度快)
- `yolov8n.pt` (备用模型)

### 性能优化
- 使用 `@st.cache_resource` 缓存模型
- 优化图像处理流程
- 适配云服务器资源限制

## 🎯 使用指南

1. 访问在线平台或本地运行应用
2. 在左侧上传胸部X光图片
3. 等待AI分析完成
4. 查看检测结果和统计信息
5. 下载或分享检测报告

## 🤝 贡献

欢迎提交问题和改进建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## ⚠️ 免责声明

- 本项目仅供学术研究和技术学习使用
- AI检测结果不能替代专业医学诊断
- 使用者需自行承担使用风险
- 如有健康问题，请及时咨询专业医生

## 📞 联系方式

- 项目主页: [GitHub Repository](https://github.com/your-username/chest-detection)
- 问题反馈: [Issues](https://github.com/your-username/chest-detection/issues)

---

⭐ 如果这个项目对您有帮助，请给个Star支持一下！ 