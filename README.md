# LLMVisualizer - GPT-2 可视化工具

一个平民级、GUI驱动的 GPT-2 可视化工具，无需专业算力，通过直观交互界面揭示大模型的核心内部机制。

## 功能特性

### 三种核心可视化维度

1. **注意力权重层级可视化**
   - 展示 GPT-2 各层各头的注意力热力图
   - 底层关注相邻 Token，高层关注核心逻辑 Token

2. **Token 推理时序流可视化**
   - 动态展示模型逐 Token 生成的过程
   - 高概率 Token 和低概率 Token 不同颜色标注

3. **语义空间层级聚类**
   - 使用 PCA/t-SNE 降维展示 Token 语义编码
   - 揭示模型对 Token 的抽象能力

## 快速开始

### 环境要求

- Python 3.8+
- 4GB+ 内存（CPU 运行即可）

### 安装

```bash
# 克隆仓库
git clone https://github.com/frankthinker/LLMVisualizer.git
cd LLMVisualizer

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

### 使用说明

1. **加载模型**: 在侧边栏选择模型大小并点击"加载模型"
2. **输入文本**: 在文本框输入中文或英文文本
3. **调整参数**: 设置生成长度、温度、Top-K 等参数
4. **生成并可视化**: 点击按钮查看三种可视化结果

## 项目结构

```
LLMVisualizer/
├── app.py           # Streamlit 主界面
├── gpt2_loader.py   # GPT-2 模型加载与文本生成
├── visualizer.py    # 可视化模块
├── requirements.txt # 依赖清单
└── wiki/           # 文档目录
```

## 技术栈

- **Streamlit** - Web GUI 框架
- **Transformers** - Hugging Face 模型库
- **PyTorch** - 深度学习框架
- **Plotly** - 交互式可视化
- **scikit-learn** - 机器学习降维

## License

MIT License
