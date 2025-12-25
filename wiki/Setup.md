# Setup Guide

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (CPU mode)
- Internet connection (for initial model download)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/frankthinker/LLMVisualizer.git
cd LLMVisualizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

For users in China, set HuggingFace mirror:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 4. Launch the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Troubleshooting

### Model Download Fails

If model download fails due to network issues:
1. Try using the HuggingFace mirror
2. Manually download model files
3. Check firewall/proxy settings

### Out of Memory

If you experience memory issues:
1. Reduce `max_new_tokens` parameter
2. Use GPT-2 Small instead of Medium
3. Close other memory-intensive applications
