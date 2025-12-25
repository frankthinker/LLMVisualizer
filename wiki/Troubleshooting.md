# 常见问题与排障

## 模型下载失败 / 超时
- 侧边栏切换到 “镜像站 (hf-mirror.com)” 或在终端执行 `export HF_ENDPOINT=https://hf-mirror.com`。
- 确保未被公司/校园代理阻断；必要时先手动在浏览器中访问 Hugging Face 并登录。

## Streamlit 显示空白或崩溃
- 检查控制台报错；若提示 `torch_dtype is deprecated` 可忽略。
- 如果出现 `segmentation fault`，通常是 tokenizers 多进程警告导致，可在终端设置 `export TOKENIZERS_PARALLELISM=false`。
- 运行 `python3 -m py_compile gpt2_visualizer.py gpt2_visualizer_streamlit/*.py` 验证语法是否正常。

## 注意力图为空
- 确认已生成足够长的文本；如果输入过短或模型未生成任何 Token，注意力图会缺失。
- 当选择超过 3 层注意力时，使用“注意力层窗口”滑块切换要查看的层。

## 中文输入效果差
- GPT-2 以英文语料训练；建议将问题翻译成英文或通过示例按钮生成英文 prompt。
- 原始输出一栏会提示仅展示前 200 个词，可在导出的 HTML/PNG 中查看完整内容。

## 模型缓存位置不对
- 默认缓存目录是 `~/.cache/huggingface/hub`。若此前手动修改过 `TRANSFORMERS_CACHE`，清理环境变量后重启应用即可恢复默认。
