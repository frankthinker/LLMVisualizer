"""Streamlit UI for the GPT-2 visual exploration tool."""
from __future__ import annotations

from typing import Dict, Optional

import platform
import subprocess
from pathlib import Path

import streamlit as st
from .gpt2_loader import (
    CACHE_DIR,
    HF_MIRROR_DEFAULT,
    MODEL_SPECS,
    GenerationArtifacts,
    GenerationSettings,
    clear_cached_models,
    describe_model,
    run_generation,
)
from .visualizer import (
    build_attention_figure,
    build_semantic_cluster_figure,
    build_token_dataframe,
    build_token_flow_chart,
    figure_to_html_bytes,
    figure_to_png_bytes,
)

SAMPLE_GROUPS = [
    {
        "key": "math_reasoning",
        "label": "示例1：数学推理",
        "description": "英文算术推理问题，观察模型对数字与逻辑词的关注。",
        "prompts": [
            "Mia has 8 apples and gives 2 apples to each of her three friends. How many apples does she have left?",
            "A bakery sold 45 tickets on Friday and twice as many on Saturday. How many tickets were sold during the weekend?"
        ],
    },
    {
        "key": "science_explain",
        "label": "示例2：科学解读",
        "description": "英文科普说明，适合查看语义链路。",
        "prompts": [
            "Explain the process of photosynthesis to a middle-school student in three clear steps.",
            "How does the water cycle move moisture from warm oceans to snowy mountains? Answer in concise English."
        ],
    },
    {
        "key": "story_logic",
        "label": "示例3：故事推理",
        "description": "英文故事链，突出指代与追踪。",
        "prompts": [
            "A cat chases a mouse, a dog chases the cat, and a boy whistles for the dog. Who controls the chase and why?",
            "Maria hands a key to Ben, Ben shares it with Lila, and Lila returns it to Maria. Who can open the locker last?"
        ],
    },
    {
        "key": "coding_reasoning",
        "label": "示例4：代码推演",
        "description": "英文代码解释，展示抽象推理。",
        "prompts": [
            "Describe step by step how a stack handles the sequence push(3), push(5), pop(), push(7).",
            "Predict what this Python loop prints: total = 0; for n in range(1, 6): total += n; print(total)."
        ],
    },
    {
        "key": "analogy_summary",
        "label": "示例5：类比总结",
        "description": "英文类比或总结，查看高层语义。",
        "prompts": [
            "Compare teamwork in an ant colony to collaboration inside a human company.",
            "What planning lessons can people learn from the way beavers build dams?"
        ],
    },
]


def _init_session_state() -> None:
    """Initialize Streamlit session state keys with defaults."""

    defaults = {
        "prompt_text": SAMPLE_GROUPS[0]["prompts"][0],
        "artifacts": None,
        "selected_token": None,
        "guide_dismissed": False,
        "topk_warned": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for group in SAMPLE_GROUPS:
        idx_key = f"sample_idx_{group['key']}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0


def _show_guide_modal() -> None:
    """Display first-launch usage hints."""

    if st.session_state.get("guide_dismissed", False):
        return

    guide_content = """
    **GPT-2 可视化工具快速入门**

    1. 在左侧输入框中输入待分析文本或加载示例。
    2. 右侧侧边栏依次选择模型规模、生成长度与采样参数。
    3. 点击 **生成并可视化**，耐心等待状态栏加载完成。
    4. 四个标签页分别展示输出文本、注意力热力图、Token 推理时序、语义空间聚类。

    📌 鼠标悬停即可查看数值详情；点击热力图 Token 可跨层高亮；任意图表都可以导出成 HTML 或 PNG。
    """

    if hasattr(st, "modal"):
        with st.modal("使用指南", key="guide-modal"):
            st.markdown(guide_content)
        if st.button("我已了解", type="primary", key="guide-dismiss-modal"):
            st.session_state["guide_dismissed"] = True
    else:
        with st.sidebar.expander("使用指南", expanded=True):
            st.markdown(guide_content)
            if st.button("关闭指南", key="guide-dismiss-expander"):
                st.session_state["guide_dismissed"] = True


def _limit_words(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def _open_cache_dir(path: Path) -> None:
    """Open the model cache directory in the system file explorer."""

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:  # pragma: no cover
        st.warning(f"无法打开文件夹：{exc}")


def _apply_theme_styles(theme_choice: str) -> None:
    """Inject CSS so the entire页面跟随所选配色."""

    if theme_choice == "dark":
        bg_color = "#0b1120"
        card_color = "#111827"
        text_color = "#e5e7eb"
        accent = "#00c2c7"
    else:
        bg_color = "#f7f8fb"
        card_color = "#ffffff"
        text_color = "#1f2937"
        accent = "#4757e6"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {bg_color};
                color: {text_color};
            }}
            .stApp [data-testid="stHeader"] {{
                background: transparent;
            }}
            .stApp div[data-testid="stSidebar"] {{
                background-color: {card_color};
            }}
            .stApp .stTabs [data-baseweb="tab-list"] button[role="tab"] {{
                background-color: {card_color};
                color: {text_color};
            }}
            .stApp .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
                border-bottom: 3px solid {accent};
            }}
            .stApp .stDataFrame, .stApp .stPlotlyChart {{
                background-color: {card_color};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> Dict[str, object]:
    """Render controls and return selected configuration."""

    st.sidebar.header("参数调节")
    theme_choice = st.sidebar.radio("配色模式", ["light", "dark"], format_func=lambda x: "亮色" if x == "light" else "暗色")
    model_size = st.sidebar.selectbox(
        "GPT-2 版本",
        options=list(MODEL_SPECS.keys()),
        format_func=lambda key: MODEL_SPECS[key]["display"],
        index=0,
    )
    spec = MODEL_SPECS[model_size]
    model_layers = spec["layers"]
    model_heads = spec["heads"]
    st.sidebar.caption(
        f"{spec['display']} · 参数量 {spec.get('params')} · 层 {spec['layers']} · 头 {spec['heads']} · 上下文 {spec['context']} tokens"
    )
    max_tokens = st.sidebar.slider("生成长度", 0, 300, 120, step=5)
    temperature = st.sidebar.slider("温度", 0.1, 1.0, 0.7, step=0.05)
    top_k = st.sidebar.slider("Top-K", 1, 50, 5, step=1)
    if top_k > 10 and not st.session_state.get("topk_warned"):
        st.sidebar.warning("Top-K 超过 10 会显著降低回答准确度，仅供研究用途。")
        st.session_state["topk_warned"] = True
    attention_layers = st.sidebar.multiselect(
        "注意力层 (可多选)",
        options=list(range(1, model_layers + 1)),
        default=[1, model_layers // 2, model_layers],
    )
    attention_heads = st.sidebar.multiselect(
        "注意力头 (可多选)",
        options=list(range(1, model_heads + 1)),
        default=list(range(1, min(12, model_heads) + 1)),
    )
    viz_dims = st.sidebar.multiselect(
        "可视化维度",
        options=["注意力权重", "Token 推理流", "语义聚类"],
        default=["注意力权重", "Token 推理流", "语义聚类"],
    )
    embed_method = st.sidebar.radio("语义降维方法", ["pca", "tsne"], format_func=lambda x: x.upper())
    context_limit = st.sidebar.slider(
        "上下文窗口 (tokens)",
        min_value=256,
        max_value=int(spec["context"]),
        value=min(768, int(spec["context"])),
        step=64,
    )
    source_choice = st.sidebar.radio(
        "模型下载来源",
        options=["official", "mirror"],
        format_func=lambda key: "Hugging Face 官网" if key == "official" else "镜像站 (hf-mirror.com)",
    )
    hf_endpoint: Optional[str] = None
    if source_choice == "mirror":
        default_mirror = st.session_state.get("mirror_endpoint", HF_MIRROR_DEFAULT)
        mirror_input = st.sidebar.text_input("镜像地址", value=default_mirror, help="示例：https://hf-mirror.com")
        resolved_mirror = mirror_input.strip() or HF_MIRROR_DEFAULT
        st.session_state["mirror_endpoint"] = resolved_mirror
        hf_endpoint = resolved_mirror
    else:
        hf_endpoint = None

    with st.sidebar.expander("模型缓存与文件"):
        if st.button("在资源管理器中查看模型缓存", key="open-cache"):
            _open_cache_dir(CACHE_DIR)
        if st.button("清理内存中的模型 (释放显存/RAM)", key="clear-cache"):
            clear_cached_models()
            st.toast("已清理所有模型缓存，下次推理会重新加载。")
        st.caption(f"缓存目录：`{CACHE_DIR}`")

    return {
        "theme": theme_choice,
        "settings": GenerationSettings(
            model_size=model_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            attention_layers=attention_layers,
            attention_heads=attention_heads,
            max_context_tokens=context_limit,
        ),
        "viz_dims": viz_dims,
        "embed_method": embed_method,
        "hf_endpoint": hf_endpoint,
    }


def _handle_actions(config: Dict[str, object]) -> None:
    """Handle generation, clearing, and export triggers."""

    artifacts = st.session_state.get("artifacts")
    report_bytes = (
        _build_report_html(artifacts, config).encode("utf-8") if artifacts else b""
    )

    col_run, col_clear, col_export = st.columns([2, 1, 1])
    run_clicked = col_run.button("生成并可视化", type="primary", width="stretch")
    clear_clicked = col_clear.button("清空所有结果", width="stretch", disabled=artifacts is None)
    col_export.download_button(
        "导出汇总 (HTML)",
        data=report_bytes,
        file_name="gpt2_visual_report.html",
        mime="text/html",
        width="stretch",
        disabled=artifacts is None,
    )

    if clear_clicked:
        st.session_state["artifacts"] = None
        st.session_state["selected_token"] = None
        st.session_state["prompt_text"] = ""
        st.toast("已清空历史结果。")

    if run_clicked:
        _run_inference(config["settings"], config["hf_endpoint"])


def _build_report_html(artifacts: GenerationArtifacts, config: Dict[str, object]) -> str:
    """Compose a lightweight HTML report summarizing key outputs."""

    text_section = f"""<h2>原始输出</h2><p>{artifacts.generated_text}</p>"""
    endpoint_label = config.get("hf_endpoint") or "huggingface.co"
    meta_section = (
        f"<p>模型: {describe_model(artifacts.model_size)} / 温度 {config['settings'].temperature} / "
        f"Top-K {config['settings'].top_k} / 下载源 {endpoint_label}</p>"
    )
    tables = build_token_dataframe(artifacts).to_html(index=False)
    return f"<html><body>{meta_section}{text_section}<h3>Token 明细</h3>{tables}</body></html>"


def _run_inference(settings: GenerationSettings, hf_endpoint: Optional[str]) -> None:
    """Execute GPT-2 generation with progress feedback."""

    prompt = st.session_state.get("prompt_text", "").strip()
    if not prompt:
        st.warning("请输入或加载一段文本。")
        return

    load_bar = st.progress(0, text="准备加载 GPT-2 模型...")
    infer_bar = st.progress(0, text="等待推理开始...")

    def progress_callback(stage: str, percent: int) -> None:
        pct = max(0, min(100, int(percent)))
        if stage == "download":
            load_bar.progress(pct, text=f"模型加载 {pct}%")
        elif stage == "inference":
            infer_bar.progress(pct, text=f"推理进度 {pct}%")

    try:
        artifacts = run_generation(
            prompt,
            settings,
            hf_endpoint=hf_endpoint,
            progress_callback=progress_callback,
        )
        infer_bar.progress(100, text="推理进度 100%")
        st.session_state["artifacts"] = artifacts
        st.session_state["selected_token"] = None
        load_bar.progress(100, text="模型加载完成")
        infer_bar.progress(100, text="推理完成")
        st.success("生成和特征抽取完成。")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"生成失败：{exc}")
    finally:
        load_bar.empty()
        infer_bar.empty()


def _render_prompt_area() -> None:
    """Render text input and sample chips."""

    st.subheader("输入区")
    st.caption("输入任意文本，或使用下方示例按钮。⚠️ GPT-2 以英文语料为主，建议优先输入英文内容。")

    button_cols = st.columns(len(SAMPLE_GROUPS))
    for col, group in zip(button_cols, SAMPLE_GROUPS):
        if col.button(
            group["label"],
            key=f"sample-btn-{group['key']}",
            help=group["description"],
            width="stretch",
        ):
            idx_key = f"sample_idx_{group['key']}"
            current_idx = st.session_state.get(idx_key, 0)
            prompt = group["prompts"][current_idx]
            st.session_state["prompt_text"] = prompt
            st.session_state[idx_key] = (current_idx + 1) % len(group["prompts"])
            st.toast(f"已经载入 {group['label']} · 示例 {current_idx + 1}")

    st.session_state["prompt_text"] = st.text_area(
        "待分析文本",
        value=st.session_state["prompt_text"],
        height=150,
        placeholder="例如：Explain how light turns into energy inside a plant leaf.",
    )
    st.caption("Tip: Use concise English prompts to obtain more stable attention and semantic visualizations.")


def _render_results(config: Dict[str, object]) -> None:
    """Show visualization tabs based on generated artifacts."""

    artifacts: Optional[GenerationArtifacts] = st.session_state.get("artifacts")
    if not artifacts:
        st.info("等待生成结果后将显示可视化内容。")
        return

    tabs = st.tabs(["原始输出", "注意力可视化", "Token 推理时序流", "语义空间聚类"])

    with tabs[0]:
        st.markdown("### 原始输出 (含 Token)")
        st.code(_limit_words(artifacts.generated_text or "(模型生成为空)", 200))
        token_df = build_token_dataframe(artifacts)
        st.dataframe(token_df, width="stretch", hide_index=True)
        st.caption("提示：概率越高的 Token 越确定，困惑度展示模型不确定的节点。")

    with tabs[1]:
        if "注意力权重" in config["viz_dims"]:
            st.markdown("### 注意力层级")
            available_layers = config["settings"].attention_layers
            visible_layers = available_layers
            if len(available_layers) > 3:
                max_start = len(available_layers) - 3
                start_idx = st.slider(
                    "选择注意力层窗口（一次最多展示 3 层）",
                    min_value=0,
                    max_value=max_start,
                    value=st.session_state.get("attention_layer_window", 0),
                    key="attention-layer-window-slider",
                )
                st.session_state["attention_layer_window"] = start_idx
                visible_layers = available_layers[start_idx : start_idx + 3]
                st.caption(f"当前展示层：{visible_layers} · 共选择 {len(available_layers)} 层，可拖动滑块切换。")
            else:
                st.caption(f"当前展示层：{visible_layers}")

            token_count = len(artifacts.tokens)
            highlight_default = min(st.session_state.get("selected_token") or 0, max(token_count - 1, 0))
            if token_count > 1:
                highlight_idx = st.slider(
                    "选择高亮 Token 索引",
                    min_value=0,
                    max_value=token_count - 1,
                    value=highlight_default,
                    key="token-highlight-slider",
                )
                st.session_state["selected_token"] = highlight_idx
                st.caption(f"当前 Token #{highlight_idx}: `{artifacts.tokens[highlight_idx]}`")
            else:
                st.session_state["selected_token"] = None
            attention_fig, summaries = build_attention_figure(
                artifacts,
                layers=visible_layers,
                heads=config["settings"].attention_heads,
                theme=config["theme"],
                selected_token=st.session_state.get("selected_token"),
            )
            st.plotly_chart(attention_fig, width="stretch")
            st.markdown("\n".join([f"- {summary}" for summary in summaries]))
            _render_download_row(attention_fig, prefix="attention")
            st.caption("底层层关注语法邻近，高层层聚焦抽象语义。通过上方滑块挑选 Token，可跨层追踪其注意力。")
        else:
            st.warning("已关闭注意力视图，可在侧边栏重新启用。")

    with tabs[2]:
        if "Token 推理流" in config["viz_dims"]:
            st.markdown("### 推理流程")
            flow_fig, flow_df = build_token_flow_chart(artifacts, theme=config["theme"])
            st.plotly_chart(flow_fig, width="stretch")
            st.dataframe(flow_df, width="stretch", hide_index=True)
            _render_download_row(flow_fig, prefix="token_flow")
            st.caption("红色标记表示模型高度自信，黄色意味着犹豫节点，可用来解释生成节奏。")
        else:
            st.warning("已关闭 Token 推理流视图。")

    with tabs[3]:
        if "语义聚类" in config["viz_dims"]:
            st.markdown("### 语义聚类")
            cluster_fig, cluster_df = build_semantic_cluster_figure(
                artifacts,
                layers=config["settings"].attention_layers,
                method=config["embed_method"],
                theme=config["theme"],
            )
            st.plotly_chart(cluster_fig, width="stretch")
            st.dataframe(cluster_df.head(100), width="stretch", hide_index=True)
            _render_download_row(cluster_fig, prefix="semantic")
            st.caption("散点位置展示 Token 在语义空间的投影，相同颜色表示语义类别。")
        else:
            st.warning("已关闭语义聚类视图。")


def _render_download_row(fig, prefix: str) -> None:
    """Render HTML and PNG download buttons for a figure."""

    col_html, col_png = st.columns(2)
    with col_html:
        st.download_button(
            "导出 HTML",
            data=figure_to_html_bytes(fig),
            file_name=f"{prefix}.html",
            mime="text/html",
            width="stretch",
        )
    with col_png:
        st.download_button(
            "导出 PNG",
            data=figure_to_png_bytes(fig),
            file_name=f"{prefix}.png",
            mime="image/png",
            width="stretch",
        )


def main() -> None:
    """Entrypoint for Streamlit."""

    st.set_page_config(page_title="GPT-2 可视化工作台", page_icon="🧠", layout="wide")
    _init_session_state()
    _show_guide_modal()
    config = _render_sidebar()
    _apply_theme_styles(config["theme"])
    st.title("GPT-2 可视化工作台")
    st.caption("面向非专业用户的层级思维透视——在 CPU 上也能运行的可视化工具。")
    st.markdown(f"当前模型：{describe_model(config['settings'].model_size)}")
    _render_prompt_area()
    _handle_actions(config)
    st.divider()
    _render_results(config)


if __name__ == "__main__":
    main()
