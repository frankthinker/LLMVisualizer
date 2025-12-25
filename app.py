"""Streamlit UI for the GPT-2 visual exploration tool."""
from __future__ import annotations

from typing import Dict, Optional

import platform
import subprocess
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
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
        "key": "science_explain",
        "label": "示例1：科学解读",
        "description": "英文科普说明，适合查看语义链路。",
        "prompts": [
            "Explain the process of photosynthesis to a middle-school student in three clear steps.",
            "How does the water cycle move moisture from warm oceans to snowy mountains? Answer in concise English.",
            "Describe how a solar eclipse happens and why it is brief.",
            "Why do metal objects feel colder than wood even when both are in the same room?",
            "Outline how vaccines train the immune system to recognize viruses.",
            "Explain plate tectonics and how it creates earthquakes at fault lines.",
            "Summarize the greenhouse effect and its role in climate change.",
            "How do bees use vibration and smell to locate flowers?",
            "Describe the difference between potential energy and kinetic energy using a roller coaster example.",
            "Explain why salt lowers the freezing point of water when we melt snow on sidewalks."
        ],
    },
    {
        "key": "story_logic",
        "label": "示例2：故事推理",
        "description": "英文故事链，突出指代与追踪。",
        "prompts": [
            "A cat chases a mouse, a dog chases the cat, and a boy whistles for the dog. Who controls the chase and why?",
            "Maria hands a key to Ben, Ben shares it with Lila, and Lila returns it to Maria. Who can open the locker last?",
            "Olivia lends her notebook to Kai, Kai forgets it in Maya's bag, and Maya mails it back. Describe the chain of responsibility.",
            "A detective hears three conflicting alibis from siblings. Explain how he can test who is lying.",
            "Grandma bakes pies, leaves one for each grandchild, but two cousins share. Who got the extra slice?",
            "Eli hides a clue under a red chair, Nora moves the chair, and Sam discovers the clue. Who actually solved the puzzle?",
            "Describe how a relay race team depends on each runner not dropping the baton.",
            "A librarian mislabels a book, a student checks it out, and the teacher relies on it. What misunderstanding could happen?",
            "Explain who ultimately owns a painting when it is leased from an artist to a gallery and bought by a collector.",
            "A pilot, a mechanic, and an air-traffic controller share partial information. Show how they cooperate to avoid a delay."
        ],
    },
    {
        "key": "coding_reasoning",
        "label": "示例3：代码推演",
        "description": "英文代码解释，展示抽象推理。",
        "prompts": [
            "Describe step by step how a stack handles the sequence push(3), push(5), pop(), push(7).",
            "Predict what this Python loop prints: total = 0; for n in range(1, 6): total += n; print(total).",
            "Explain what happens when a queue processes enqueue(1), enqueue(4), dequeue(), enqueue(9), dequeue().",
            "In pseudocode, what does a binary search do when the target is smaller than the middle element?",
            "Trace the values of i and sum in: sum=1; for i in range(1,4): sum *= (i+1).",
            "Why does a recursive factorial function need a base case, and what happens without it?",
            "Walk through how a hash map resolves collisions using linear probing.",
            "Explain the time complexity difference between bubble sort and merge sort in simple terms.",
            "Given a Python dictionary comprehension `{k: k*k for k in range(1,5)}`, list the key-value pairs.",
            "Describe how depth-first search explores a tree compared to breadth-first search."
        ],
    },
    {
        "key": "analogy_summary",
        "label": "示例4：类比总结",
        "description": "英文类比或总结，查看高层语义。",
        "prompts": [
            "Compare teamwork in an ant colony to collaboration inside a human company.",
            "What planning lessons can people learn from the way beavers build dams?",
            "How is a library similar to a well-organized knowledge base inside a computer?",
            "Relate the growth of a city to the way neurons form connections in the brain.",
            "Why is mentoring a new teammate similar to transplanting a seedling into fertile soil?",
            "Explain how a symphony orchestra resembles a cross-functional software team.",
            "Compare a bee colony's decision making to how open-source communities choose priorities.",
            "What can managers learn from the way penguins huddle for warmth in winter?",
            "Relate agile sprints to a relay race where baton handoffs represent knowledge transfer.",
            "How does the ecosystem of a coral reef mirror the dependencies inside a complex product system?",
            "Summarize what human leaders can learn about resilience from migrating birds."
        ],
    },
    {
        "key": "math_reasoning",
        "label": "示例5：数学推理",
        "description": "英文算术推理问题，观察模型对数字与逻辑词的关注。",
        "prompts": [
            "Mia has 8 apples and gives 2 apples to each of her three friends. How many apples does she have left?",
            "A bakery sold 45 tickets on Friday and twice as many on Saturday. How many tickets were sold during the weekend?",
            "A train travels 120 miles in 3 hours. What is its average speed per hour?",
            "James had 250 dollars, spent 37 on lunch and 45 on books. How much money remains?",
            "A recipe needs 3 cups of flour per batch. How much flour is required for 5 batches?",
            "Lena bikes 15 km to school and the same distance home. How far does she ride in 4 days of classes?",
            "Two numbers add to 48 and differ by 12. What are the two numbers?",
            "A factory produces 1,200 screws a day. How many screws in 6.5 days?",
            "A bookshelf has 5 equally spaced shelves and is 2 meters tall. How far apart are the shelves?",
            "If a car uses 60 liters of fuel to travel 420 km, how many kilometers per liter does it achieve?"
        ],
    },
]


def _init_session_state() -> None:
    """Initialize Streamlit session state keys with defaults."""

    defaults = {
        "prompt_text": SAMPLE_GROUPS[0]["prompts"][0],
        "artifacts": None,
        "last_artifacts": None,
        "selected_token": None,
        "guide_dismissed": False,
        "topk_warned": False,
        "progress_load_pct": 0,
        "progress_load_text": "准备加载 GPT-2 模型…",
        "progress_infer_pct": 0,
        "progress_infer_text": "等待推理开始…",
        "ui_locked": False,
        "run_pending": False,
        "inference_running": False,
        "queued_settings": None,
        "queued_endpoint": None,
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


def _is_busy() -> bool:
    """Return True when UI should stay disabled (loading or queued)."""

    return bool(
        st.session_state.get("ui_locked", False)
        or st.session_state.get("inference_running", False)
        or st.session_state.get("run_pending", False)
    )


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
        button_text = "#3b82f6"
    else:
        bg_color = "#f7f8fb"
        card_color = "#ffffff"
        text_color = "#1f2937"
        accent = "#4757e6"
        button_text = text_color

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
            .stApp button, .stApp [role="button"] {{
                color: {button_text};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_progress_row() -> Dict[str, DeltaGenerator]:
    """Always display load/inference progress bars using session-backed values."""

    st.markdown("#### 加载与推理进度")
    col_load, col_infer = st.columns(2)
    with col_load:
        st.caption("模型加载状态")
        load_bar = st.progress(
            st.session_state.get("progress_load_pct", 0),
            text=st.session_state.get("progress_load_text", "准备加载 GPT-2 模型..."),
        )
    with col_infer:
        st.caption("推理执行状态")
        infer_bar = st.progress(
            st.session_state.get("progress_infer_pct", 0),
            text=st.session_state.get("progress_infer_text", "等待推理开始..."),
        )
    return {"load": load_bar, "infer": infer_bar}


def _render_sidebar() -> Dict[str, object]:
    """Render controls and return selected configuration."""

    is_generating = _is_busy()
    st.sidebar.header("参数调节")
    theme_choice = st.sidebar.radio(
        "配色模式",
        ["light", "dark"],
        format_func=lambda x: "亮色" if x == "light" else "暗色",
        disabled=is_generating,
    )
    model_size = st.sidebar.selectbox(
        "GPT-2 版本",
        options=list(MODEL_SPECS.keys()),
        format_func=lambda key: MODEL_SPECS[key]["display"],
        index=0,
        disabled=is_generating,
    )
    spec = MODEL_SPECS[model_size]
    model_layers = spec["layers"]
    model_heads = spec["heads"]
    st.sidebar.caption(
        f"{spec['display']} · 参数量 {spec.get('params')} · 层 {spec['layers']} · 头 {spec['heads']} · 上下文 {spec['context']} tokens"
    )
    max_tokens = st.sidebar.slider("生成长度", 0, 300, 120, step=5, disabled=is_generating)
    temperature = st.sidebar.slider("温度", 0.1, 1.0, 0.7, step=0.05, disabled=is_generating)
    top_k = st.sidebar.slider("Top-K", 1, 50, 5, step=1, disabled=is_generating)
    if top_k > 10 and not st.session_state.get("topk_warned"):
        st.sidebar.warning("Top-K 超过 10 会显著降低回答准确度，仅供研究用途。")
        st.session_state["topk_warned"] = True
    attention_layers = st.sidebar.multiselect(
        "注意力层 (可多选)",
        options=list(range(1, model_layers + 1)),
        default=[1, model_layers // 2, model_layers],
        disabled=is_generating,
    )
    attention_heads = st.sidebar.multiselect(
        "注意力头 (可多选)",
        options=list(range(1, model_heads + 1)),
        default=list(range(1, min(12, model_heads) + 1)),
        disabled=is_generating,
    )
    viz_dims = st.sidebar.multiselect(
        "可视化维度",
        options=["注意力权重", "Token 推理流", "语义聚类"],
        default=["注意力权重", "Token 推理流", "语义聚类"],
        disabled=is_generating,
    )
    embed_method = st.sidebar.radio(
        "语义降维方法",
        ["pca", "tsne"],
        format_func=lambda x: x.upper(),
        disabled=is_generating,
    )
    context_limit = st.sidebar.slider(
        "上下文窗口 (tokens)",
        min_value=256,
        max_value=int(spec["context"]),
        value=min(768, int(spec["context"])),
        step=64,
        disabled=is_generating,
    )
    source_choice = st.sidebar.radio(
        "模型下载来源",
        options=["official", "mirror"],
        format_func=lambda key: "Hugging Face 官网" if key == "official" else "镜像站 (hf-mirror.com)",
        disabled=is_generating,
    )
    hf_endpoint: Optional[str] = None
    if source_choice == "mirror":
        default_mirror = st.session_state.get("mirror_endpoint", HF_MIRROR_DEFAULT)
        mirror_input = st.sidebar.text_input(
            "镜像地址",
            value=default_mirror,
            help="示例：https://hf-mirror.com",
            disabled=is_generating,
        )
        resolved_mirror = mirror_input.strip() or HF_MIRROR_DEFAULT
        st.session_state["mirror_endpoint"] = resolved_mirror
        hf_endpoint = resolved_mirror
    else:
        hf_endpoint = None

    with st.sidebar.expander("模型缓存与文件"):
        if st.button("在资源管理器中查看模型缓存", key="open-cache", disabled=is_generating):
            _open_cache_dir(CACHE_DIR)
        if st.button("清理内存中的模型 (释放显存/RAM)", key="clear-cache", disabled=is_generating):
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

    is_generating = _is_busy()
    col_run, col_clear, col_export = st.columns([2, 1, 1])
    run_clicked = col_run.button(
        "生成并可视化",
        type="primary",
        width="stretch",
        disabled=is_generating,
    )
    clear_clicked = col_clear.button(
        "清空所有结果", width="stretch", disabled=is_generating or artifacts is None
    )
    col_export.download_button(
        "导出汇总 (HTML)",
        data=report_bytes,
        file_name="gpt2_visual_report.html",
        mime="text/html",
        width="stretch",
        disabled=is_generating or artifacts is None,
    )

    if clear_clicked:
        st.session_state["artifacts"] = None
        st.session_state["last_artifacts"] = None
        st.session_state["selected_token"] = None
        st.session_state["prompt_text"] = ""
        st.toast("已清空历史结果。")

    if run_clicked and not is_generating:
        st.session_state["queued_settings"] = config["settings"]
        st.session_state["queued_endpoint"] = config["hf_endpoint"]
        # 同步动作：立刻锁定全部按钮，并排队启动推理；rerun 触发后模型下载马上开始
        st.session_state["ui_locked"] = True
        st.session_state["run_pending"] = True
        st.rerun()


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
        st.session_state["run_pending"] = False
        st.session_state["inference_running"] = False
        st.session_state["ui_locked"] = False
        return

    progress_widgets = st.session_state.get("progress_widgets")
    if not progress_widgets:
        progress_widgets = _render_progress_row()
        st.session_state["progress_widgets"] = progress_widgets
    load_bar = progress_widgets["load"]
    infer_bar = progress_widgets["infer"]

    def safe_progress(bar, percent: int, text: str) -> None:
        pct = max(0, min(100, int(percent)))
        try:
            bar.progress(pct, text=text)
        except Exception:
            pass

    def progress_callback(stage: str, percent: int) -> None:
        if stage == "download":
            safe_progress(load_bar, percent, f"模型加载 {percent}%")
            st.session_state["progress_load_pct"] = percent
            st.session_state["progress_load_text"] = f"模型加载 {percent}%"
        elif stage == "inference":
            safe_progress(infer_bar, percent, f"推理进度 {percent}%")
            st.session_state["progress_infer_pct"] = percent
            st.session_state["progress_infer_text"] = f"推理进度 {percent}%"

    try:
        artifacts = run_generation(
            prompt,
            settings,
            hf_endpoint=hf_endpoint,
            progress_callback=progress_callback,
        )
        safe_progress(infer_bar, 100, "推理进度 100%")
        st.session_state["progress_infer_pct"] = 100
        st.session_state["progress_infer_text"] = "推理进度 100%"
        st.session_state["artifacts"] = artifacts
        st.session_state["last_artifacts"] = artifacts
        st.session_state["selected_token"] = None
        safe_progress(load_bar, 100, "模型加载完成")
        safe_progress(infer_bar, 100, "推理完成")
        st.session_state["progress_load_pct"] = 100
        st.session_state["progress_load_text"] = "模型加载完成"
        st.session_state["progress_infer_pct"] = 100
        st.session_state["progress_infer_text"] = "推理完成"
        st.success("生成和特征抽取完成。")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"生成失败：{exc}")
    finally:
        try:
            load_bar.empty()
            infer_bar.empty()
        except Exception:
            pass


def _render_prompt_area() -> None:
    """Render text input and sample chips."""

    st.subheader("输入区")
    st.caption("输入任意文本，或使用下方示例按钮。⚠️ GPT-2 以英文语料为主，建议优先输入英文内容。")

    is_generating = _is_busy()
    button_cols = st.columns(len(SAMPLE_GROUPS))
    for col, group in zip(button_cols, SAMPLE_GROUPS):
        if col.button(
            group["label"],
            key=f"sample-btn-{group['key']}",
            help=group["description"],
            width="stretch",
            disabled=is_generating,
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
    if artifacts is None:
        artifacts = st.session_state.get("last_artifacts")
    if not artifacts:
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
    st.session_state["progress_widgets"] = _render_progress_row()
    if st.session_state.get("run_pending"):
        st.session_state["ui_locked"] = True
        st.session_state["inference_running"] = True
        st.session_state["queued_settings"] = st.session_state.get("queued_settings")
        st.session_state["queued_endpoint"] = st.session_state.get("queued_endpoint")
        st.session_state["run_pending"] = False
        st.session_state["inference_triggered"] = True
        st.rerun()

    if st.session_state.get("inference_triggered"):
        st.session_state["inference_triggered"] = False
        settings = st.session_state.pop("queued_settings", None)
        endpoint = st.session_state.pop("queued_endpoint", None)
        _run_inference(settings, endpoint)
        st.session_state["inference_running"] = False
        st.session_state["ui_locked"] = False
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
