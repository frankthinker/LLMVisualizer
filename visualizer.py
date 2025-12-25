"""Plotly-based visual helpers for GPT-2 introspection."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .gpt2_loader import GenerationArtifacts

THEME_TEMPLATE = {"light": "plotly_white", "dark": "plotly_dark"}
LOGIC_KEYWORDS = {
    "因为",
    "所以",
    "如果",
    "则",
    "not",
    "and",
    "or",
    "then",
    "therefore",
    "因此",
    "但是",
}


def categorize_token(token: str) -> str:
    """Rudimentary token classifier for semantic coloring."""

    stripped = token.replace("▁", "").strip().lower()
    if not stripped:
        return "空白"
    if stripped.replace(".", "", 1).isdigit():
        return "数字"
    if stripped in LOGIC_KEYWORDS:
        return "逻辑词"
    if any(char.isdigit() for char in stripped):
        return "数值符号"
    if stripped.endswith("ing") or stripped.endswith("ed"):
        return "动词"
    if stripped in {"the", "a", "an"}:
        return "冠词"
    if stripped in {"猫", "狗", "人", "苹果", "学生"}:
        return "名词"
    if stripped in {"?", "!", ",", ".", "。", "，"}:
        return "标点"
    return "其他"


def build_token_dataframe(artifacts: GenerationArtifacts) -> pd.DataFrame:
    """Merge tokens with statistics for tabular inspection."""

    gen_positions = {metric.position: metric for metric in artifacts.token_metrics}
    generated_start = len(artifacts.token_ids) - len(artifacts.token_metrics)
    rows: List[Dict[str, object]] = []

    for idx, token in enumerate(artifacts.tokens):
        metric = gen_positions.get(idx)
        rows.append(
            {
                "序号": idx,
                "Token": token,
                "阶段": "输入" if idx < generated_start else "生成",
                "概率": round(metric.probability, 4) if metric else None,
                "排名": metric.rank if metric else None,
                "困惑度": round(metric.perplexity, 2) if metric else None,
                "语义类别": categorize_token(token),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_attention(attention_tensor: np.ndarray, heads: Sequence[int]) -> np.ndarray:
    """Average attention for the selected heads (1-indexed)."""

    if attention_tensor.ndim != 3:
        raise ValueError("Attention tensor must be (heads, seq, seq)")
    seq_len = attention_tensor.shape[-1]
    valid_heads = [head - 1 for head in heads if 0 < head <= attention_tensor.shape[0]]
    if not valid_heads:
        valid_heads = list(range(min(8, attention_tensor.shape[0])))
    return attention_tensor[valid_heads].mean(axis=0).reshape(seq_len, seq_len)


def build_attention_figure(
    artifacts: GenerationArtifacts,
    layers: Sequence[int],
    heads: Sequence[int],
    theme: str = "light",
    selected_token: Optional[int] = None,
) -> Tuple[go.Figure, List[str]]:
    """Create multi-panel heatmaps that compare attention across layers."""

    available_layers = sorted(artifacts.attentions.keys())
    filtered_layers = [layer for layer in layers if layer in artifacts.attentions]
    if not filtered_layers:
        filtered_layers = [available_layers[0], available_layers[len(available_layers) // 2], available_layers[-1]]
    filtered_layers = filtered_layers[:3]

    fig = make_subplots(
        rows=1,
        cols=len(filtered_layers),
        subplot_titles=[f"Layer {layer}" for layer in filtered_layers],
        horizontal_spacing=0.05,
        shared_yaxes=True,
    )

    tokens = artifacts.tokens
    summaries: List[str] = []

    for col, layer in enumerate(filtered_layers, start=1):
        attention = artifacts.attentions[layer]
        aggregated = _aggregate_attention(attention, heads)
        heatmap = go.Heatmap(
            z=aggregated,
            x=list(range(len(tokens))),
            y=list(range(len(tokens))),
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="权重" if col == len(filtered_layers) else None),
            customdata=np.array(tokens),
            hovertemplate="源 Token %{y}<br>目标 Token %{x}<br>权重 %{z:.3f}<extra>Layer {layer}</extra>",
        )
        fig.add_trace(heatmap, row=1, col=col)

        max_focus = tokens[np.argmax(aggregated.mean(axis=1))]
        summaries.append(
            f"Layer {layer}: 高亮关注 `{max_focus}` 与相邻 Token，反映{'底层语法' if layer <= 4 else '全局语义' if layer >= available_layers[-1] - 2 else '语义耦合'}。"
        )

        if selected_token is not None and 0 <= selected_token < len(tokens):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(tokens))),
                    y=[selected_token] * len(tokens),
                    mode="markers",
                    marker=dict(color="gold", size=6),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[selected_token] * len(tokens),
                    y=list(range(len(tokens))),
                    mode="markers",
                    marker=dict(color="gold", size=6),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    template = THEME_TEMPLATE.get(theme, "plotly_white")
    fig.update_layout(
        template=template,
        title="注意力权重层级对比",
        margin=dict(t=80, b=50, l=30, r=30),
    )
    fig.update_xaxes(title_text="目标 Token 索引")
    fig.update_yaxes(title_text="源 Token 索引")
    return fig, summaries


def build_token_flow_chart(
    artifacts: GenerationArtifacts,
    theme: str = "light",
) -> Tuple[go.Figure, pd.DataFrame]:
    """Visualize per-token probability and perplexity dynamics."""

    if not artifacts.token_metrics:
        raise ValueError("当前生成长度太短，缺少 Token 度量数据。")
    df = pd.DataFrame(
        [
            {
                "step": idx + 1,
                "position": metric.position,
                "token": metric.token,
                "probability": metric.probability,
                "perplexity": metric.perplexity,
                "rank": metric.rank,
            }
            for idx, metric in enumerate(artifacts.token_metrics)
        ]
    )

    colors = np.where(df["probability"] > 0.5, "#d64045", np.where(df["probability"] > 0.2, "#f4d35e", "#00a6a6"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    prob_trace = go.Scatter(
        x=df["step"],
        y=df["probability"],
        mode="lines+markers",
        marker=dict(color=colors, size=10),
        line=dict(color="#5c4b51"),
        name="生成概率",
        hovertemplate="Step %{x}<br>Token %{text}<br>Prob %{y:.3f}",
        text=df["token"],
    )
    perplex_trace = go.Scatter(
        x=df["step"],
        y=df["perplexity"],
        mode="lines+markers",
        line=dict(color="#247ba0", dash="dot"),
        name="困惑度",
        hovertemplate="Step %{x}<br>Perplexity %{y:.2f}",
    )
    fig.add_trace(prob_trace, secondary_y=False)
    fig.add_trace(perplex_trace, secondary_y=True)

    frames = []
    for idx in range(len(df)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=df["step"][: idx + 1], y=df["probability"][: idx + 1]),
                    go.Scatter(x=df["step"][: idx + 1], y=df["perplexity"][: idx + 1]),
                ],
                name=f"Step {idx + 1}",
            )
        )

    fig.frames = frames
    steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 400, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]

    fig.update_layout(
        template=THEME_TEMPLATE.get(theme, "plotly_white"),
        title="Token 推理时序流",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "播放", "method": "animate", "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}]},
                    {"label": "暂停", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[{"steps": steps, "currentvalue": {"prefix": "步数: "}}],
        margin=dict(t=60, b=50, l=40, r=60),
    )
    fig.update_xaxes(title_text="生成步数")
    fig.update_yaxes(title_text="概率", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="困惑度", secondary_y=True)
    return fig, df


def _prepare_layer_set(hidden_states: Dict[int, np.ndarray], layers: Sequence[int]) -> List[int]:
    available = sorted(hidden_states.keys())
    if not layers:
        return [available[0], available[len(available) // 2], available[-1]]
    deduped = [layer for layer in layers if layer in hidden_states]
    if not deduped:
        return [available[0], available[len(available) // 2], available[-1]]
    return deduped[:3]


def build_semantic_cluster_figure(
    artifacts: GenerationArtifacts,
    layers: Sequence[int],
    method: str = "pca",
    theme: str = "light",
    max_tokens: int = 200,
) -> Tuple[go.Figure, pd.DataFrame]:
    """Project token embeddings into 2D and color by semantic type."""

    selected_layers = _prepare_layer_set(artifacts.hidden_states, layers)
    tokens = artifacts.tokens
    indices = np.linspace(0, len(tokens) - 1, num=min(len(tokens), max_tokens), dtype=int)

    rows: List[pd.DataFrame] = []
    for layer in selected_layers:
        embedding = artifacts.hidden_states[layer]
        sampled = embedding[indices]
        reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2, init="pca", learning_rate="auto")
        reduced = reducer.fit_transform(sampled)
        layer_df = pd.DataFrame(
            {
                "x": reduced[:, 0],
                "y": reduced[:, 1],
                "token": [tokens[idx] for idx in indices],
                "layer": f"Layer {layer}",
                "语义类别": [categorize_token(tokens[idx]) for idx in indices],
            }
        )
        rows.append(layer_df)

    result_df = pd.concat(rows, ignore_index=True)
    fig = px.scatter(
        result_df,
        x="x",
        y="y",
        color="语义类别",
        facet_col="layer",
        hover_data={"token": True, "layer": False},
        template=THEME_TEMPLATE.get(theme, "plotly_white"),
    )
    fig.update_layout(title="语义空间聚类", legend_title="类别", margin=dict(t=60, b=40, l=40, r=40))
    return fig, result_df


def figure_to_html_bytes(fig: go.Figure) -> bytes:
    """Serialize Plotly figure into standalone HTML."""

    return go.Figure(fig).to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def figure_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 600) -> bytes:
    """Convert Plotly figure to PNG using Kaleido if available."""

    return go.Figure(fig).to_image(format="png", width=width, height=height)
