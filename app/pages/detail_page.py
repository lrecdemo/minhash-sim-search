import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
from typing import List
import networkx as nx
from typing import Tuple


def create_plotly_network(G: nx.Graph, layout_type: str = "spring", theme: str = "light") -> go.Figure:
    if layout_type == "spring":
        k_value = max(1, len(G.nodes()) ** 0.5 / 10)
        iterations = min(100, max(30, len(G.nodes()) * 2))
        pos = nx.spring_layout(G, k=k_value, iterations=iterations, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        if len(G.nodes()) > 1 and len(G.nodes()) <= 100:
            pos = nx.kamada_kawai_layout(G)
        else:
            k_value = max(1, len(G.nodes()) ** 0.5 / 10)
            pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    node_x, node_y = [], []
    node_text, node_info, node_colors, node_sizes = [], [], [], []

    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0

    base_size = max(8, min(20, 100 / len(G.nodes()) ** 0.3)) if len(G.nodes()) > 0 else 15
    max_additional_size = max(5, min(25, 150 / len(G.nodes()) ** 0.3)) if len(G.nodes()) > 0 else 20

    if theme == "dark":
        bg_color = 'rgba(20, 20, 30, 0.9)'
        font_color = 'white'
        edge_color = 'rgba(128, 128, 128, 0.4)'
        annotation_color = 'gray'
        marker_line_color = 'white'
    else:
        bg_color = 'rgba(255, 255, 255, 1)'
        font_color = 'black'
        edge_color = 'rgba(100, 100, 100, 0.3)'
        annotation_color = 'darkgray'
        marker_line_color = 'black'


    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_data = G.nodes[node]
        confidence = node_data.get('confidence', 0)
        text_preview = node_data.get('text', 'No text')
        batch_id = node_data.get('batch_id', 'Unknown')
        if len(G.nodes()) > 50:
            node_text.append(f"{node}")
        else:
            node_text.append(f"#{node}")

        connections = degrees.get(node, 0)
        node_info.append(f"Document #{node}<br>"
                         f"Confidence: {confidence:.1%}<br>"
                         f"Connections: {connections}<br>"
                         f"Batch: {batch_id}<br>"
                         f"Preview: {text_preview}")

        if theme == "dark":
            if confidence >= 0.8:
                node_colors.append('rgba(46, 125, 50, 0.8)')
            elif confidence >= 0.6:
                node_colors.append('rgba(255, 193, 7, 0.8)')
            else:
                node_colors.append('rgba(244, 67, 54, 0.8)')
        else:
            if confidence >= 0.75:
                node_colors.append('rgba(76, 175, 80, 0.9)')
            elif confidence >= 0.6:
                node_colors.append('rgba(255, 152, 0, 0.9)')
            else:
                node_colors.append('rgba(244, 67, 54, 0.9)')

        if max_degree > min_degree:
            normalized_degree = (connections - min_degree) / (max_degree - min_degree)
        else:
            normalized_degree = 0.5
        node_size = base_size + (normalized_degree * max_additional_size)
        node_sizes.append(node_size)

    edge_x, edge_y = [], []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G.edges[edge].get('weight', 0)
        edge_weights.append(weight)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color=edge_color),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color=marker_line_color),
            opacity=0.8
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(
            size=max(8, min(12, 60 / len(G.nodes()) ** 0.3)) if len(G.nodes()) > 0 else 10,
            color=font_color,
            family="Arial Black"
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=node_info,
        showlegend=False
    ))

    fig.update_layout(
        title="",
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text=f"Node size = connections | Color = confidence | {len(G.nodes())} nodes, {len(G.edges())} edges",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color=annotation_color, size=10)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color),
        height=600
    )

    return fig
def create_document_graph_from_clustering(docs_df: pd.DataFrame,
                                          clustering_similarities: List[Tuple[int, int, float]],
                                          similarity_threshold: float = 0.3):
    G = nx.Graph()
    for idx, doc in docs_df.iterrows():
        doc_id = doc.get('id', doc['original_index'])
        text_preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']

        G.add_node(doc_id,
                   text=text_preview,
                   full_text=doc['text'],
                   confidence=doc['certainty'],
                   original_index=doc['original_index'],
                   batch_id=doc.get('batch_id', 'Unknown'))
    index_to_id = {doc['original_index']: doc.get('id', doc['original_index'])
                   for _, doc in docs_df.iterrows()}
    edges_added = 0
    max_edges = min(len(clustering_similarities), 1000)
    sorted_similarities = sorted(clustering_similarities, key=lambda x: x[2], reverse=True)

    for doc1_idx, doc2_idx, similarity in sorted_similarities:
        if edges_added >= max_edges:
            break

        if similarity >= similarity_threshold:
            if doc1_idx in index_to_id and doc2_idx in index_to_id:
                doc_id1 = index_to_id[doc1_idx]
                doc_id2 = index_to_id[doc2_idx]
                if doc_id1 != doc_id2:
                    G.add_edge(doc_id1, doc_id2, weight=similarity)
                    edges_added += 1

    return G


def display_cluster_details(df: pd.DataFrame, cluster_id: int):
    cluster_docs = df[df['cluster_id'] == cluster_id].copy()

    in_search_context = st.session_state.get('in_search_context', False)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# Cluster {cluster_id} - All Documents")
        batch_info = ""
        if 'batch_id' in cluster_docs.columns:
            batches = cluster_docs['batch_id'].nunique()
        st.markdown(
            f"**{len(cluster_docs)} documents • {cluster_docs['certainty'].mean():.0%} avg confidence**")
    with col2:
        if not in_search_context:
            if st.button("← Back to Overview", type="secondary", key=f"back_to_overview_{cluster_id}"):
                st.session_state.view_mode = 'overview'
                st.rerun()

    view_mode = st.radio(
        "",
        ["List View", "Graph View"],
        horizontal=True,
        key=f"view_mode_cluster_{cluster_id}"
    )

    if view_mode == "Graph View":
        if len(cluster_docs) > 200:
            st.warning(
                f"Large cluster ({len(cluster_docs)} documents). Graph may be slow to render. Consider using List View or increasing the similarity threshold.")

        graph_col1, graph_col2, graph_col3 = st.columns([2, 2, 1])

        with graph_col1:
            default_threshold = 0.5 if len(cluster_docs) > 100 else 0.3
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=0.9,
                value=default_threshold,
                step=0.05,
                help="Higher values show fewer, stronger connections. Recommended for large clusters.",
                key=f"similarity_threshold_{cluster_id}"
            )

        with graph_col2:
            if len(cluster_docs) > 100:
                default_layout = "circular"
                help_text = "Circular layout recommended for large clusters"
            else:
                default_layout = "spring"
                help_text = "Spring layout works well for smaller clusters"

            layout_type = st.selectbox(
                "Layout Algorithm",
                ["spring", "circular", "kamada_kawai"],
                index=["spring", "circular", "kamada_kawai"].index(default_layout),
                format_func=lambda x: {
                    "spring": "Force-directed (best for <100 nodes)",
                    "circular": "Circular (good for large clusters)",
                    "kamada_kawai": "Kamada-Kawai (best for <50 nodes)"
                }[x],
                help=help_text,
                key=f"layout_type_{cluster_id}"
            )
        with graph_col3:
            theme = st.selectbox(
                "Theme",
                ["dark", "light"],
                format_func=lambda x: x.capitalize(),
                key=f"theme_{cluster_id}"
            )

        if len(cluster_docs) > 1:
            try:
                with st.spinner("Creating network graph..."):
                    cluster_doc_indices = set(cluster_docs['original_index'].tolist())

                    all_similarities = st.session_state.get('clustering_similarities', [])
                    cluster_similarities = [
                        (doc1, doc2, sim) for doc1, doc2, sim in all_similarities
                        if doc1 in cluster_doc_indices and doc2 in cluster_doc_indices
                    ]

                    G = create_document_graph_from_clustering(cluster_docs, cluster_similarities,
                                                              similarity_threshold)
                    if len(G.edges()) == 0:
                        st.warning(
                            f"No connections found at {similarity_threshold} threshold. Try lowering the threshold.")
                    else:
                        fig = create_plotly_network(G, layout_type,theme)
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Network Statistics")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    st.metric("Nodes", len(G.nodes()))
                with stats_col2:
                    st.metric("Edges", len(G.edges()))
                with stats_col3:
                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
                    st.metric("Avg Connections", f"{avg_degree:.1f}")
                with stats_col4:
                    density = nx.density(G) if len(G.nodes()) > 1 else 0
                    st.metric("Network Density", f"{density:.3f}")

            except Exception as e:
                st.error(f"Error creating graph visualization: {str(e)}")
                st.info("Falling back to list view...")
                view_mode = "List View"
        else:
            st.info("Graph view requires at least 2 documents. Showing list view instead.")
            view_mode = "List View"

    if view_mode == "List View":
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input(
                "Search within this cluster:",
                placeholder="Enter words or phrases to find specific documents...",
                key=f"search_cluster_{cluster_id}"
            )
        with search_col2:
            sort_option = st.selectbox(
                "Sort by:",
                options=["confidence", "original", "length"],
                format_func=lambda x:
                {"confidence": "Confidence", "original": "Original Order", "length": "Text Length"}[x],
                key=f"sort_cluster_{cluster_id}"
            )

        filtered_docs = cluster_docs.copy()
        search_terms = []
        if search_query.strip():
            search_terms = [term.strip() for term in search_query.split() if term.strip()]
            mask = cluster_docs['text'].str.contains(search_query.strip(), case=False, na=False)
            search_filtered = cluster_docs[mask]
            if len(search_filtered) == 0:
                st.warning(f"No documents found containing '{search_query}' in this cluster.")
            else:
                st.success(f"Found {len(search_filtered)} documents matching '{search_query}'")
                filtered_docs = search_filtered

        if sort_option == "confidence":
            filtered_docs = filtered_docs.sort_values('certainty', ascending=False)
        elif sort_option == "original":
            filtered_docs = filtered_docs.sort_values('original_index')
        elif sort_option == "length":
            filtered_docs['text_length'] = filtered_docs['text'].str.len()
            filtered_docs = filtered_docs.sort_values('text_length', ascending=False)

        docs_per_page = 50 if len(filtered_docs) > 100 else len(filtered_docs)

        page_key = f"current_page_{cluster_id}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1

        if len(filtered_docs) > 50:
            total_pages = (len(filtered_docs) - 1) // docs_per_page + 1
            current_page = st.session_state[page_key]

            if current_page > total_pages:
                st.session_state[page_key] = 1
                current_page = 1

            start_idx = (current_page - 1) * docs_per_page
            end_idx = min(start_idx + docs_per_page, len(filtered_docs))
            page_docs = filtered_docs.iloc[start_idx:end_idx]
        else:
            page_docs = filtered_docs

        st.markdown(f"### Documents ({len(page_docs)} of {len(filtered_docs)} shown)")
        for i, (_, doc) in enumerate(page_docs.iterrows()):
            expand_default = (i < 5) and (len(page_docs) <= 20)
            doc_id = doc.get('id', doc['original_index']) if 'id' in doc else doc['original_index']
            with st.expander(f"Document #{doc_id}",
                             expanded=expand_default):
                st.markdown(f"**{doc_id}:** {doc['text']}")
                cols = st.columns(4) if 'batch_id' in doc else st.columns(3)
                with cols[0]:
                    st.caption(f"Original Position: #{doc['original_index']}")
                with cols[1]:
                    st.caption(f"Confidence: {doc['certainty']:.1%}")
                with cols[2]:
                    st.caption(f"Text Length: {len(doc['text'])} chars")
                if 'batch_id' in doc and len(cols) > 3:
                    with cols[3]:
                        st.caption(f"Batch: {doc['batch_id']}")

        if len(filtered_docs) > 50:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                new_page = st.selectbox(
                    f"Page (showing {docs_per_page} documents per page):",
                    options=range(1, total_pages + 1),
                    index=current_page - 1,
                    format_func=lambda
                        x: f"Page {x} (Documents {(x - 1) * docs_per_page + 1}-{min(x * docs_per_page, len(filtered_docs))})",
                    key=f"page_selector_{cluster_id}"
                )
                if new_page != current_page:
                    st.session_state[page_key] = new_page
                    st.rerun()