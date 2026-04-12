from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_comparison.streamlit_app import render_architecture_comparison
from compare_three_backbones.src.streamlit_app import render_compare_three_backbones
from semantic_vs_instance.streamlit_app import render_semantic_vs_instance


def main() -> None:
    st.set_page_config(page_title="Image Segmentation Demos", layout="wide")
    st.title("Assignment 2 Image Segmentation")
    st.caption("Single UI to run all 3 demos")

    tab1, tab2, tab3 = st.tabs([
        "Architecture Comparison",
        "Compare Three Backbones",
        "Semantic vs Instance",
    ])

    with tab1:
        render_architecture_comparison(use_sidebar=False, key_prefix="tab_arch")

    with tab2:
        render_compare_three_backbones(use_sidebar=False, key_prefix="tab_ctb")

    with tab3:
        render_semantic_vs_instance(use_sidebar=False, key_prefix="tab_svs")


if __name__ == "__main__":
    main()
