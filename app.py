"""UAL Code Converter — Foundry to Databricks Migration Tool (Streamlit App)."""

import io
import os
import uuid
import zipfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag_engine import KeywordIndex, chunk_file
from import_resolver import resolve_imports, build_dependency_graph
from converter import convert_file, chat_with_code
from validator import validate_conversion

# Load .env if present
load_dotenv(override=True)
config = os.environ

# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UAL Code Converter",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .stApp {background-color: #f8f9fb;}
    .main-header {
        background: linear-gradient(135deg, #0032A0 0%, #1a4fbd 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {color: white; margin: 0; font-size: 1.6rem;}
    .main-header p {color: #a8c4f0; margin: 0.2rem 0 0 0; font-size: 0.85rem;}
    .stat-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        border: 1px solid #e5e7eb; text-align: center;
    }
    .stat-card .number {font-size: 2rem; font-weight: 700; color: #0032A0;}
    .stat-card .label {font-size: 0.8rem; color: #6b7280;}
    .file-row {
        background: white; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 0.8rem 1rem; margin-bottom: 0.5rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 9999px;
        font-size: 0.7rem; font-weight: 600;
    }
    .badge-green {background: #dcfce7; color: #166534;}
    .badge-yellow {background: #fef9c3; color: #854d0e;}
    .badge-red {background: #fee2e2; color: #991b1b;}
    .badge-blue {background: #dbeafe; color: #1e40af;}
    .badge-bronze {background: #fed7aa; color: #9a3412;}
    .badge-silver {background: #e5e7eb; color: #374151;}
    .badge-gold {background: #fef3c7; color: #92400e;}
    .issue-box {
        background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px;
        padding: 0.8rem; margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ────────────────────────────────────────────────────────


def init_state():
    defaults = {
        "projects": {},  # id -> {name, created_at}
        "active_project": None,  # project id
        "files": {},  # project_id -> {file_path -> {id, content, file_name}}
        "indexes": {},  # project_id -> KeywordIndex instance
        "conversions": {},  # project_id -> {conv_id -> {result dict}}
        "api_url": config.get("API_URL", "https://quasarmarket.coforge.com/qag/llmrouter-api/v2/chat/completions"),
        "api_key": config.get("API_KEY", ""),
        "model_name": config.get("MODEL_NAME", "gpt-5-2"),
        "chat_history": [],  # list of {role, content}
        "uploaded_file_keys": set(),  # track already-processed uploads
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ─── Helper Functions ───────────────────────────────────────────────────────────


def get_project_files() -> dict[str, str]:
    """Return {file_path: content} for the active project."""
    pid = st.session_state.active_project
    return {
        fp: info["content"]
        for fp, info in st.session_state.files.get(pid, {}).items()
    }


def get_index() -> KeywordIndex:
    """Get or create the RAG index for the active project."""
    pid = st.session_state.active_project
    if pid not in st.session_state.indexes:
        st.session_state.indexes[pid] = KeywordIndex()
    return st.session_state.indexes[pid]


def add_file_to_project(file_name: str, file_path: str, content: str):
    """Add a file to the active project, chunk it, and index it."""
    pid = st.session_state.active_project
    if pid not in st.session_state.files:
        st.session_state.files[pid] = {}

    file_id = str(uuid.uuid4())
    st.session_state.files[pid][file_path] = {
        "id": file_id,
        "file_name": file_name,
        "content": content,
        "size": len(content),
    }

    # Chunk and index
    chunks = chunk_file(file_id, file_path, content)
    get_index().add_chunks(chunks)

    return len(chunks)


def get_project_conversions() -> dict:
    pid = st.session_state.active_project
    return st.session_state.conversions.get(pid, {})


# ─── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✈️ UAL Code Converter")
    st.caption("Foundry → Databricks Migration")
    st.divider()

    # LLM API Config
    api_url = st.text_input(
        "🌐 API URL",
        value=st.session_state.api_url,
        placeholder="https://quasarmarket.coforge.com/...",
        help="LLM Router endpoint URL",
    )
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url

    api_key = st.text_input(
        "🔑 X-API-KEY",
        value=st.session_state.api_key,
        type="password",
        placeholder="your-api-key",
        help="Your key stays in your browser session only.",
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key

    model_name = st.text_input(
        "🤖 Model Name",
        value=st.session_state.model_name,
        placeholder="gpt-5-2",
        help="Model name as configured in LLM Router",
    )
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name

    if not st.session_state.api_key:
        st.warning("Set X-API-KEY to enable conversion & chat.", icon="⚠️")

    st.divider()

    # Projects
    st.markdown("**Projects**")
    new_name = st.text_input("New project name", placeholder="e.g., UAL-AirOps", label_visibility="collapsed")
    if st.button("➕ Create Project", use_container_width=True, disabled=not new_name.strip()):
        pid = str(uuid.uuid4())
        st.session_state.projects[pid] = {
            "name": new_name.strip(),
            "created_at": datetime.now().isoformat(),
        }
        st.session_state.active_project = pid
        st.rerun()

    for pid, proj in st.session_state.projects.items():
        is_active = st.session_state.active_project == pid
        label = f"{'▶ ' if is_active else ''}{proj['name']}"
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"proj_{pid}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.active_project = pid
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{pid}", help="Delete project"):
                st.session_state.projects.pop(pid, None)
                st.session_state.files.pop(pid, None)
                st.session_state.indexes.pop(pid, None)
                st.session_state.conversions.pop(pid, None)
                if st.session_state.active_project == pid:
                    st.session_state.active_project = None
                st.rerun()

    st.divider()
    st.caption("UDH 3.0 Migration Tool")

# ─── Main Area ──────────────────────────────────────────────────────────────────

if not st.session_state.active_project:
    # Dashboard
    st.markdown(
        '<div class="main-header"><h1>UAL Code Converter</h1>'
        "<p>Migrate Palantir Foundry Python transforms to Databricks notebooks with AI-powered conversion and RAG</p></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    steps = [
        ("📁", "Upload", "Foundry .py files"),
        ("🔍", "RAG Index", "Chunk & analyze"),
        ("🤖", "AI Convert", "LLM Router"),
        ("📦", "Output", "Databricks notebooks"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="number">{icon}</div>'
                f'<div class="label"><b>{title}</b><br>{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.info("👈 Create a project in the sidebar to get started.", icon="ℹ️")
    st.stop()

# ─── Active Project ─────────────────────────────────────────────────────────────

project = st.session_state.projects[st.session_state.active_project]

st.markdown(
    f'<div class="main-header"><h1>{project["name"]}</h1>'
    f"<p>Foundry → Databricks Code Conversion</p></div>",
    unsafe_allow_html=True,
)

tab_arch, tab_files, tab_convert, tab_output, tab_chat = st.tabs(
    ["🏗️ Architecture", "📁 Files", "🔄 Convert", "📦 Output", "💬 Ask AI"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 0: ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════

with tab_arch:
    st.subheader("Standalone RAG Agent: Architecture")
    st.caption("UAL MARS Platform Pattern — Code Conversion Application")

    # Architecture diagram using HTML/CSS matching the reference PNG
    st.markdown(
        """
        <div style="background: white; border-radius: 12px; padding: 2rem; border: 1px solid #e5e7eb; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: flex-start; justify-content: center; gap: 0; position: relative; min-height: 420px;">

                <!-- User -->
                <div style="display: flex; flex-direction: column; align-items: center; margin-top: 50px; min-width: 80px;">
                    <div style="width: 50px; height: 50px; background: #1a1a2e; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 24px;">👤</span>
                    </div>
                    <span style="font-size: 11px; color: #6b7280; margin-top: 6px; font-weight: 600;">User</span>
                </div>

                <!-- Arrow -->
                <div style="display: flex; align-items: center; margin-top: 65px; padding: 0 8px;">
                    <span style="font-size: 22px; color: #374151;">→</span>
                </div>

                <!-- Web Application -->
                <div style="display: flex; flex-direction: column; align-items: center; min-width: 160px;">
                    <div style="background: linear-gradient(135deg, #0032A0, #1a4fbd); color: white; padding: 28px 24px; border-radius: 10px; text-align: center; width: 160px; margin-top: 30px; box-shadow: 0 4px 12px rgba(0,50,160,0.3);">
                        <div style="font-weight: 700; font-size: 14px;">Web Application</div>
                        <div style="font-size: 10px; opacity: 0.8; margin-top: 4px;">Streamlit UI</div>
                    </div>
                    <div style="font-size: 10px; color: #6b7280; margin-top: 6px;">Upload, Convert, Chat</div>
                </div>

                <!-- Arrows: User query / Generate answer -->
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 30px; padding: 0 6px; min-width: 120px;">
                    <div style="text-align: center; margin-bottom: 4px;">
                        <span style="font-size: 10px; color: #374151; font-weight: 500;">User query</span>
                    </div>
                    <div style="font-size: 20px; color: #0032A0; letter-spacing: -2px;">→→→→→</div>
                    <div style="font-size: 20px; color: #0032A0; letter-spacing: -2px; margin-top: 12px;">←←←←←</div>
                    <div style="text-align: center; margin-top: 4px;">
                        <span style="font-size: 10px; color: #374151; font-weight: 500;">Generate answer</span>
                    </div>
                </div>

                <!-- GenAI/Agent Application -->
                <div style="display: flex; flex-direction: column; align-items: center; min-width: 180px;">
                    <div style="background: linear-gradient(135deg, #0032A0, #1a4fbd); color: white; padding: 28px 24px; border-radius: 10px; text-align: center; width: 180px; margin-top: 30px; box-shadow: 0 4px 12px rgba(0,50,160,0.3);">
                        <div style="font-weight: 700; font-size: 14px;">GenAI / Agent</div>
                        <div style="font-weight: 700; font-size: 14px;">Application</div>
                        <div style="font-size: 10px; opacity: 0.8; margin-top: 4px;">LLM API</div>
                    </div>
                    <div style="font-size: 10px; color: #6b7280; margin-top: 6px;">Conversion + RAG Chat</div>
                </div>

                <!-- Arrows: MCP call / response -->
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 30px; padding: 0 6px; min-width: 100px;">
                    <div style="text-align: center; margin-bottom: 4px;">
                        <span style="font-size: 10px; color: #374151; font-weight: 500;">MCP call</span>
                    </div>
                    <div style="font-size: 20px; color: #0032A0; letter-spacing: -2px;">→→→→</div>
                    <div style="font-size: 20px; color: #0032A0; letter-spacing: -2px; margin-top: 12px;">←←←←</div>
                    <div style="text-align: center; margin-top: 4px;">
                        <span style="font-size: 10px; color: #374151; font-weight: 500;">Response</span>
                    </div>
                </div>

                <!-- MCP Tools icon -->
                <div style="display: flex; flex-direction: column; align-items: center; margin-top: 40px; min-width: 80px;">
                    <div style="font-size: 40px;">🔗</div>
                    <span style="font-size: 10px; color: #6b7280; font-weight: 600; margin-top: 4px;">MCP Tools</span>
                </div>
            </div>

            <!-- Bottom row: Batch Application + Vector Store -->
            <div style="display: flex; align-items: center; justify-content: center; gap: 0; margin-top: -30px;">
                <div style="min-width: 240px;"></div>
                <!-- Spacer for alignment -->

                <!-- Batch Application -->
                <div style="display: flex; flex-direction: column; align-items: center; min-width: 180px;">
                    <div style="background: linear-gradient(135deg, #0032A0, #1a4fbd); color: white; padding: 28px 24px; border-radius: 10px; text-align: center; width: 180px; box-shadow: 0 4px 12px rgba(0,50,160,0.3);">
                        <div style="font-weight: 700; font-size: 14px;">Batch Application</div>
                        <div style="font-size: 10px; opacity: 0.8; margin-top: 4px;">File Processor</div>
                    </div>
                    <div style="font-size: 10px; color: #6b7280; margin-top: 6px;">Chunking + Indexing</div>
                </div>

                <!-- Arrow: Ingestion -->
                <div style="display: flex; flex-direction: column; align-items: center; padding: 0 10px;">
                    <div style="font-size: 20px; color: #0032A0; letter-spacing: -2px;">→→→→→</div>
                    <span style="font-size: 10px; color: #374151; font-weight: 500;">Ingestion</span>
                </div>

                <!-- Vector Store -->
                <div style="display: flex; flex-direction: column; align-items: center; min-width: 140px;">
                    <div style="font-size: 40px;">🗄️</div>
                    <div style="font-weight: 700; font-size: 13px; color: #1a1a2e;">Milvus Vector Store</div>
                    <div style="font-size: 10px; color: #6b7280;">RAG Index</div>
                </div>

                <!-- Arrow: Retrieve (going up to GenAI) -->
                <div style="display: flex; flex-direction: column; align-items: center; padding: 0 10px;">
                    <span style="font-size: 10px; color: #374151; font-weight: 500;">Retrieve</span>
                    <div style="font-size: 20px; color: #0032A0;">↑</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Architecture Details
    st.divider()
    st.subheader("End-to-End Data Flow")

    flow_data = [
        ("1. Data Sources", "SFTP, Teradata, Oracle", "Raw operational data from legacy source systems"),
        ("2. Ingestion", "Apache Airflow", "Foundry is being replaced. Airflow stays as orchestrator for DAGs"),
        ("3. Compute", "Databricks (PySpark)", "Runs all transformation logic — the new engine replacing Foundry"),
        ("4. Bronze Layer", "AWS S3 + Delta Lake", "Raw data landed exactly as-is from source. No transformations"),
        ("5. Silver Layer", "AWS S3 + Delta Lake", "Cleaned, deduplicated, validated, standardised data"),
        ("6. Gold Layer", "AWS S3 + Delta Lake", "Business-ready aggregated data, optimised for analytics"),
        ("7. Serving", "AWS Redshift + ICON", "Redshift for SQL analytics. ICON for governance and metadata"),
        ("8. AI / ML", "MARS (AWS Bedrock)", "UAL's AI platform — consumes Gold data for production AI models"),
        ("9. Consumers", "BI Tools, Employees", "Dashboards, internal tools, customer-facing AI notifications"),
    ]

    for step, tech, desc in flow_data:
        col1, col2, col3 = st.columns([2, 2, 4])
        col1.markdown(f"**{step}**")
        col2.code(tech, language=None)
        col3.caption(desc)

    st.divider()
    st.subheader("Technology Stack")

    tech_stack = {
        "Palantir Foundry": "Proprietary platform being replaced — contains existing Python transform code to migrate",
        "Apache Airflow": "Open-source workflow orchestration — defines DAGs for pipeline scheduling. Stays in new architecture",
        "Databricks": "Cloud data + AI platform on Apache Spark — UAL's new compute engine on AWS",
        "AWS S3": "Object storage for all data files (Bronze/Silver/Gold). The lakehouse foundation",
        "Delta Lake": "Storage layer adding ACID transactions, time travel, schema enforcement on S3",
        "AWS Redshift": "Cloud data warehouse for fast SQL analytics. BI tools connect here",
        "ICON Framework": "UAL's governance framework — Ingest, Catalog, Optimize, Notify",
        "MARS": "UAL's AI/ML platform on AWS Bedrock — runs production LLM models",
        "LLM Router": "Coforge QuasarMarket AI gateway — powers code conversion and RAG chat in this tool",
    }

    for tech, desc in tech_stack.items():
        st.markdown(f"**{tech}** — {desc}")

    st.divider()
    st.subheader("Medallion Architecture")

    col_b, col_s, col_g = st.columns(3)
    with col_b:
        st.markdown(
            '<div class="stat-card">'
            '<div class="number" style="color: #9a3412;">🟤</div>'
            '<div class="label"><b>Bronze — Raw Zone</b><br>'
            'Data lands exactly as it came from source. No changes, no cleaning. Audit trail.</div></div>',
            unsafe_allow_html=True,
        )
    with col_s:
        st.markdown(
            '<div class="stat-card">'
            '<div class="number" style="color: #374151;">⚪</div>'
            '<div class="label"><b>Silver — Cleaned Zone</b><br>'
            'Cleaned, validated, standardised. Duplicates removed, nulls handled, types fixed.</div></div>',
            unsafe_allow_html=True,
        )
    with col_g:
        st.markdown(
            '<div class="stat-card">'
            '<div class="number" style="color: #92400e;">🟡</div>'
            '<div class="label"><b>Gold — Business Zone</b><br>'
            'Aggregated for business use cases. Pre-joined, pre-calculated. Ready for BI + MARS.</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Foundry vs Databricks")

    comparison = {
        "Made by": ("Palantir Technologies", "Databricks Inc."),
        "Who uses it": ("Business analysts + non-technical teams", "Data engineers + scientists"),
        "How you work": ("Click, drag & drop — visual UI", "Write Python / SQL code"),
        "Cost": ("Expensive fixed enterprise license", "Pay per use — consumption based"),
        "Code portability": ("Hard to extract — proprietary format", "Standard Python / SQL — portable"),
        "AWS integration": ("Works but external to ecosystem", "Native — built for AWS"),
        "UAL status": ("Being replaced (this project)", "New platform (this project)"),
    }

    col_h1, col_h2, col_h3 = st.columns([2, 3, 3])
    col_h1.markdown("**Aspect**")
    col_h2.markdown("**Palantir Foundry**")
    col_h3.markdown("**Databricks**")

    for aspect, (foundry, databricks) in comparison.items():
        c1, c2, c3 = st.columns([2, 3, 3])
        c1.markdown(f"**{aspect}**")
        c2.caption(foundry)
        c3.caption(databricks)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: FILES
# ════════════════════════════════════════════════════════════════════════════════

with tab_files:
    project_files = get_project_files()
    stats = get_index().get_stats()
    convs = get_project_conversions()
    done_count = sum(1 for c in convs.values() if c["status"] == "done")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(project_files))
    c2.metric("RAG Chunks", stats["chunk_count"])
    c3.metric("Converted", done_count)
    c4.metric("Pending", len(project_files) - done_count)

    st.subheader("Upload Files")
    uploaded = st.file_uploader(
        "Drop .py or .zip files",
        type=["py", "zip"],
        accept_multiple_files=True,
        help="Upload Foundry Python transform files. ZIP archives are auto-extracted.",
    )

    if uploaded:
        # Build unique keys for each uploaded file to avoid re-processing on rerun
        new_files = []
        for f in uploaded:
            file_key = f"{f.name}_{f.size}"
            if file_key not in st.session_state.uploaded_file_keys:
                new_files.append((f, file_key))

        if new_files:
            import os
            total_added = 0
            for f, file_key in new_files:
                if f.name.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                            for entry in zf.namelist():
                                if not entry.endswith(".py"):
                                    continue
                                if "__pycache__" in entry:
                                    continue
                                if any(p.startswith(".") for p in entry.split("/")):
                                    continue
                                content = zf.read(entry).decode("utf-8")
                                fname = os.path.basename(entry)
                                add_file_to_project(fname, entry, content)
                                total_added += 1
                    except Exception as e:
                        st.error(f"Failed to extract {f.name}: {e}")
                elif f.name.endswith(".py"):
                    content = f.read().decode("utf-8")
                    add_file_to_project(f.name, f.name, content)
                    total_added += 1

                st.session_state.uploaded_file_keys.add(file_key)

            if total_added > 0:
                st.success(f"Uploaded and indexed {total_added} file(s)")
                st.rerun()

    # File List
    st.subheader("Uploaded Files")
    if not project_files:
        st.info("No files uploaded yet. Upload .py or .zip files above.")
    else:
        for fp, content in project_files.items():
            info = st.session_state.files[st.session_state.active_project][fp]
            col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
            col1.markdown(f"**{info['file_name']}**")
            col2.caption(fp)
            col3.markdown('<span class="badge badge-green">indexed</span>', unsafe_allow_html=True)
            if col4.button("🗑", key=f"delf_{fp}", help="Remove file"):
                idx = get_index()
                idx.remove_file(info["id"])
                del st.session_state.files[st.session_state.active_project][fp]
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: CONVERT
# ════════════════════════════════════════════════════════════════════════════════

with tab_convert:
    project_files = get_project_files()

    if not project_files:
        st.info("Upload files first in the Files tab.")
    else:
        st.subheader("Select File to Convert")

        file_paths = list(project_files.keys())
        selected_fp = st.selectbox("Choose a file", file_paths, format_func=lambda x: x)

        if selected_fp:
            # Show file preview
            with st.expander("📄 Preview source code", expanded=False):
                st.code(project_files[selected_fp], language="python", line_numbers=True)

            # Resolve imports
            imports = resolve_imports(selected_fp, project_files)
            if imports:
                st.info(f"🔗 **{len(imports)} imported file(s)** will be included as context:")
                for imp in imports:
                    st.markdown(f"- `{imp}`")
            else:
                st.caption("No internal imports detected.")

            # Target layer
            col1, col2 = st.columns([1, 2])
            with col1:
                target_layer = st.selectbox(
                    "Target Layer",
                    ["bronze", "silver", "gold"],
                    index=1,
                    format_func=lambda x: {"bronze": "🟤 Bronze (Raw)", "silver": "⚪ Silver (Cleaned)", "gold": "🟡 Gold (Business)"}[x],
                )

            # Convert button
            if not st.session_state.api_key:
                st.warning("Set your X-API-KEY in the sidebar to convert.", icon="🔑")
            else:
                if st.button("🚀 Convert with AI", type="primary", use_container_width=True):
                    with st.spinner("Converting with AI... This may take 15–30 seconds."):
                        try:
                            # Prepare imported files context
                            imported_context = [
                                {"path": imp, "content": project_files[imp]}
                                for imp in imports
                            ]

                            converted_code = convert_file(
                                api_url=st.session_state.api_url,
                                api_key=st.session_state.api_key,
                                model=st.session_state.model_name,
                                file_path=selected_fp,
                                file_content=project_files[selected_fp],
                                target_layer=target_layer,
                                imported_files=imported_context,
                            )

                            # Validate
                            issues = validate_conversion(converted_code)

                            # Store result
                            conv_id = str(uuid.uuid4())
                            pid = st.session_state.active_project
                            if pid not in st.session_state.conversions:
                                st.session_state.conversions[pid] = {}

                            st.session_state.conversions[pid][conv_id] = {
                                "id": conv_id,
                                "file_path": selected_fp,
                                "target_layer": target_layer,
                                "original_code": project_files[selected_fp],
                                "converted_code": converted_code,
                                "status": "done",
                                "issues": issues,
                                "imported_files": imports,
                                "converted_at": datetime.now().isoformat(),
                            }

                            st.success(f"Conversion complete! {len(issues)} issue(s) found.")

                        except Exception as e:
                            st.error(f"Conversion failed: {e}")
                            # Store error
                            conv_id = str(uuid.uuid4())
                            pid = st.session_state.active_project
                            if pid not in st.session_state.conversions:
                                st.session_state.conversions[pid] = {}
                            st.session_state.conversions[pid][conv_id] = {
                                "id": conv_id,
                                "file_path": selected_fp,
                                "target_layer": target_layer,
                                "original_code": project_files[selected_fp],
                                "converted_code": "",
                                "status": "error",
                                "issues": [],
                                "imported_files": imports,
                                "error": str(e),
                                "converted_at": datetime.now().isoformat(),
                            }

            # Show latest result for selected file
            convs = get_project_conversions()
            matching = [c for c in convs.values() if c["file_path"] == selected_fp and c["status"] == "done"]
            if matching:
                latest = max(matching, key=lambda c: c["converted_at"])

                st.divider()
                st.subheader("Conversion Result")

                # Validation issues
                if latest["issues"]:
                    st.warning(f"⚠️ {len(latest['issues'])} unconverted API(s) detected:")
                    for issue in latest["issues"]:
                        st.markdown(
                            f'<div class="issue-box"><b>Line {issue.line_number}:</b> '
                            f'<code>{issue.pattern}</code> — <code>{issue.line}</code></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("All Foundry APIs successfully converted!")

                # Side-by-side
                col_orig, col_conv = st.columns(2)
                with col_orig:
                    st.markdown("**Original (Foundry)**")
                    st.code(latest["original_code"], language="python", line_numbers=True)
                with col_conv:
                    st.markdown("**Converted (Databricks)**")
                    st.code(latest["converted_code"], language="python", line_numbers=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: OUTPUT
# ════════════════════════════════════════════════════════════════════════════════

with tab_output:
    convs = get_project_conversions()

    if not convs:
        st.info("No conversions yet. Go to the Convert tab to start converting files.")
    else:
        completed = [c for c in convs.values() if c["status"] == "done"]
        errors = [c for c in convs.values() if c["status"] == "error"]

        st.subheader(f"Conversion Output ({len(completed)} completed, {len(errors)} errors)")

        # Download All as ZIP
        if completed:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for conv in completed:
                    out_name = conv["file_path"].replace(".py", "_databricks.py")
                    zf.writestr(out_name, conv["converted_code"])
            zip_buffer.seek(0)

            st.download_button(
                "📦 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="converted_notebooks.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

        st.divider()

        # List all conversions
        for conv_id, conv in convs.items():
            layer_badge = {"bronze": "badge-bronze", "silver": "badge-silver", "gold": "badge-gold"}.get(conv["target_layer"], "badge-blue")
            status_badge = "badge-green" if conv["status"] == "done" else "badge-red"

            col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 2])
            col1.markdown(f"`{conv['file_path']}`")
            col2.markdown(f'<span class="badge {layer_badge}">{conv["target_layer"]}</span>', unsafe_allow_html=True)
            col3.markdown(f'<span class="badge {status_badge}">{conv["status"]}</span>', unsafe_allow_html=True)

            issue_count = len(conv.get("issues", []))
            if conv["status"] == "done" and issue_count > 0:
                col4.markdown(f'<span class="badge badge-yellow">⚠️ {issue_count}</span>', unsafe_allow_html=True)
            elif conv["status"] == "done":
                col4.markdown('<span class="badge badge-green">✓ Clean</span>', unsafe_allow_html=True)
            else:
                col4.write("")

            with col5:
                if conv["status"] == "done":
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        out_name = conv["file_path"].replace(".py", "_databricks.py")
                        st.download_button(
                            "⬇️",
                            data=conv["converted_code"],
                            file_name=out_name,
                            mime="text/x-python",
                            key=f"dl_{conv_id}",
                            help="Download converted file",
                        )
                    with bcol2:
                        if st.button("👁", key=f"view_{conv_id}", help="View side-by-side"):
                            st.session_state[f"view_conv_{conv_id}"] = True

            # Expandable side-by-side view
            if st.session_state.get(f"view_conv_{conv_id}"):
                with st.expander(f"📄 {conv['file_path']} — Side by Side", expanded=True):
                    if conv.get("issues"):
                        st.warning(f"⚠️ {len(conv['issues'])} unconverted API(s):")
                        for issue in conv["issues"]:
                            st.caption(f"Line {issue.line_number}: `{issue.pattern}` — `{issue.line}`")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original (Foundry)**")
                        st.code(conv["original_code"], language="python", line_numbers=True)
                    with c2:
                        st.markdown("**Converted (Databricks)**")
                        st.code(conv["converted_code"], language="python", line_numbers=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: ASK AI (RAG Chat)
# ════════════════════════════════════════════════════════════════════════════════

with tab_chat:
    project_files = get_project_files()

    if not project_files:
        st.info("Upload files first in the Files tab to enable Ask AI.")
    elif not st.session_state.api_key:
        st.warning("Set your X-API-KEY in the sidebar to use Ask AI.", icon="🔑")
    else:
        st.subheader("Ask AI About Your Code")
        st.caption("Questions are answered using RAG — only your uploaded code is used as context.")

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chunks"):
                    with st.expander("📎 Source chunks used"):
                        for chunk in msg["chunks"]:
                            st.caption(f"`{chunk.file_path}` lines {chunk.start_line}–{chunk.end_line} (score: {chunk.score:.2f})")

        # Chat input
        if prompt := st.chat_input("Ask about your code... e.g., 'What does cleaned_case.py do?'"):
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Retrieve relevant chunks
            idx = get_index()
            chunks = idx.search(prompt, top_k=5)

            # Build conversation history for API
            api_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]  # exclude current message
            ]

            # Call LLM
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chat_with_code(
                            api_url=st.session_state.api_url,
                            api_key=st.session_state.api_key,
                            model=st.session_state.model_name,
                            message=prompt,
                            chunks=chunks,
                            history=api_history,
                        )
                        st.markdown(response)

                        if chunks:
                            with st.expander("📎 Source chunks used"):
                                for chunk in chunks:
                                    st.caption(
                                        f"`{chunk.file_path}` lines {chunk.start_line}–{chunk.end_line} (score: {chunk.score:.2f})"
                                    )

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "chunks": chunks,
                        })

                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
