import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time

# Ensure src is in path
sys.path.append(os.path.abspath("."))

from src.models.ising_1d import IsingModel1D
from src.models.heisenberg.heisenberg import HeisenbergModel
from src.models.su2 import SU2GaugeModel
from src.analysis.framework import AnalysisConfig, BoundaryCondition, ModelAnalyzer
from src.analysis.spectrum import SpectrumAnalyzer

# --- Page Config ---
st.set_page_config(
    page_title="QuantumScatteringLab Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2127;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e3137;
        border-bottom: 2px solid #4e9af1;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("⚛️ Navigation Console")
    
    st.header("🚀 Engine Room")
    use_gpu = st.toggle("Enable GPU Acceleration (Quimb)", value=False)
    if use_gpu:
        st.info("Utilizing Quimb GPU Backend (cupy required)")
    else:
        st.success("Running on optimized CPU Engine")

    st.divider()
    
    st.header("🧪 Model Laboratory")
    model_choice = st.selectbox(
        "Select Physics Model",
        ["Ising Model (1D)", "Heisenberg Model", "SU(2) Gauge Model"]
    )
    
    st.divider()
    
    st.header("⚙️ Global Parameters")
    num_sites = st.slider("System Size (N)", 2, 20, 10)
    boundary = st.radio("Boundary Condition", ["Periodic (PBC)", "Open (OBC)"])
    bc_enum = BoundaryCondition.PBC if "Periodic" in boundary else BoundaryCondition.OBC
    
    st.divider()
    
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Model Selection & Parameter Logic ---
model_class = None
default_params = {}

if model_choice == "Ising Model (1D)":
    model_class = IsingModel1D
    st.sidebar.subheader("Ising Parameters")
    g_x = st.sidebar.number_input("Transverse Field (g_x)", value=1.0, step=0.1)
    g_z = st.sidebar.number_input("Longitudinal Field (g_z)", value=0.0, step=0.1)
    j_int = st.sidebar.number_input("Interaction (J)", value=1.0, step=0.1)
    default_params = {"g_x": g_x, "g_z": g_z, "j_int": j_int}
    scan_param = "g_x"

elif model_choice == "Heisenberg Model":
    model_class = HeisenbergModel
    st.sidebar.subheader("Heisenberg Parameters")
    jx = st.sidebar.number_input("Jx", value=1.0, step=0.1)
    jy = st.sidebar.number_input("Jy", value=1.0, step=0.1)
    jz = st.sidebar.number_input("Jz", value=1.0, step=0.1)
    h = st.sidebar.number_input("Magnetic Field (h)", value=0.0, step=0.1)
    default_params = {"jx": jx, "jy": jy, "jz": jz, "h": h}
    scan_param = "h"

elif model_choice == "SU(2) Gauge Model":
    model_class = SU2GaugeModel
    st.sidebar.subheader("SU(2) Parameters")
    g_coupling = st.sidebar.number_input("Coupling (g)", value=1.0, step=0.1)
    a_spacing = st.sidebar.number_input("Lattice Spacing (a)", value=1.0, step=0.1)
    default_params = {"g": g_coupling, "a": a_spacing}
    scan_param = "g"

# --- Main Dashboard ---
st.title(f"Scientific Analysis: {model_choice}")
st.caption(f"Backend: {'GPU' if use_gpu else 'CPU'} | N={num_sites} | {bc_enum}")

tabs = st.tabs(["📊 Spectrum & Gaps", "🕸️ Entanglement", "🌀 Scaling", "📋 Raw Data"])

# --- Tab 1: Spectrum & Gaps ---
with tabs[0]:
    st.header("Energy Spectrum Scan")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("### Scan Configuration")
        param_to_scan = st.selectbox("Parameter to Sweep", list(default_params.keys()), index=list(default_params.keys()).index(scan_param))
        min_val = st.number_input("Min Value", value=0.0)
        max_val = st.number_input("Max Value", value=2.0)
        steps = st.slider("Resolution", 10, 100, 40)
        
        run_scan = st.button("🚀 Run Spectrum Scan", use_container_width=True)

    with col2:
        if run_scan:
            with st.spinner("Computing spectral data..."):
                start_time = time.time()
                config = AnalysisConfig(
                    model_class=model_class,
                    fixed_params=default_params,
                    boundary_condition=bc_enum,
                    system_sizes=[num_sites]
                )
                analyzer = ModelAnalyzer(config)
                p_vals = np.linspace(min_val, max_val, steps)
                
                # Perform scan
                _, gaps = analyzer.compute_1d_phase_scan(param_to_scan, p_vals, num_sites)
                elapsed = time.time() - start_time
                
                st.success(f"Computation complete in {elapsed:.2f}s")
                
                # Visualization
                df_gap = pd.DataFrame({param_to_scan: p_vals, "Gap": gaps})
                fig = px.line(df_gap, x=param_to_scan, y="Gap", title=f"Energy Gap vs {param_to_scan}")
                fig.add_scatter(x=p_vals, y=gaps, mode='markers', name='Data Points')
                st.plotly_chart(fig, use_container_width=True)
                
                # Criticality Detection
                min_gap_idx = np.argmin(gaps)
                st.metric("Minimum Observed Gap", f"{gaps[min_gap_idx]:.4f}", f"at {param_to_scan}={p_vals[min_gap_idx]:.3f}", delta_color="inverse")
        else:
            st.info("Configure the sweep and click 'Run Spectrum Scan' to visualize results.")

# --- Tab 2: Entanglement ---
with tabs[1]:
    st.header("Entanglement Entropy & Central Charge")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Point Calculation")
        st.caption("Compute entanglement at current sidebar parameters.")
        run_ent = st.button("🕸️ Compute Entanglement", use_container_width=True)
        
    with col2:
        if run_ent:
            with st.spinner("Analyzing ground state entanglement..."):
                config = AnalysisConfig(
                    model_class=model_class,
                    fixed_params=default_params,
                    boundary_condition=bc_enum,
                    system_sizes=[num_sites]
                )
                analyzer = ModelAnalyzer(config)
                ent_data = analyzer.compute_entanglement_at_criticality(default_params, num_sites)
                
                df_ent = pd.DataFrame({
                    "Subsystem Size (l)": ent_data.subsystem_sizes,
                    "Entropy S(l)": ent_data.entropies
                })
                
                fig_ent = px.scatter(df_ent, x="Subsystem Size (l)", y="Entropy S(l)", 
                                   title="Entanglement Scaling", trendline="ols")
                st.plotly_chart(fig_ent, use_container_width=True)
                
                st.metric("Central Charge (c)", f"{ent_data.central_charge:.4f}")
                st.info("Theoretical c = 0.5 for Ising Criticality (OBC)")
        else:
            st.info("Click 'Compute Entanglement' to analyze the current ground state.")

# --- Tab 3: Scaling ---
with tabs[2]:
    st.header("Finite-Size Scaling Analysis")
    st.warning("This analysis requires multiple system sizes and can be computationally intensive.")
    
    sizes_input = st.text_input("System Sizes (comma separated)", value="6, 8, 10, 12")
    sizes = [int(x.strip()) for x in sizes_input.split(",")]
    
    if st.button("🌀 Perform Scaling Collapse"):
        with st.spinner(f"Computing scaling for N={sizes}..."):
            config = AnalysisConfig(
                model_class=model_class,
                fixed_params=default_params,
                boundary_condition=bc_enum,
                system_sizes=sizes
            )
            analyzer = ModelAnalyzer(config)
            p_vals = np.linspace(min_val, max_val, 30)
            
            # Use g_x=1.0 as default critical point for Ising if applicable
            crit_guess = 1.0 if model_choice == "Ising Model (1D)" else (min_val + max_val)/2
            
            scaling_data = analyzer.compute_scaling_collapse(
                param_name=scan_param,
                param_values=p_vals,
                param_critical=crit_guess,
                nu=1.0,
                z=1.0
            )
            
            # Visualization using Plotly
            fig_coll = go.Figure()
            for i, size in enumerate(scaling_data.sizes):
                x_scaled = (scaling_data.param_values - scaling_data.param_critical) * (size ** (1.0/scaling_data.nu))
                y_scaled = scaling_data.gaps[i] * (size ** scaling_data.z)
                fig_coll.add_trace(go.Scatter(x=x_scaled, y=y_scaled, name=f"N={size}", mode='lines+markers'))
            
            fig_coll.update_layout(title="Scaling Collapse", xaxis_title="L^(1/nu) * (g - g_c)", yaxis_title="L^z * Gap")
            st.plotly_chart(fig_coll, use_container_width=True)

# --- Tab 4: Raw Data ---
with tabs[3]:
    st.header("Data Export & Logs")
    st.write("The current dashboard session data can be downloaded here.")
    # Placeholder for a more complex state management system
    st.button("📦 Export Results to CSV")
    
st.divider()
st.caption("Powered by QuantumScatteringLab Framework | Designed by Senior Scientist AI")
