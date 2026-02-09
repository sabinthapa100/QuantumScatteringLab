
import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import { Play, RefreshCw, Sliders, Zap, Settings, Activity, Info } from "lucide-react";
import katex from 'katex';
import 'katex/dist/katex.min.css';
import "./App.css";

const Latex = ({ children }) => {
  try {
    const html = katex.renderToString(children, { throwOnError: false });
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  } catch (e) {
    return <span>{children}</span>;
  }
};

export default function App() {
  const [params, setParams] = useState({
    model_type: "ising_1d",
    num_sites: 20,
    g_x: 1.0,
    g_z: 0.0,
    dt: 0.1,
    steps: 50,
    wp1: { x0: 5.0, k0: 0.35, sigma: 1.0 },
    wp2: { x0: 15.0, k0: -0.35, sigma: 1.0 }
  });

  const [modelInfo, setModelInfo] = useState({ latex: "Loading...", description: "" });
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState("heatmap"); // heatmap, spectrum

  // Fetch model info when type changes
  useEffect(() => {
    axios.get(`http://localhost:8000/model-info?model_type=${params.model_type}`)
      .then(res => setModelInfo(res.data))
      .catch(err => console.error(err));
  }, [params.model_type]);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://localhost:8000/simulate", params);
      setHeatmapData(response.data.heatmap);
      if (response.data.real_shape) {
        // Could enable 2D view if implemented
      }
    } catch (err) {
      console.error(err);
      let msg = "Simulation Failed";
      if (err.response && err.response.data) {
        msg += ": " + (typeof err.response.data.detail === 'string' ? err.response.data.detail : JSON.stringify(err.response.data));
      } else {
        msg += ": " + err.message;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const SliderInput = ({ label, value, onChange, min, max, step }) => (
    <div className="control-group">
      <div className="flex justify-between items-center mb-1">
        <label>{label}</label>
        <span className="text-xs text-primary font-mono">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value} onChange={onChange}
        className="w-full accent-cyan-500"
      />
    </div>
  );

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <Zap className="accent-icon" size={28} />
          <div>
            <h1>Quantum Scattering Lab</h1>
            <span className="version">v2.1 Pro</span>
          </div>
        </div>
        <div className="status-indicator">
          <Activity size={16} color={loading ? "#ffd700" : "#00ff9d"} />
          <span>{loading ? "Processing Quantum Dynamics..." : "System Ready"}</span>
        </div>
      </header>

      <main className="main-layout">
        {/* Sidebar Controls */}
        <aside className="sidebar">
          <div className="section-title">
            <Sliders size={16} /> Model Configuration
          </div>

          <div className="control-group">
            <label>Physics Model</label>
            <select
              value={params.model_type}
              onChange={(e) => setParams({ ...params, model_type: e.target.value })}
            >
              <option value="ising_1d">1D Transverse Ising Model</option>
              <option value="ising_2d">2D Transverse Ising Model</option>
              <option value="su2_gauge">SU(2) Lattice Gauge Theory</option>
            </select>
          </div>

          <div className="latex-card">
            <div className="text-xs text-muted mb-2 font-bold">HAMILTONIAN</div>
            <div className="latex-content">
              <Latex>{modelInfo.latex}</Latex>
            </div>
          </div>

          <div className="control-row">
            <div className="control-group">
              <label>System Size (L)</label>
              <input type="number" min="4" max="100"
                value={params.num_sites} onChange={(e) => setParams({ ...params, num_sites: parseInt(e.target.value) })} />
            </div>
            <div className="control-group">
              <label>Steps (T)</label>
              <input type="number" min="10" max="500"
                value={params.steps} onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })} />
            </div>
          </div>

          <div className="divider"></div>
          <div className="section-title"><Settings size={16} /> Parameters</div>

          <SliderInput
            label="g_x (Transverse Field)"
            value={params.g_x}
            onChange={(e) => setParams({ ...params, g_x: parseFloat(e.target.value) })}
            min={0} max={3.0} step={0.1}
          />

          <SliderInput
            label="g_z (Longitudinal Field)"
            value={params.g_z}
            onChange={(e) => setParams({ ...params, g_z: parseFloat(e.target.value) })}
            min={0} max={1.0} step={0.05}
          />

          <div className="divider"></div>
          <div className="section-title">Wavepackets (Initial State)</div>

          {/* WP1 */}
          <div className="wavepacket-card cyan-border">
            <div className="card-header cyan-text">Left Particle</div>
            <div className="control-row">
              <input type="number" placeholder="x0" className="mini-input"
                value={params.wp1.x0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, x0: parseFloat(e.target.value) } })} />
              <input type="number" placeholder="k0" className="mini-input"
                value={params.wp1.k0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, k0: parseFloat(e.target.value) } })} />
            </div>
          </div>

          {/* WP2 */}
          <div className="wavepacket-card pink-border">
            <div className="card-header pink-text">Right Particle</div>
            <div className="control-row">
              <input type="number" placeholder="x0" className="mini-input"
                value={params.wp2.x0} onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, x0: parseFloat(e.target.value) } })} />
              <input type="number" placeholder="k0" className="mini-input"
                value={params.wp2.k0} onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, k0: parseFloat(e.target.value) } })} />
            </div>
          </div>

          <button
            onClick={runSimulation}
            disabled={loading}
            className={`run-btn ${loading ? "loading" : ""}`}
          >
            {loading ? <RefreshCw className="spin" /> : <Play size={18} />}
            {loading ? "Calculating..." : "Run Simulation"}
          </button>

          {error && <div className="error-msg">{error}</div>}
        </aside>

        {/* Visualization Area */}
        <section className="display-area">
          {!heatmapData ? (
            <div className="empty-state">
              <Zap size={64} opacity={0.2} />
              <p>Configure parameters and click Run Simulation.</p>
              <div className="mt-4 text-sm text-muted max-w-md">
                Simulate quantum dynamics using MPS tensor networks.
                Observe scattering, entanglement, and particle production in real-time.
              </div>
            </div>
          ) : (
            <div className="plot-container">
              <div className="plot-header flex justify-between">
                <h2>Energy Density Evolution <span className="sub">Vacuum Subtracted</span></h2>
                <div className="flex gap-2">
                  <button className={`px-3 py-1 text-xs rounded ${viewMode === 'heatmap' ? 'bg-cyan-900 text-cyan-300' : 'bg-neutral-800'}`} onClick={() => setViewMode('heatmap')}>Heatmap</button>
                  <button className={`px-3 py-1 text-xs rounded ${viewMode === '3d' ? 'bg-cyan-900 text-cyan-300' : 'bg-neutral-800'}`} onClick={() => setViewMode('3d')}>Surface (3D)</button>
                </div>
              </div>
              <div className="plot-wrapper">
                <Plot
                  data={[
                    viewMode === 'heatmap' ? {
                      z: heatmapData,
                      type: "heatmap",
                      colorscale: "Viridis",
                      zsmooth: "best",
                      showscale: true,
                      colorbar: { title: "Density", titleside: 'right', titlefont: { color: '#888' }, tickfont: { color: '#888' } }
                    } : {
                      z: heatmapData,
                      type: "surface",
                      colorscale: "Viridis",
                      showscale: true
                    }
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#aaa', family: 'Inter, sans-serif' },
                    margin: { l: 50, r: 20, b: 50, t: 30 },
                    xaxis: { title: 'Site Index (Space)', color: '#888' },
                    yaxis: { title: 'Time Step (Evolution)', color: '#888', autorange: 'reversed' },
                    scene: { // For 3D surface
                      xaxis: { title: 'Site', color: '#888' },
                      yaxis: { title: 'Time', color: '#888' },
                      zaxis: { title: 'Energy', color: '#888' }
                    }
                  }}
                  useResizeHandler={true}
                  style={{ width: "100%", height: "100%" }}
                  config={{ responsive: true }}
                />
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
