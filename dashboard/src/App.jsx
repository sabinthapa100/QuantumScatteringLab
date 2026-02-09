import React, { useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import { Play, Pause, RefreshCw, Sliders, Zap, Settings, Activity } from "lucide-react";
import "./App.css";

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

  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://localhost:8000/simulate", params);
      setHeatmapData(response.data.heatmap);
    } catch (err) {
      console.error(err);
      setError("Simulation Failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <Zap className="accent-icon" size={28} />
          <div>
            <h1>Quantum Scattering Lab</h1>
            <span className="version">v2.0 Premium</span>
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
              <option value="su2_gauge">SU(2) Lattice Gauge Theory</option>
            </select>
          </div>

          <div className="control-row">
            <div className="control-group">
              <label>System Size (L)</label>
              <input type="number"
                value={params.num_sites} onChange={(e) => setParams({ ...params, num_sites: parseInt(e.target.value) })} />
            </div>
            <div className="control-group">
              <label>Steps (T)</label>
              <input type="number"
                value={params.steps} onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })} />
            </div>
          </div>

          <div className="divider"></div>
          <div className="section-title"><Settings size={16} /> Parameters</div>

          <div className="control-row">
            <div className="control-group">
              <label>g_x (Transverse)</label>
              <input type="number" step="0.1"
                value={params.g_x} onChange={(e) => setParams({ ...params, g_x: parseFloat(e.target.value) })} />
            </div>
            <div className="control-group">
              <label>g_z (Longitudinal)</label>
              <input type="number" step="0.1"
                value={params.g_z} onChange={(e) => setParams({ ...params, g_z: parseFloat(e.target.value) })} />
            </div>
          </div>

          <div className="divider"></div>
          <div className="section-title">Input State (Wavepackets)</div>

          {/* WP1 */}
          <div className="wavepacket-card cyan-border">
            <div className="card-header cyan-text">Particle 1 (Left)</div>
            <div className="control-row">
              <input type="number" placeholder="x0" className="mini-input"
                value={params.wp1.x0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, x0: parseFloat(e.target.value) } })} />
              <input type="number" placeholder="k0" className="mini-input"
                value={params.wp1.k0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, k0: parseFloat(e.target.value) } })} />
            </div>
          </div>

          {/* WP2 */}
          <div className="wavepacket-card pink-border">
            <div className="card-header pink-text">Particle 2 (Right)</div>
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
            {loading ? "Simulating..." : "Run Experiment"}
          </button>

          {error && <div className="error-msg">{error}</div>}
        </aside>

        {/* Visualization Area */}
        <section className="display-area">
          {!heatmapData ? (
            <div className="empty-state">
              <Zap size={64} opacity={0.2} />
              <p>Configure parameters and click Run Experiment to visualize quantum dynamics.</p>
            </div>
          ) : (
            <div className="plot-container">
              <div className="plot-header">
                <h2>Energy Density Evolution <span className="sub">Vacuum Subtracted</span></h2>
              </div>
              <div className="plot-wrapper">
                <Plot
                  data={[
                    {
                      z: heatmapData,
                      type: "heatmap",
                      colorscale: "Viridis",
                      zsmooth: "best",
                      showscale: true,
                      colorbar: {
                        title: "Density",
                        titleside: 'right',
                        titlefont: { color: '#888' },
                        tickfont: { color: '#888' }
                      }
                    }
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#aaa', family: 'Inter, sans-serif' },
                    margin: { l: 50, r: 50, b: 50, t: 30 },
                    xaxis: { title: 'Site Index (Space)', color: '#888' },
                    yaxis: { title: 'Time Step (Evolution)', color: '#888', autorange: 'reversed' }
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
