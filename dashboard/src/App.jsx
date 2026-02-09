import React, { useState, useEffect } from "react";
import axios from "axios";
import katex from 'katex';
import 'katex/dist/katex.min.css';
import "./App.css";

const Latex = ({ children }) => {
  try {
    const html = katex.renderToString(children, { throwOnError: false, displayMode: false });
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  } catch (e) {
    return <span>{children}</span>;
  }
};

export default function App() {
  const [params, setParams] = useState({
    model_type: "ising_1d",
    num_sites: 20,
    g_x: 1.2,
    g_z: 0.1,
    dt: 0.1,
    steps: 50,
    wp1: { x0: 5, k0: 0.35, sigma: 1 },
    wp2: { x0: 15, k0: -0.35, sigma: 1 }
  });

  const [modelInfo, setModelInfo] = useState({ name: "", latex: "", description: "" });
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [time, setTime] = useState(0);

  // Fetch model info
  useEffect(() => {
    axios.get(`http://localhost:8000/model-info?model_type=${params.model_type}`)
      .then(res => setModelInfo(res.data))
      .catch(err => console.error("Model info fetch error:", err));
  }, [params.model_type]);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      console.log("Sending request:", params);
      const response = await axios.post("http://localhost:8000/simulate", params);
      console.log("Got response:", response.data);
      setHeatmapData(response.data.heatmap);
    } catch (err) {
      console.error("Simulation error:", err);
      let msg = "Simulation Failed";
      if (err.response?.data?.detail) {
        msg += typeof err.response.data.detail === 'string'
          ? `: ${err.response.data.detail}`
          : `: ${JSON.stringify(err.response.data.detail)}`;
      } else {
        msg += `: ${err.message}`;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const renderCanvas = () => {
    if (!heatmapData || heatmapData.length === 0) return null;

    const canvas = document.getElementById('heatmap-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const rows = heatmapData.length;
    const cols = heatmapData[0].length;

    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    heatmapData.forEach(row => row.forEach(val => {
      if (val < min) min = val;
      if (val > max) max = val;
    }));

    const range = max - min || 1;

    // Draw heatmap
    heatmapData.forEach((row, t) => {
      row.forEach((val, x) => {
        const normalized = (val - min) / range;
        const hue = (1 - normalized) * 240; // Blue to Red
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(x * cellWidth, t * cellHeight, cellWidth, cellHeight);
      });
    });
  };

  useEffect(() => {
    if (heatmapData) renderCanvas();
  }, [heatmapData]);

  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <h1>⚛️ QUANTUM SCATTERING LAB</h1>
          <span className="version">v2.1 Pro</span>
        </div>
        <div className="status">
          <span className={loading ? 'pulse' : ''}>●</span>
          {loading ? "Computing..." : "Ready"}
        </div>
      </header>

      <main className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="section">
            <h3>Model Selection</h3>
            <select value={params.model_type} onChange={(e) => setParams({ ...params, model_type: e.target.value })}>
              <option value="ising_1d">1D Ising Model</option>
              <option value="ising_2d">2D Ising Model</option>
              <option value="su2_gauge">SU(2) Gauge Theory</option>
            </select>
          </div>

          {modelInfo.latex && (
            <div className="latex-box">
              <strong>{modelInfo.name}</strong>
              <div className="latex-eq">
                <Latex>{modelInfo.latex}</Latex>
              </div>
              <p className="desc">{modelInfo.description}</p>
            </div>
          )}

          <div className="section">
            <h3>System Parameters</h3>
            <label>
              System Size (L): {params.num_sites}
              <input
                type="range"
                min="5"
                max="50"
                value={params.num_sites}
                onChange={(e) => setParams({ ...params, num_sites: parseInt(e.target.value) })}
              />
            </label>

            <label>
              Transverse Field (g_x): {params.g_x.toFixed(2)}
              <input
                type="range"
                min="0"
                max="3"
                step="0.1"
                value={params.g_x}
                onChange={(e) => setParams({ ...params, g_x: parseFloat(e.target.value) })}
              />
            </label>

            <label>
              Longitudinal Field (g_z): {params.g_z.toFixed(2)}
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.g_z}
                onChange={(e) => setParams({ ...params, g_z: parseFloat(e.target.value) })}
              />
            </label>

            <label>
              Time Steps: {params.steps}
              <input
                type="range"
                min="10"
                max="200"
                value={params.steps}
                onChange={(e) => setParams({ ...params, steps: parseInt(e.target.value) })}
              />
            </label>
          </div>

          <div className="section">
            <h3>Wavepackets</h3>
            <div className="wp-group">
              <strong>Left Particle</strong>
              <input
                type="number"
                placeholder="x0"
                value={params.wp1.x0}
                onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, x0: parseFloat(e.target.value) } })}
              />
              <input
                type="number"
                placeholder="k0"
                value={params.wp1.k0}
                onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, k0: parseFloat(e.target.value) } })}
              />
            </div>

            <div className="wp-group">
              <strong>Right Particle</strong>
              <input
                type="number"
                placeholder="x0"
                value={params.wp2.x0}
                onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, x0: parseFloat(e.target.value) } })}
              />
              <input
                type="number"
                placeholder="k0"
                value={params.wp2.k0}
                onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, k0: parseFloat(e.target.value) } })}
              />
            </div>
          </div>

          <button
            onClick={runSimulation}
            disabled={loading}
            className="run-btn"
          >
            {loading ? "⏳ Running..." : "▶ Run Simulation"}
          </button>

          {error && <div className="error-msg">{error}</div>}
        </aside>

        {/* Display Area */}
        <section className="display-area">
          {!heatmapData ? (
            <div className="empty-state">
              <h2>⚛</h2>
              <p>Configure parameters and run simulation to visualize quantum dynamics</p>
            </div>
          ) : (
            <div className="viz-panel">
              <h3>Energy Density Evolution (Vacuum Subtracted)</h3>
              <canvas id="heatmap-canvas" width="800" height="400"></canvas>
              <div className="colorbar">
                <div className="gradient"></div>
                <span>Min</span>
                <span>Max</span>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
