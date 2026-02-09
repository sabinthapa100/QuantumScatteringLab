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

const EntropySparkline = ({ data, currentIdx }) => {
  if (!data || data.length === 0) return null;
  const max = Math.max(...data) || 1;
  const points = data.map((v, i) => `${(i / data.length) * 100},${100 - (v / max) * 100}`).join(" ");

  return (
    <svg viewBox="0 0 100 100" className="sparkline">
      <polyline points={points} fill="none" stroke="#a032ff" strokeWidth="2" />
      <circle cx={(currentIdx / data.length) * 100} cy={100 - (data[currentIdx] / max) * 100} r="4" fill="#f00acc" />
    </svg>
  );
};

export default function App() {
  const [params, setParams] = useState({
    model: "ising_1d",
    num_sites: 20,
    g_x: 1.2,
    g_z: 0.1,
    dt: 0.1,
    steps: 100,
    wp1: { x0: 5, k0: 0.35, sigma: 1.5 },
    wp2: { x0: 15, k0: -0.35, sigma: 1.5 },
    wp3: null
  });

  const [modelInfo, setModelInfo] = useState({ name: "", latex: "", description: "", parameters: [] });
  const [heatmapData, setHeatmapData] = useState(null);
  const [entropy, setEntropy] = useState([]);
  const [logs, setLogs] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTimeStep, setCurrentTimeStep] = useState(0);

  const addLog = (msg) => {
    setLogs(prev => [...prev.slice(-9), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  // Fetch model info when model changes
  useEffect(() => {
    axios.get(`http://localhost:8000/model-info?model_type=${params.model}`)
      .then(res => {
        setModelInfo(res.data);
        addLog(`Model changed to: ${res.data.name}`);
      })
      .catch(err => console.error("Model info fetch error:", err));
  }, [params.model]);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    addLog(`Launching ${params.model} simulation...`);
    try {
      const response = await axios.post("http://localhost:8000/simulate", params);
      setHeatmapData(response.data.heatmap);
      setEntropy(response.data.entropy || []);
      setAnalysis(response.data.analysis);
      setCurrentTimeStep(0);
      addLog("Simulation complete. Analyzing scattering data...");

      // Live playback
      let t = 0;
      const interval = setInterval(() => {
        if (t >= response.data.heatmap.length - 1) {
          clearInterval(interval);
          addLog("Playback finished.");
        } else {
          t++;
          setCurrentTimeStep(t);
        }
      }, 30);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setError("Simulation Failed: " + msg);
      addLog("ERROR: " + msg);
    } finally {
      setLoading(false);
    }
  };

  const computeFFT = (data) => {
    if (!data) return [];
    const N = data.length;
    const fft = [];
    for (let k = 0; k < N; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const phi = (2 * Math.PI * k * n) / N;
        re += data[n] * Math.cos(phi);
        im -= data[n] * Math.sin(phi);
      }
      fft.push(Math.sqrt(re * re + im * im));
    }
    return fft;
  };

  const renderVisuals = () => {
    if (!heatmapData) return null;
    const currentData = heatmapData[currentTimeStep];
    const fftData = computeFFT(currentData);

    const canvas = document.getElementById('main-canvas');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (params.model === "ising_2d") {
        const L = Math.sqrt(currentData.length);
        const cellSize = Math.min(w, h) / L;
        currentData.forEach((val, i) => {
          const x = i % L;
          const y = Math.floor(i / L);
          const norm = Math.min(Math.abs(val) * 2.5, 1);
          const hue = (1 - norm) * 200 + 160;
          ctx.fillStyle = `hsla(${hue}, 80%, 50%, ${norm})`;
          ctx.fillRect(x * cellSize, y * cellSize, cellSize - 1, cellSize - 1);
        });
      } else {
        const step = w / currentData.length;
        ctx.beginPath();
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00f2fe';
        currentData.forEach((val, i) => {
          const y = h / 2 - val * 70;
          if (i === 0) ctx.moveTo(0, y);
          else ctx.lineTo(i * step, y);
        });
        ctx.stroke();
      }
    }

    const fftCanvas = document.getElementById('fft-canvas');
    if (fftCanvas) {
      const ctx = fftCanvas.getContext('2d');
      ctx.clearRect(0, 0, fftCanvas.width, fftCanvas.height);
      const step = fftCanvas.width / fftData.length;
      ctx.fillStyle = '#a032ff';
      fftData.forEach((val, i) => {
        const hVal = (val / 15) * fftCanvas.height;
        ctx.fillRect(i * step, fftCanvas.height - hVal, step - 1, hVal);
      });
    }
  };

  useEffect(() => {
    renderVisuals();
  }, [heatmapData, currentTimeStep]);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo-container">
          <span className="atom-logo">⚛</span>
          <h1>Quantum Scattering Lab</h1>
          <span className="version">v3.0 ENTERPRISE</span>
        </div>
        <div className="status">
          <span style={{ color: loading ? '#f00acc' : '#00f2fe' }}>●</span>
          {loading ? "PROFILING TENSORS..." : "LABORATORY ONLINE"}
        </div>
      </header>

      <main className="main-layout">
        <aside className="sidebar">
          <div className="section">
            <h3>Physics Engine</h3>
            <select value={params.model} onChange={(e) => setParams({ ...params, model: e.target.value })}>
              <option value="ising_1d">1D Ising Model</option>
              <option value="ising_2d">2D Lattice</option>
              <option value="su2_gauge">SU(2) Gauge Theory</option>
            </select>
          </div>

          <div className="latex-box">
            <strong>{modelInfo.name}</strong>
            <div className="latex-eq">
              <Latex>{modelInfo.latex || "Loading..."}</Latex>
            </div>
            <p className="desc">{modelInfo.description}</p>
          </div>

          <div className="section">
            <h3>Configuration</h3>
            <label>Sites (L): {params.num_sites}</label>
            <input type="range" min="4" max="100" value={params.num_sites}
              onChange={e => setParams({ ...params, num_sites: parseInt(e.target.value) })} />

            <label>Transverse Field (g_x): {params.g_x.toFixed(2)}</label>
            <input type="range" min="0" max="2" step="0.05" value={params.g_x}
              onChange={e => setParams({ ...params, g_x: parseFloat(e.target.value) })} />

            <label>Longitudinal Field (g_z): {params.g_z.toFixed(2)}</label>
            <input type="range" min="-1" max="1" step="0.05" value={params.g_z}
              onChange={e => setParams({ ...params, g_z: parseFloat(e.target.value) })} />
          </div>

          <div className="section">
            <h3>Wavepackets (x0, k0)</h3>
            <div className="wp-group">
              <input type="number" value={params.wp1.x0} onChange={e => setParams({ ...params, wp1: { ...params.wp1, x0: parseFloat(e.target.value) } })} />
              <input type="number" value={params.wp1.k0} onChange={e => setParams({ ...params, wp1: { ...params.wp1, k0: parseFloat(e.target.value) } })} />
            </div>
            <div className="wp-group">
              <input type="number" value={params.wp2.x0} onChange={e => setParams({ ...params, wp2: { ...params.wp2, x0: parseFloat(e.target.value) } })} />
              <input type="number" value={params.wp2.k0} onChange={e => setParams({ ...params, wp2: { ...params.wp2, k0: parseFloat(e.target.value) } })} />
            </div>
            <button style={{ fontSize: '0.6rem', padding: '4px', cursor: 'pointer' }} onClick={() => setParams({ ...params, wp3: params.wp3 ? null : { x0: 10, k0: 0, sigma: 1.5 } })}>
              {params.wp3 ? "[-] Remove 3rd Packet" : "[+] Add 3rd Packet"}
            </button>
            {params.wp3 && (
              <div className="wp-group">
                <input type="number" value={params.wp3.x0} onChange={e => setParams({ ...params, wp3: { ...params.wp3, x0: parseFloat(e.target.value) } })} />
                <input type="number" value={params.wp3.k0} onChange={e => setParams({ ...params, wp3: { ...params.wp3, k0: parseFloat(e.target.value) } })} />
              </div>
            )}
          </div>

          <button className="run-btn" onClick={runSimulation} disabled={loading}>
            {loading ? "QUANTUM PROFILING..." : "DEPLOY SIMULATION"}
          </button>

          <div className="section scientific-console">
            <h3>Scientific Console</h3>
            <div className="log-container-inner">
              {logs.map((log, i) => <div key={i} className="log-line">{log}</div>)}
              {loading && <div className="log-line blink">Calculating MPS contraction...</div>}
            </div>
          </div>

          {error && <div className="error-msg">{error}</div>}
        </aside>

        <section className="display-area">
          <div className="viz-panel">
            <h3>Real-time Dynamics (t = {(currentTimeStep * params.dt).toFixed(2)})</h3>
            <canvas id="main-canvas" width="800" height="400"></canvas>

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <div style={{ flex: 1 }}>
                <h4 className="sub-header">MOMENTUM SPACE (FFT)</h4>
                <canvas id="fft-canvas" width="400" height="100"></canvas>
              </div>
              <div style={{ flex: 1 }}>
                <h4 className="sub-header">ENTANGLEMENT ENTROPY S(t)</h4>
                <div className="entropy-viz">
                  {entropy.length > 0 && (
                    <EntropySparkline data={entropy} currentIdx={currentTimeStep} />
                  )}
                </div>
              </div>
              {analysis && (
                <div className="analysis-box">
                  <h4 className="sub-header">S-MATRIX ESTIMATE</h4>
                  <div className="metric">T: {analysis.transmission.toFixed(3)}</div>
                  <div className="metric">R: {analysis.reflection.toFixed(3)}</div>
                  <div className="metric">η: {analysis.inelasticity.toFixed(3)}</div>
                </div>
              )}
            </div>
          </div>

          <div className="metrics-mini">
            <div className="metric-card-mini">
              <span className="metric-label-mini">Solver Baseline</span>
              <span className="metric-value-mini">DMRG</span>
            </div>
            <div className="metric-card-mini">
              <span className="metric-label-mini">Backend</span>
              <span className="metric-value-mini">QUIMB MPS</span>
            </div>
            <div className="metric-card-mini">
              <span className="metric-label-mini">Max Bond Dim</span>
              <span className="metric-value-mini">χ = 128</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
