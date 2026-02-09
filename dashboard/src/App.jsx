import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, Settings, Layers, Box, Zap, Activity } from 'lucide-react';
import axios from 'axios';

const App = () => {
  const [model, setModel] = useState('1d_ising');
  const [params, setParams] = useState({
    num_sites: 20,
    g_x: 1.25,
    g_z: 0.15,
    dt: 0.125,
    steps: 100,
    wp1: { x0: 5, k0: 0.4, sigma: 1.0 },
    wp2: { x0: 15, k0: -0.4, sigma: 1.0 }
  });

  const [logs, setLogs] = useState(["Engine initialized. Ready for simulation.", "Phase: Advanced Scattering Mode."]);

  const addLog = (msg) => setLogs(prev => [msg, ...prev].slice(0, 5));

  const runSimulation = async () => {
    setIsSimulating(true);
    addLog(`Starting ${model} simulation...`);
    try {
      const response = await axios.post('http://localhost:8000/simulate', {
        model,
        ...params
      });
      setSimulationData(response.data);
      addLog("Simulation complete. Data rendered.");
    } catch (error) {
      addLog("Error: Simulation engine unreachable.");
      console.error("Simulation failed", error);
    } finally {
      setIsSimulating(false);
    }
  };

  useEffect(() => {
    if (simulationData && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      renderHeatmap(ctx, simulationData.heatmap);
    }
  }, [simulationData]);

  const renderHeatmap = (ctx, data) => {
    const rows = data.length;
    const cols = data[0].length;
    const cellW = canvasRef.current.width / cols;
    const cellH = canvasRef.current.height / rows;

    // Magma-style color scale
    const getColor = (v) => {
      const i = Math.min(Math.max(v * 10, 0), 1);
      return `rgb(${i * 255}, ${Math.pow(i, 2) * 200}, ${Math.pow(i, 4) * 100})`;
    };

    data.forEach((row, i) => {
      row.forEach((val, j) => {
        ctx.fillStyle = getColor(val);
        ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
      });
    });
  };

  return (
    <div className="dashboard-container">
      <header>
        <div className="lab-title">QUANTUM SCATTERING LAB</div>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <div style={{ fontSize: '0.8rem', color: '#a0a0a0' }}>
            <Activity size={14} inline /> Simulation Engine: <span style={{ color: '#00f2ff' }}>Online</span>
          </div>
        </div>
      </header>

      <aside className="controls">
        <div className="control-group">
          <label><Box size={14} /> Model Configuration</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="1d_ising">1D Ising Model (Transverse/Longitudinal)</option>
            <option value="2d_ising">2D Ising Model (Square Lattice)</option>
            <option value="su2_gauge">SU(2) Gauge Theory (Plaquette Chain)</option>
          </select>
        </div>

        <div className="control-group">
          <label><Zap size={14} /> Global Couplings</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div>
              <label style={{ fontSize: '0.6rem' }}>gx (Transverse)</label>
              <input type="number" value={params.g_x} step="0.05" onChange={(e) => setParams({ ...params, g_x: parseFloat(e.target.value) })} />
            </div>
            <div>
              <label style={{ fontSize: '0.6rem' }}>gz (Longitudinal)</label>
              <input type="number" value={params.g_z} step="0.05" onChange={(e) => setParams({ ...params, g_z: parseFloat(e.target.value) })} />
            </div>
          </div>
        </div>

        <div className="control-group">
          <label>Wavepacket 1 (x0, k0, sigma)</label>
          <input type="range" min="0" max={params.num_sites} value={params.wp1.x0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, x0: parseFloat(e.target.value) } })} />
          <input type="range" min="-1" max="1" step="0.1" value={params.wp1.k0} onChange={(e) => setParams({ ...params, wp1: { ...params.wp1, k0: parseFloat(e.target.value) } })} />
        </div>

        <div className="control-group">
          <label>Wavepacket 2 (x0, k0, sigma)</label>
          <input type="range" min="0" max={params.num_sites} value={params.wp2.x0} onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, x0: parseFloat(e.target.value) } })} />
          <input type="range" min="-1" max="1" step="0.1" value={params.wp2.k0} onChange={(e) => setParams({ ...params, wp2: { ...params.wp2, k0: parseFloat(e.target.value) } })} />
        </div>

        <button className="run-btn" onClick={runSimulation} disabled={isSimulating}>
          {isSimulating ? 'Simulating...' : <><Play size={16} /> START SIMULATION</>}
        </button>
      </aside>

      <main className="display">
        <div className="visual-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
            <label>Energy Density Evolution (t vs x)</label>
            <RotateCcw size={16} style={{ cursor: 'pointer', opacity: 0.5 }} onClick={() => setSimulationData(null)} />
          </div>
          <div className="heatmap-container">
            <canvas ref={canvasRef} width={800} height={600} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
        </div>

        <div className="visual-card" style={{ flexGrow: 0, height: '150px' }}>
          <label>Engine logs</label>
          <div style={{ marginTop: '10px', fontSize: '0.8rem', color: '#888', fontFamily: 'monospace' }}>
            {logs.map((log, i) => (
              <div key={i} style={{ marginBottom: '4px', borderLeft: '2px solid' + (log.includes('Error') ? '#ff0055' : '#00f2ff'), paddingLeft: '8px' }}>
                {new Date().toLocaleTimeString()} - {log}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
