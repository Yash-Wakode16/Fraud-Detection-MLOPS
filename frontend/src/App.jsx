
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const HISTORY_KEY = "fraud-detection-history";

const initialForm = {
  step: 100,
  type: "TRANSFER",
  amount: 5000,
  oldbalanceOrg: 10000,
  newbalanceOrig: 5000,
  oldbalanceDest: 2000,
  newbalanceDest: 7000,
};

const pages = [
  { id: "overview", icon: "▦", label: "Overview" },
  { id: "predict", icon: "⌁", label: "Live Prediction" },
  { id: "history", icon: "▤", label: "Prediction History" },
  { id: "performance", icon: "◫", label: "Model Performance" },
  { id: "monitoring", icon: "◉", label: "Monitoring" },
  { id: "about", icon: "ⓘ", label: "About Model" },
];

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function MetricCard({ label, value, detail, icon }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}

function Status({ fraud }) {
  return (
    <span className={`status ${fraud ? "danger" : "safe"}`}>
      {fraud ? "Flagged" : "Not flagged"}
    </span>
  );
}

function App() {
  const [page, setPage] = useState("overview");
  const [form, setForm] = useState(initialForm);
  const [history, setHistory] = useState(readHistory);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    let active = true;

    const checkApi = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (active) setApiOnline(response.ok && data.model_loaded);
      } catch {
        if (active) setApiOnline(false);
      }
    };

    checkApi();
    const interval = setInterval(checkApi, 15000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const total = history.length;
  const flagged = history.filter((item) => item.is_fraud).length;
  const safe = total - flagged;
  const averageScore = total
    ? history.reduce((sum, item) => sum + item.fraud_probability, 0) / total
    : 0;

  const updateField = (event) => {
    const { name, value } = event.target;
    setForm((previous) => ({
      ...previous,
      [name]: name === "type" ? value : value,
    }));
  };

  async function predictTransaction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = {
      ...form,
      step: Number(form.step),
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.[0]?.msg || data.detail || "Prediction failed.");
      }

      const item = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2, 7)}`,
        createdAt: new Date().toLocaleString(),
        amount: payload.amount,
        type: payload.type,
        fraud_probability: Number(data.fraud_probability),
        is_fraud: Boolean(data.is_fraud),
        threshold: Number(data.threshold),
        decision: data.decision,
        model: data.model,
      };

      setResult(item);
      setHistory((previous) => [item, ...previous].slice(0, 500));
    } catch (err) {
      setError(
        err.message ||
          "Cannot connect to the API. Check that FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  function clearHistory() {
    if (window.confirm("Delete all saved browser prediction history?")) {
      setHistory([]);
      setResult(null);
    }
  }

  const filteredHistory = history.filter((item) =>
    `${item.type} ${item.decision} ${item.createdAt}`
      .toLowerCase()
      .includes(search.toLowerCase())
  );

  function renderOverview() {
    return (
      <>
        <div className="welcome">
          <div>
            <p className="eyebrow">FRAUD INTELLIGENCE</p>
            <h2>Overview</h2>
            <p className="muted">
              A local view of your transaction predictions and model activity.
            </p>
          </div>
          <button className="primary-btn" onClick={() => setPage("predict")}>
            + Analyze transaction
          </button>
        </div>

        <div className="metric-grid">
          <MetricCard label="Predictions recorded" value={total} detail="This browser's saved history" icon="↗" />
          <MetricCard label="Transactions flagged" value={flagged} detail="Flagged by the model threshold" icon="!" />
          <MetricCard label="Not flagged" value={safe} detail="Below the decision threshold" icon="✓" />
          <MetricCard label="Average fraud score" value={`${(averageScore * 100).toFixed(2)}%`} detail="Across recorded predictions" icon="◌" />
        </div>

        <div className="two-column">
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Prediction activity</h3>
                <p className="muted">Latest saved transactions</p>
              </div>
              <button className="text-btn" onClick={() => setPage("history")}>View history →</button>
            </div>
            {history.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">⌁</div>
                <h3>No predictions yet</h3>
                <p>Analyze a transaction to populate this dashboard.</p>
                <button className="primary-btn" onClick={() => setPage("predict")}>Make first prediction</button>
              </div>
            ) : (
              <TransactionTable rows={history.slice(0, 5)} />
            )}
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Local activity breakdown</h3>
                <p className="muted">Based on saved predictions</p>
              </div>
            </div>
            <div className="breakdown">
              <div className="breakdown-row">
                <span><i className="dot red" /> Flagged</span>
                <strong>{flagged}</strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill red-fill" style={{ width: `${total ? flagged / total * 100 : 0}%` }} />
              </div>
              <div className="breakdown-row">
                <span><i className="dot green" /> Not flagged</span>
                <strong>{safe}</strong>
              </div>
              <div className="progress-track">
                <div className="progress-fill green-fill" style={{ width: `${total ? safe / total * 100 : 0}%` }} />
              </div>
              <div className="note">
                These are model decisions, not confirmed real-world fraud labels.
              </div>
            </div>
          </section>
        </div>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>System status</h3>
              <p className="muted">Local services and model</p>
            </div>
          </div>
          <div className="service-grid">
            <Service name="FastAPI backend" status={apiOnline ? "Online" : "Offline"} online={apiOnline} />
            <Service name="XGBoost model" status={apiOnline ? "Loaded" : "Unverified"} online={apiOnline} />
            <Service name="React dashboard" status="Running" online />
            <Service name="Prediction history" status="Browser storage" online />
          </div>
        </section>
      </>
    );
  }

  function renderPrediction() {
    const fields = [
      ["step", "Transaction step", "number"],
      ["amount", "Transaction amount", "number"],
      ["oldbalanceOrg", "Origin balance before", "number"],
      ["newbalanceOrig", "Origin balance after", "number"],
      ["oldbalanceDest", "Destination balance before", "number"],
      ["newbalanceDest", "Destination balance after", "number"],
    ];

    return (
      <>
        <PageHeading title="Live Prediction" subtitle="Submit transaction details to your local XGBoost API." />
        <div className="two-column prediction-layout">
          <section className="panel">
            <h3>Transaction details</h3>
            <p className="muted">Enter the transaction values to analyze.</p>
            <form onSubmit={predictTransaction} className="prediction-form">
              <label className="field">
                Transaction type
                <select name="type" value={form.type} onChange={updateField}>
                  {["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].map((type) => (
                    <option key={type}>{type}</option>
                  ))}
                </select>
              </label>
              {fields.map(([name, label]) => (
                <label className="field" key={name}>
                  {label}
                  <input
                    name={name}
                    type="number"
                    min="0"
                    step="any"
                    required
                    value={form[name]}
                    onChange={updateField}
                  />
                </label>
              ))}
              {error && <div className="error-box">{error}</div>}
              <button className="primary-btn full-width" disabled={loading || !apiOnline}>
                {loading ? "Analyzing transaction..." : "Run fraud analysis"}
              </button>
              {!apiOnline && <p className="muted small">Start the FastAPI backend to enable predictions.</p>}
            </form>
          </section>

          <section className="panel result-panel">
            <h3>Analysis result</h3>
            <p className="muted">The latest response from your model.</p>
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">◉</div>
                <h3>Awaiting transaction</h3>
                <p>Submit the form to see the model's output here.</p>
              </div>
            ) : (
              <div className="result-content">
                <Status fraud={result.is_fraud} />
                <div className="result-score">{(result.fraud_probability * 100).toFixed(4)}%</div>
                <p className="muted">Estimated fraud probability</p>
                <div className="progress-track score-track">
                  <div className={`progress-fill ${result.is_fraud ? "red-fill" : "green-fill"}`} style={{ width: `${result.fraud_probability * 100}%` }} />
                </div>
                <div className="detail-row"><span>Decision</span><strong>{result.decision}</strong></div>
                <div className="detail-row"><span>Decision threshold</span><strong>{result.threshold.toFixed(6)}</strong></div>
                <div className="detail-row"><span>Model</span><strong>{result.model}</strong></div>
                <div className="note">A prediction is a model assessment, not proof that a transaction is fraudulent or safe.</div>
                <button className="secondary-btn full-width" onClick={() => { setForm(initialForm); setResult(null); setError(""); }}>Reset form</button>
              </div>
            )}
          </section>
        </div>
      </>
    );
  }

  function renderHistory() {
    return (
      <>
        <PageHeading title="Prediction History" subtitle="Browse predictions saved in this browser." />
        <section className="panel">
          <div className="table-toolbar">
            <input className="search-input" placeholder="Search type, decision, or date..." value={search} onChange={(e) => setSearch(e.target.value)} />
            <button className="secondary-btn" onClick={() => setSearch("")}>Clear search</button>
            <button className="danger-btn" onClick={clearHistory} disabled={!history.length}>Clear history</button>
          </div>
          {filteredHistory.length ? (
            <TransactionTable rows={filteredHistory} />
          ) : (
            <div className="empty-state"><h3>No matching predictions</h3><p>Run a prediction or change your search.</p></div>
          )}
          <p className="muted small">{filteredHistory.length} record(s) shown. Maximum retained: 500.</p>
        </section>
      </>
    );
  }

  function renderPerformance() {
    const metrics = [
      ["Precision", "1.0000", "No false positives in the saved test evaluation"],
      ["Recall", "0.9762", "164 of 168 fraud cases detected"],
      ["F1-score", "0.9880", "Harmonic mean of precision and recall"],
      ["PR-AUC", "1.0000", "Area under the precision-recall curve"],
      ["ROC-AUC", "1.0000", "Area under the ROC curve"],
      ["False negatives", "4", "Fraud cases missed in the test evaluation"],
    ];

    return (
      <>
        <PageHeading title="Model Performance" subtitle="Stored test-set evaluation for the current XGBoost model." />
        <div className="info-banner">These are offline test results from your saved evaluation report. They are not live monitoring metrics. The near-perfect scores should be revalidated with a time-based split and a review of feature availability before making production claims.</div>
        <div className="metric-grid performance-grid">
          {metrics.map(([label, value, detail]) => <MetricCard key={label} label={label} value={value} detail={detail} icon="◫" />)}
        </div>
        <section className="panel">
          <div className="panel-heading">
            <div><h3>Confusion matrix</h3><p className="muted">Saved XGBoost test results</p></div>
          </div>
          <div className="confusion-matrix">
            <div className="matrix-cell matrix-label">Actual / Predicted</div>
            <div className="matrix-cell matrix-label">Not fraud</div>
            <div className="matrix-cell matrix-label">Fraud</div>
            <div className="matrix-cell matrix-label">Not fraud</div>
            <div className="matrix-cell true-cell"><strong>127,085</strong><span>True negatives</span></div>
            <div className="matrix-cell false-cell"><strong>0</strong><span>False positives</span></div>
            <div className="matrix-cell matrix-label">Fraud</div>
            <div className="matrix-cell false-cell"><strong>4</strong><span>False negatives</span></div>
            <div className="matrix-cell true-cell"><strong>164</strong><span>True positives</span></div>
          </div>
          <p className="muted small">Test set: 127,253 transactions. Threshold: approximately 0.999801.</p>
        </section>
      </>
    );
  }

  function renderMonitoring() {
    const types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"];
    return (
      <>
        <PageHeading title="Monitoring" subtitle="Basic local monitoring based on prediction records stored in this browser." />
        <div className="metric-grid">
          <MetricCard label="Recorded predictions" value={total} detail="Saved locally" icon="↗" />
          <MetricCard label="Flagged decisions" value={flagged} detail="Model flagged" icon="!" />
          <MetricCard label="Mean fraud score" value={`${(averageScore * 100).toFixed(3)}%`} detail="Across saved predictions" icon="◌" />
          <MetricCard label="API status" value={apiOnline ? "Online" : "Offline"} detail="Checked every 15 seconds" icon="●" />
        </div>
        <section className="panel">
          <h3>Predictions by transaction type</h3>
          <p className="muted">Counts reflect browser history only.</p>
          <div className="type-monitor">
            {types.map((type) => {
              const count = history.filter((item) => item.type === type).length;
              const percentage = total ? count / total * 100 : 0;
              return (
                <div className="type-row" key={type}>
                  <div className="type-row-top"><span>{type}</span><strong>{count}</strong></div>
                  <div className="progress-track"><div className="progress-fill blue-fill" style={{ width: `${percentage}%` }} /></div>
                </div>
              );
            })}
          </div>
          <div className="note">This page does not yet measure data drift, ground-truth fraud rates, or model degradation. Those require stored prediction logs and verified outcomes over time.</div>
        </section>
      </>
    );
  }

  function renderAbout() {
    return (
      <>
        <PageHeading title="About the Model" subtitle="Technical overview of your fraud detection application." />
        <div className="two-column">
          <section className="panel">
            <h3>Model information</h3>
            <div className="detail-row"><span>Algorithm</span><strong>XGBoost Classifier</strong></div>
            <div className="detail-row"><span>Task</span><strong>Binary classification</strong></div>
            <div className="detail-row"><span>Dataset</span><strong>PaySim</strong></div>
            <div className="detail-row"><span>Threshold</span><strong>0.999801</strong></div>
            <div className="detail-row"><span>API framework</span><strong>FastAPI</strong></div>
            <div className="detail-row"><span>Frontend</span><strong>React + Vite</strong></div>
            <div className="detail-row"><span>Experiment tracking</span><strong>MLflow file store</strong></div>
          </section>
          <section className="panel">
            <h3>Feature engineering</h3>
            <p className="muted">The model receives transaction information and derived balance-difference features.</p>
            <ul className="feature-list">
              <li>Transaction step and type</li>
              <li>Transaction amount</li>
              <li>Origin and destination balances</li>
              <li>Origin balance error</li>
              <li>Destination balance error</li>
            </ul>
            <div className="note">Before real-world use, verify that all features are available at the moment a transaction needs to be assessed.</div>
          </section>
        </div>
        <section className="panel">
          <h3>Project architecture</h3>
          <div className="architecture">
            <div className="architecture-node">React Dashboard</div>
            <span>↓ HTTP request</span>
            <div className="architecture-node">FastAPI /predict</div>
            <span>↓ feature preparation</span>
            <div className="architecture-node">XGBoost + threshold</div>
            <span>↓ JSON prediction</span>
            <div className="architecture-node">Result + browser history</div>
          </div>
        </section>
      </>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">F</div>
          <div><strong>FraudSight</strong><span>ML INTELLIGENCE</span></div>
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {pages.map((item) => (
            <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => setPage(item.id)}>
              <span className="nav-icon">{item.icon}</span>{item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="api-status"><span className={`status-light ${apiOnline ? "online" : ""}`} /><div><strong>Local API</strong><span>{apiOnline ? "Connected" : "Disconnected"}</span></div></div>
          <div className="sidebar-foot">Portfolio project · Local environment</div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div><span className="breadcrumb">Workspace</span><span className="breadcrumb-divider">/</span><strong>{pages.find((item) => item.id === page)?.label}</strong></div>
          <div className="topbar-right"><span className={`connection-pill ${apiOnline ? "connected" : "disconnected"}`}><i />{apiOnline ? "API Connected" : "API Offline"}</span><div className="avatar">ML</div></div>
        </header>
        <div className="page-content">
          {page === "overview" && renderOverview()}
          {page === "predict" && renderPrediction()}
          {page === "history" && renderHistory()}
          {page === "performance" && renderPerformance()}
          {page === "monitoring" && renderMonitoring()}
          {page === "about" && renderAbout()}
          <footer className="footer">FraudSight · Local fraud detection analytics <span>For demonstration and learning purposes</span></footer>
        </div>
      </main>
    </div>
  );
}

function PageHeading({ title, subtitle }) {
  return <div className="page-heading"><div><p className="eyebrow">FRAUD INTELLIGENCE</p><h2>{title}</h2><p className="muted">{subtitle}</p></div></div>;
}

function Service({ name, status, online }) {
  return <div className="service-item"><span className={`status-light ${online ? "online" : ""}`} /><div><strong>{name}</strong><span>{status}</span></div></div>;
}

function TransactionTable({ rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Transaction</th><th>Type</th><th>Amount</th><th>Fraud score</th><th>Decision</th></tr></thead>
        <tbody>
          {rows.map((item) => (
            <tr key={item.id}>
              <td><span className="transaction-id">TX-{item.id.slice(-7).toUpperCase()}</span><span className="table-date">{item.createdAt}</span></td>
              <td>{item.type}</td>
              <td>{Number(item.amount).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
              <td>{(item.fraud_probability * 100).toFixed(3)}%</td>
              <td><Status fraud={item.is_fraud} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
