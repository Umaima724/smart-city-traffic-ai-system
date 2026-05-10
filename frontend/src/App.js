import React, { useState } from 'react';
import RequestForm from './components/RequestForm';
import ResponseDisplay from './components/ResponseDisplay';
import './App.css';

const API_URL = 'http://localhost:8000';  // Change this to your backend URL

function App() {
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (requestData) => {
        setLoading(true);
        setError(null);
        setResponse(null);

        try {
            const res = await fetch(`${API_URL}/api/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`HTTP ${res.status}: ${errorText}`);
            }

            const data = await res.json();
            setResponse(data);
        } catch (err) {
            console.error('Fetch error:', err);
            setError(err.message || 'Failed to connect to backend. Is it running on port 8000?');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>🚦 Smart City Traffic AI System</h1>
                <p>AI-Powered Traffic Management & Emergency Response</p>
            </header>

            <main className="app-main">
                <div className="form-section">
                    <RequestForm onSubmit={handleSubmit} loading={loading} />
                </div>

                <div className="result-section">
                    {loading && (
                        <div className="loading">
                            <div className="spinner"></div>
                            <p>Processing request through AI modules...</p>
                        </div>
                    )}

                    {error && (
                        <div className="error-box">
                            <h3>❌ Error</h3>
                            <p>{error}</p>
                            <p style={{fontSize: '0.9rem', marginTop: '10px'}}>
                                Make sure backend is running: <code>uvicorn api.main:app --reload</code>
                            </p>
                        </div>
                    )}

                    {response && <ResponseDisplay data={response} />}
                </div>
            </main>
        </div>
    );
}

export default App;