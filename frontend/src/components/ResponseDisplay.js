import React from 'react';
import './ResponseDisplay.css';


function ResponseDisplay({ data }) {
    const getStatusIcon = (status) => {
        if (status === 'Approved') return '✅';
        if (status === 'Rejected') return '❌';
        return '⚠️';
    };

    const getPriorityColor = (priority) => {
        if (priority === 'Critical') return '#dc3545';
        if (priority === 'High') return '#fd7e14';
        if (priority === 'Normal') return '#28a745';
        return '#6c757d';
    };

    return (
        <div className="response-display">
            <h2>System Response</h2>

            <div className="status-banner" style={{
                background: data.status === 'Approved' ? '#d4edda' : 
                           data.status === 'Rejected' ? '#f8d7da' : '#fff3cd',
                borderColor: data.status === 'Approved' ? '#c3e6cb' :
                            data.status === 'Rejected' ? '#f5c6cb' : '#ffeeba'
            }}>
                <span className="status-icon">{getStatusIcon(data.status)}</span>
                <div>
                    <h3>{data.status}</h3>
                    <p>{data.decision_message || 'No message'}</p>
                </div>
            </div>

            <div className="modules-used">
                <h4>Modules Activated</h4>
                <div className="module-chips">
                    {data.modules_used?.map((mod, i) => (
                        <span key={i} className="module-chip">{mod}</span>
                    ))}
                </div>
            </div>

            {data.predicted_priority && (
                <div className="info-card priority-card">
                    <h4>🎯 Priority Assessment</h4>
                    <span className="priority-badge" style={{
                        background: getPriorityColor(data.predicted_priority),
                        color: 'white'
                    }}>
                        {data.predicted_priority}
                    </span>
                </div>
            )}

            {data.policy_status && (
                <div className="info-card">
                    <h4>📋 Policy Validation</h4>
                    <p><strong>Status:</strong> {data.policy_status}</p>
                    {data.policy_reason && <p><strong>Reason:</strong> {data.policy_reason}</p>}
                </div>
            )}

            {data.recommended_route && (
                <div className="info-card route-card">
                    <h4>🗺️ Recommended Route</h4>
                    <div className="route-path">
                        {data.recommended_route.map((node, i) => (
                            <React.Fragment key={i}>
                                <span className="route-node">{node}</span>
                                {i < data.recommended_route.length - 1 && (
                                    <span className="route-arrow">→</span>
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                    <p className="route-stats">
                        <span>⏱️ {data.estimated_travel_time} min</span>
                        {data.estimated_cost && <span> | 💰 Cost: {data.estimated_cost}</span>}
                    </p>
                </div>
            )}

            {data.control_action && (
                <div className="info-card control-card">
                    <h4>🚦 Control Allocation</h4>
                    {typeof data.control_action === 'object' ? (
                        <div>
                            <p><strong>Plan:</strong> {data.control_action.plan_type || 'N/A'}</p>
                            {data.control_action.signals && (
                                <div className="signals-list">
                                    {data.control_action.signals.map((sig, i) => (
                                        <div key={i} className="signal-item">
                                            <span className="signal-id">{sig.signal_id}</span>
                                            <span className="signal-phase">{sig.phase}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ) : (
                        <p>{data.control_action}</p>
                    )}
                </div>
            )}

            {data.explanatory_text && (
                <div className="info-card explanation-card">
                    <h4>📝 Detailed Explanation</h4>
                    <pre className="explanation-text">{data.explanatory_text}</pre>
                </div>
            )}
        </div>
    );
}

export default ResponseDisplay;