import React, { useState } from 'react';
import './RequestForm.css';


const CATEGORIES = [
    { value: 'Route_Request', label: '📍 Route Request', desc: 'Standard navigation' },
    { value: 'Policy_Check', label: '✓ Policy Check', desc: 'Validate authorization' },
    { value: 'Control_Allocation_Request', label: '🚦 Control Allocation', desc: 'Signal control' },
    { value: 'Emergency_Response_Request', label: '🚨 Emergency Response', desc: 'Priority routing' },
    { value: 'Integrated_City_Service_Request', label: '🏙️ Integrated Service', desc: 'Full coordination' }
];

const VEHICLES = ['Civilian', 'Ambulance', 'Fire_Truck', 'Police'];
const SEVERITIES = ['None', 'Low', 'Medium', 'High'];
const TIME_SENS = ['Normal', 'High'];
const LOCATIONS = [
    'Central_Junction', 'North_Station', 'River_Bridge', 'Police_HQ',
    'Traffic_Control_Center', 'Stadium', 'East_Market', 'Airport_Road',
    'City_Hospital', 'South_Residential', 'West_Terminal',
    'Fire_Station', 'Industrial_Zone'
];

function RequestForm({ onSubmit, loading }) {
    const [form, setForm] = useState({
        request_id: `REQ-${Date.now()}`,
        vehicle_type: 'Civilian',
        request_category: 'Route_Request',
        current_location: 'Central_Junction',
        destination: 'City_Hospital',
        incident_severity: 'None',
        time_sensitivity: 'Normal',
        traffic_density: 0,
        priority_claim: 0,
        control_zone: null,
        description_note: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setForm(prev => ({
            ...prev,
            [name]: name === 'traffic_density' || name === 'priority_claim' 
                ? parseInt(value) || 0 
                : value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(form);
    };

    const isEmergency = form.vehicle_type !== 'Civilian';
    const needsAdvanced = ['Control_Allocation_Request', 'Emergency_Response_Request', 'Integrated_City_Service_Request']
        .includes(form.request_category);

    return (
        <form onSubmit={handleSubmit} className="request-form">
            <h2>Submit Traffic Request</h2>

            <div className="form-group">
                <label>Request ID</label>
                <input type="text" name="request_id" value={form.request_id} 
                    onChange={handleChange} required />
            </div>

            <div className="form-row">
                <div className="form-group">
                    <label>Vehicle Type</label>
                    <select name="vehicle_type" value={form.vehicle_type} onChange={handleChange}>
                        {VEHICLES.map(v => <option key={v} value={v}>{v}</option>)}
                    </select>
                </div>

                <div className="form-group">
                    <label>Request Category</label>
                    <select name="request_category" value={form.request_category} onChange={handleChange}>
                        {CATEGORIES.map(c => (
                            <option key={c.value} value={c.value}>{c.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="form-row">
                <div className="form-group">
                    <label>From (Current Location)</label>
                    <select name="current_location" value={form.current_location} onChange={handleChange}>
                        {LOCATIONS.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>

                <div className="form-group">
                    <label>To (Destination)</label>
                    <select name="destination" value={form.destination} onChange={handleChange}>
                        {LOCATIONS.map(l => <option key={l} value={l}>{l}</option>)}
                    </select>
                </div>
            </div>

            {needsAdvanced && (
                <div className="advanced-fields">
                    <h3>Advanced Options</h3>
                    
                    <div className="form-row">
                        <div className="form-group">
                            <label>Incident Severity</label>
                            <select name="incident_severity" value={form.incident_severity} onChange={handleChange}>
                                {SEVERITIES.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>

                        <div className="form-group">
                            <label>Time Sensitivity</label>
                            <select name="time_sensitivity" value={form.time_sensitivity} onChange={handleChange}>
                                {TIME_SENS.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                    </div>

                    <div className="form-row">
                        <div className="form-group">
                            <label>Traffic Density (0-10)</label>
                            <input type="range" name="traffic_density" min="0" max="10" 
                                value={form.traffic_density} onChange={handleChange} />
                            <span className="range-value">{form.traffic_density}</span>
                        </div>

                        <div className="form-group">
                            <label>Priority Claim (0-3)</label>
                            <input type="range" name="priority_claim" min="0" max="3" 
                                value={form.priority_claim} onChange={handleChange} />
                            <span className="range-value">{form.priority_claim}</span>
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Control Zone (optional)</label>
                        <input type="text" name="control_zone" value={form.control_zone} 
                            onChange={handleChange} placeholder="e.g., S1_Central_Junction" />
                    </div>
                </div>
            )}

            <div className="form-group">
                <label>Description</label>
                <textarea name="description_note" value={form.description_note} 
                    onChange={handleChange} rows="2" placeholder="Optional notes..." />
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Processing...' : '🚀 Process Request'}
            </button>
        </form>
    );
}

export default RequestForm;