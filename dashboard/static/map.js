const map = L.map('map').setView([20, 0], 2);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 18
}).addTo(map);

const markers = new Map();
let focusedOnce = false;

function vesselColor(vessel) {
    if (vessel.flagged && vessel.ml_flagged) return '#8b1e3f';
    if (vessel.flagged) return '#d62828';
    if (vessel.ml_flagged) return '#ef8354';
    return '#1769aa';
}

function markerIcon(vessel) {
    return L.divIcon({
        className: 'vessel-marker',
        html: `<span style="background:${vesselColor(vessel)}"></span>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8]
    });
}

function addLine(panel, label, value) {
    const line = document.createElement('p');
    const heading = document.createElement('strong');
    heading.textContent = `${label}: `;
    line.append(heading, document.createTextNode(value ?? '—'));
    panel.appendChild(line);
}

function showVesselDetail(vessel) {
    const panel = document.getElementById('vessel-detail-panel');
    panel.replaceChildren();
    addLine(panel, 'MMSI', vessel.mmsi);
    addLine(panel, 'Location', `${Number(vessel.lat).toFixed(6)}, ${Number(vessel.lon).toFixed(6)}`);
    addLine(panel, 'Speed', vessel.sog == null ? 'unknown' : `${vessel.sog} knots`);
    addLine(panel, 'Course', vessel.cog == null ? 'unknown' : `${vessel.cog}°`);
    addLine(panel, 'Channel', vessel.channel);
    addLine(panel, 'Last report', vessel.timestamp);
    addLine(panel, 'Rule result', vessel.flagged ? `Flagged: ${vessel.rule_anomaly_type || 'rule event'}` : 'No rule flag');
    addLine(panel, 'ML result', vessel.ml_flagged ? 'Flagged' : vessel.ml_flagged === false ? 'Not flagged' : 'No ML score received');
    if (vessel.ml_flagged != null) {
        addLine(panel, 'ML source channel', vessel.ml_channel);
        addLine(panel, 'ML score time', vessel.ml_timestamp);
    }
    if (vessel.ml_score != null) {
        addLine(panel, 'ML anomaly score', Number(vessel.ml_score).toPrecision(6));
        addLine(panel, 'ML score cutoff', Number(vessel.ml_threshold).toPrecision(6));
    }

    const shap = vessel.shap_explanation;
    if (shap && Array.isArray(shap.features)) {
        const heading = document.createElement('h6');
        heading.textContent = 'SHAP explanation for anomaly score';
        panel.appendChild(heading);
        addLine(panel, 'Method', shap.method);
        addLine(panel, 'Baseline score', Number(shap.base_value).toPrecision(6));
        addLine(panel, 'Explained score', Number(shap.score).toPrecision(6));
        addLine(panel, 'Additivity residual', Number(shap.additivity_residual).toExponential(2));
        const list = document.createElement('ul');
        shap.features
            .slice()
            .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
            .slice(0, 5)
            .forEach(item => {
                const entry = document.createElement('li');
                entry.textContent = `${item.label || item.name} = ${Number(item.value).toPrecision(5)}: ` +
                    `${Number(item.shap_value) >= 0 ? '+' : ''}${Number(item.shap_value).toPrecision(5)} ` +
                    `(${item.effect})`;
                list.appendChild(entry);
            });
        panel.appendChild(list);
        const note = document.createElement('small');
        note.className = 'text-muted';
        note.textContent = 'Approximate permutation SHAP relative to 20 training examples. Positive contributions raise this record’s anomaly score; negative contributions lower it. This explains the model score, not the cause of a real-world incident.';
        panel.appendChild(note);
    } else if (shap && shap.status === 'error') {
        addLine(panel, 'SHAP status', shap.message || 'Explanation unavailable');
    }
}

async function refreshVessels() {
    const status = document.getElementById('map-status');
    try {
        const response = await fetch('/api/vessels', { cache: 'no-store' });
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const vessels = await response.json();
        const present = new Set();

        vessels.forEach(vessel => {
            const id = String(vessel.mmsi);
            const latitude = Number(vessel.lat);
            const longitude = Number(vessel.lon);
            if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return;
            present.add(id);
            const latlng = [latitude, longitude];
            const existing = markers.get(id);
            if (existing) {
                existing.vessel = vessel;
                existing.marker.setLatLng(latlng).setIcon(markerIcon(vessel));
            } else {
                const marker = L.marker(latlng, { icon: markerIcon(vessel) }).addTo(map);
                marker.on('click', () => showVesselDetail(markers.get(id).vessel));
                markers.set(id, { marker, vessel });
            }
        });

        for (const [id, item] of markers) {
            if (!present.has(id)) {
                map.removeLayer(item.marker);
                markers.delete(id);
            }
        }

        if (!focusedOnce && markers.size) {
            const bounds = L.featureGroup(Array.from(markers.values(), item => item.marker)).getBounds();
            if (bounds.isValid()) map.fitBounds(bounds.pad(0.15), { maxZoom: 12 });
            focusedOnce = true;
        }
        status.textContent = `${markers.size} vessels · refreshed ${new Date().toLocaleTimeString()}`;
        status.className = 'small text-muted mt-2';
    } catch (error) {
        status.textContent = `Map data unavailable: ${error.message}`;
        status.className = 'small text-danger mt-2';
    }
}

refreshVessels();
setInterval(refreshVessels, 3000);
