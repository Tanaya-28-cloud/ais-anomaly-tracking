// TODO: update these coordinates to center on whichever port your
// team's dataset covers (this example uses LA/Long Beach).
const map = L.map('map').setView([33.75, -118.26], 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 18
}).addTo(map);

// mmsi -> Leaflet marker, so repeated polls update existing markers
// in place instead of piling up duplicates
let markers = {};

function colorForVessel(vessel) {
    return vessel.flagged ? 'red' : 'blue';
}

function markerIcon(vessel) {
    return L.divIcon({
        className: '',
        html: `<div style="background:${colorForVessel(vessel)}; width:12px; height:12px; border-radius:50%; border:2px solid white;"></div>`,
        iconSize: [12, 12]
    });
}

function showVesselDetail(vessel) {
    document.getElementById('vessel-detail-panel').innerHTML = `
        <p><strong>MMSI:</strong> ${vessel.mmsi}</p>
        <p><strong>Speed:</strong> ${vessel.sog} knots</p>
        <p><strong>Course:</strong> ${vessel.cog}&deg;</p>
        <p><strong>Channel:</strong> ${vessel.channel}</p>
        <p><strong>Status:</strong> ${vessel.flagged ? '<span class="text-danger">Flagged</span>' : 'Normal'}</p>
    `;
    // TODO(Phase 2): once SHAP explanations exist, add a line here
    // pulling from a new /api/anomalies/<mmsi> endpoint.
}

async function refreshVessels() {
    const res = await fetch('/api/vessels');
    const vessels = await res.json();

    vessels.forEach(vessel => {
        const latlng = [vessel.lat, vessel.lon];
        if (markers[vessel.mmsi]) {
            markers[vessel.mmsi].setLatLng(latlng);
            markers[vessel.mmsi].setIcon(markerIcon(vessel));
        } else {
            const marker = L.marker(latlng, { icon: markerIcon(vessel) }).addTo(map);
            marker.on('click', () => showVesselDetail(vessel));
            markers[vessel.mmsi] = marker;
        }
    });
}

// Poll every 3 seconds — matches the pace of the simulated replay
// stream. Easy to swap for a websocket connection later for true
// push updates if you want that polish once the core pipeline works.
refreshVessels();
setInterval(refreshVessels, 3000);
