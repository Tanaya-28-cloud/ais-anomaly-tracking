from flask import Blueprint, jsonify
from models.vessel_store import get_current_vessels

api_bp = Blueprint("api", __name__)


@api_bp.route("/vessels")
def vessels():
    return jsonify(get_current_vessels())


# Add new endpoints here as later phases come online, e.g.:
#
# @api_bp.route("/anomalies/<int:mmsi>")
# def anomaly_detail(mmsi):
#     return jsonify(get_anomaly_detail(mmsi))
#
# @api_bp.route("/logs/verify")
# def verify_log():
#     return jsonify(run_chain_verification())
