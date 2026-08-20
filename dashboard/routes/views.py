from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
@views_bp.route("/map")
def map_view():
    return render_template("map.html")


# Placeholder pages for Phase 2 features — scaffolded now so the nav
# and routing structure doesn't need to change later, only the
# template content and any new API endpoints they call.

@views_bp.route("/logs")
def logs_view():
    return render_template("placeholder.html", page_name="Log Verification")


@views_bp.route("/stats")
def stats_view():
    return render_template("placeholder.html", page_name="Statistics")
