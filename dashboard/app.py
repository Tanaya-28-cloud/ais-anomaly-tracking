from flask import Flask
from routes.views import views_bp
from routes.api import api_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
