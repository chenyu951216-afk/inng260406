from flask import Flask, jsonify, render_template
from config.settings import settings
from services.dashboard_state_service import DashboardStateService

app = Flask(__name__, template_folder="templates")
state_service = DashboardStateService()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/dashboard")
def dashboard():
    return jsonify(state_service.build_state())

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", str(settings.ui_port)))
    app.run(host=settings.ui_host, port=port, debug=False)
