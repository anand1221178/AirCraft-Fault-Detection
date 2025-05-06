from flask import Flask, render_template, send_from_directory
import os
import re

app = Flask(__name__)

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "Codes", "Data", "experiment_results", "analysis_output"))

# --- Helper: Parse classification report section into HTML table ---
def format_classification_report(text):
    lines = text.splitlines()
    section = []
    capture = False
    for line in lines:
        if "Classification Report" in line:
            capture = True
            continue
        if capture:
            if line.strip() == "":
                break
            section.append(line)

    if not section or len(section) < 2:
        # fallback to plain if not parsed
        return f"<pre>{text}</pre>"

    # Generate table
    table = "<table border='1' cellpadding='5' style='background:#fff; border-collapse: collapse;'>"
    table += "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>"
    for line in section[1:]:
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) == 5:
            table += f"<tr><td>{parts[0]}</td><td>{parts[1]}</td><td>{parts[2]}</td><td>{parts[3]}</td><td>{parts[4]}</td></tr>"
    table += "</table>"

    return table


@app.route("/")
def index():
    # Evaluation report files to load
    report_files = {
        "Engine (Full DBN)": "report_engine_Full_DBN.txt",
        "EGT Sensor Health": "report_egt_health_Full_DBN.txt",
        "Vibration Sensor Health": "report_vib_health_Full_DBN.txt",
        "Engine (Vanilla DBN)": "report_engine_Vanilla_DBN.txt",
        "Engine (Rule-Based)": "report_engine_Rule_Based.txt"
    }

    # Confusion matrix image files
    matrix_images = {
        "Engine Confusion Matrix": "cm_engine_Full_DBN.png",
        "EGT Sensor Matrix": "cm_egt_health_Full_DBN.png",
        "Vibration Sensor Matrix": "cm_vib_health_Full_DBN.png"
    }

    # Load report content and format classification tables
    report_contents = {}
    for name, filename in report_files.items():
        file_path = os.path.join(RESULTS_PATH, filename)
        try:
            with open(file_path, "r") as f:
                raw = f.read()
                report_contents[name] = format_classification_report(raw)
        except FileNotFoundError:
            report_contents[name] = f"<pre>[Missing file: {filename}]</pre>"

    return render_template("index.html", reports=report_contents, matrices=matrix_images)

@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(RESULTS_PATH, filename)

@app.route("/scenario_plot/<filename>")
def get_scenario_plot(filename):
    plot_path = os.path.abspath(os.path.join(BASE_PATH, "..", "Codes", "Data", "inference_plots_all_scenarios"))
    return send_from_directory(plot_path, filename)

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)
