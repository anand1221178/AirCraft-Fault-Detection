from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

BASE_PATH = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "Codes", "Data", "experiment_results", "analysis_output"))

@app.route("/")
def index():
    # Files to load
    report_files = {
        "Engine (Full DBN)": "report_engine_Full_DBN.txt",
        "EGT Sensor Health": "report_egt_health_Full_DBN.txt",
        "Vibration Sensor Health": "report_vib_health_Full_DBN.txt",
        "Engine (Vanilla DBN)": "report_engine_Vanilla_DBN.txt",
        "Engine (Rule-Based)": "report_engine_Rule_Based.txt"
    }


    matrix_images = {
        "Engine Confusion Matrix": "cm_engine_Full_DBN.png",
        "EGT Sensor Matrix": "cm_egt_health_Full_DBN.png",
        "Vibration Sensor Matrix": "cm_vib_health_Full_DBN.png"
    }

    # Load report content into strings
    report_contents = {}
    for name, filename in report_files.items():
        file_path = os.path.join(RESULTS_PATH, filename)
        try:
            with open(file_path, "r") as f:
                report_contents[name] = f.read()
        except FileNotFoundError:
            report_contents[name] = f"[Missing file: {filename}]"

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
