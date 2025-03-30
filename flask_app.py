from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import cv2
import json
import glob

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ZONES_FILE = "zones.json"

# Load or initialize zones.json
if not os.path.exists(ZONES_FILE):
    with open(ZONES_FILE, "w") as f:
        json.dump({}, f)

def clear_upload_cache():
    """Removes all previous uploads to start fresh."""
    files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*"))
    for f in files:
        os.remove(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        clear_upload_cache()  # Clear old uploads
        video = request.files["video"]
        zone = request.form["zone"]

        if video:
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], "video.mp4")
            video.save(video_path)

            # Extract first frame
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            cap.release()

            if success:
                original_height, original_width, _ = frame.shape  # Get original image resolution
                print(f"Original Frame Resolution: {original_width}x{original_height}")  

                # Resize only if width > 1000px
                max_width = 1000
                scale_factor = 1  # Default scale factor

                if original_width > max_width:
                    scale_factor = max_width / original_width  # Scaling factor
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))  # Resize image

                    print(f"Resized Frame Resolution: {new_width}x{new_height}")  # Print resized image resolution

                frame_path = os.path.join(app.config["UPLOAD_FOLDER"], "frame.jpg")
                cv2.imwrite(frame_path, frame)  # Save resized image

                # Store scale factor for later use
                with open(os.path.join(app.config["UPLOAD_FOLDER"], "scale.json"), "w") as f:
                    json.dump({"scale_factor": scale_factor}, f)

                return redirect(url_for("annotate", zone=zone))

    return render_template("index.html")


@app.route("/annotate/<zone>")
def annotate(zone):
    frame_path = os.path.join(app.config["UPLOAD_FOLDER"], "frame.jpg")

    if not os.path.exists(frame_path):
        return "Error: Frame not found", 404

    # Load scale factor
    scale_file = os.path.join(app.config["UPLOAD_FOLDER"], "scale.json")
    if not os.path.exists(scale_file):
        return jsonify({"error": "Scale factor not found"}), 500

    with open(scale_file, "r") as f:
        scale_data = json.load(f)
        scale_factor = scale_data["scale_factor"]

    # Load existing zones
    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)

    # Prepare scaled coordinates for other zones
    other_zones = {
        k: [(int(x * scale_factor), int(y * scale_factor)) for x, y in v]
        for k, v in zones.items() if k != zone
    }

    return render_template("annotate.html", zone=zone, frame_url="/uploads/frame.jpg", other_zones=other_zones)


@app.route("/save_zone", methods=["POST"])
def save_zone():
    data = request.get_json()
    zone = data.get("zone")
    points = data.get("points")

    if len(points) != 4:
        return jsonify({"error": "Polygon must have exactly 4 points"}), 400

    # Load scale factor
    scale_file = os.path.join(app.config["UPLOAD_FOLDER"], "scale.json")
    if not os.path.exists(scale_file):
        return jsonify({"error": "Scale factor not found"}), 500

    with open(scale_file, "r") as f:
        scale_data = json.load(f)
        scale_factor = scale_data["scale_factor"]

    # Convert points back to original resolution
    original_points = [(int(x / scale_factor), int(y / scale_factor)) for x, y in points]

    with open(ZONES_FILE, "r") as f:
        zones = json.load(f)
    zones[zone] = original_points  # Save in original size

    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=4)

    return jsonify({"message": "Zone updated successfully"}), 200

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
