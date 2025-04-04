from flask import (
    request,
    jsonify,
    session,
    current_app,
    render_template,
    Blueprint,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
import os
from app.utils.data_handler import DataHandler
from app.utils.model_handler import ModelHandler
import threading
import time
import subprocess
from datetime import datetime
from scripts.dvc_setup import add_to_dvc, push_to_remote

# Create Blueprint
main = Blueprint("main", __name__)

# Global variables for training state
training_state = {
    "in_progress": False,
    "progress": 0,
    "logs": [],
    "completed": False,
    "results": None,
    "model_handler": None,  # This will store the actual ModelHandler instance
}

# Global variables for data loading state
data_loading_state = {
    "in_progress": False,
    "progress": 0,
    "error": None,
    "preview_data": None,
    "columns": None,
}


@main.context_processor
def inject_now():
    """
    Inject the current time into the template
    """
    return {"now": datetime.utcnow()}


@main.route("/")
def index():
    """
    Renders the index.html template.
    """
    return render_template("index.html", training_completed=False)


@main.route("/about")
def about():
    """
    Renders the about.html template.
    """
    # Team members data
    team_members = [
        {
            "name": "Paul Alognon-Anani",
            "avatar": "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        },
        {
            "name": "Castelnau Godefroy Ondongo",
            "avatar": "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        },
        {
            "name": "Amadou Tidiane Diallo",
            "avatar": "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        },
        {
            "name": "Mayombo Abel M.O",
            "avatar": "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        },
        {
            "name": "Joan-Yves Darys Anguilet",
            "avatar": "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",
        },
    ]

    return render_template("about.html", team_members=team_members)


@main.route("/setup", methods=["GET", "POST"])
def setup():
    """
    Renders the setup.html template.
    """
    if request.method == "POST":
        try:
            config = request.get_json()
            if not config:
                return jsonify({"error": "No configuration data provided"}), 400

            # Validate required fields
            required_fields = ["model_type", "features", "target", "train_size"]
            for field in required_fields:
                if field not in config:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            # Store configuration in session
            session["model_config"] = config
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # GET request
    if "filepath" not in session:
        return redirect(url_for("main.index"))

    return render_template("setup.html", training_completed=False)


@main.route("/train", methods=["GET"])
def train():
    """
    Renders the train.html template with model configuration.
    """
    # Get the model configuration from the session
    model_config = session.get("model_config", {})

    if not model_config:
        return redirect(url_for("main.setup"))

    # Get training completion status from the global state
    training_completed = training_state.get("completed", False)

    return render_template(
        "train.html",
        model_type=model_config.get("model_type", ""),
        target_column=model_config.get("target", ""),
        selected_columns=model_config.get("features", []),
        normalize=model_config.get("normalize", True),
        training_completed=training_completed,
    )


def process_data_async(filepath):
    """Background task for data processing"""
    global data_loading_state

    try:
        # Update progress for file saving
        data_loading_state["progress"] = 20

        # Add to DVC (without pushing)
        try:
            add_to_dvc(filepath)
            data_loading_state["progress"] = 40
        except Exception as dvc_error:
            print(f"Warning: Failed to add to DVC: {str(dvc_error)}")

        # Load and process data
        data_handler = DataHandler(filepath)
        data_loading_state["progress"] = 60

        preview_data = data_handler.get_preview()
        data_loading_state["progress"] = 80

        columns = data_handler.get_columns()
        data_loading_state["progress"] = 100

        # Store results in state
        data_loading_state["preview_data"] = preview_data.to_dict(orient="records")
        data_loading_state["columns"] = columns
        data_loading_state["filepath"] = filepath
        data_loading_state["in_progress"] = False
        data_loading_state["error"] = None

    except Exception as e:
        data_loading_state["error"] = str(e)
        data_loading_state["in_progress"] = False
        # Clean up file if error occurs
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass


@main.route("/upload", methods=["POST"])
def upload_file():
    """
    Uploads a file to the server and starts the data processing in the background.
    """
    global data_loading_state

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

        try:
            # Save file
            file.save(filepath)

            # Reset and start loading state
            data_loading_state = {
                "in_progress": True,
                "progress": 0,
                "error": None,
                "preview_data": None,
                "columns": None,
            }

            # Start processing in background
            thread = threading.Thread(target=process_data_async, args=(filepath,))
            thread.start()

            return jsonify({"success": True, "message": "File upload started"})

        except Exception as e:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


@main.route("/data_loading_status")
def data_loading_status():
    """Get the current status of data loading"""
    if data_loading_state["error"]:
        return jsonify({"in_progress": False, "error": data_loading_state["error"]})
    elif data_loading_state["in_progress"]:
        return jsonify(
            {"in_progress": True, "progress": data_loading_state["progress"]}
        )
    elif data_loading_state["preview_data"] is not None:
        # Store in session for later use
        session["filepath"] = data_loading_state["filepath"]
        session["preview_data"] = data_loading_state["preview_data"]
        session["columns"] = data_loading_state["columns"]

        return jsonify(
            {
                "in_progress": False,
                "completed": True,
                "preview": data_loading_state["preview_data"],
                "columns": data_loading_state["columns"],
            }
        )
    else:
        return jsonify({"in_progress": False, "completed": False})


@main.route("/get_dataset_columns")
def get_dataset_columns():
    """
    Returns the columns of the uploaded file.
    """
    if "filepath" not in session:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        data_handler = DataHandler(session["filepath"])
        return jsonify({"columns": data_handler.get_columns()["names"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/save_config", methods=["POST"])
def save_config():
    """
    Saves the configuration of the model.
    """
    config = request.json
    session["model_config"] = config
    return jsonify({"success": True})


@main.route("/get_config")
def get_config():
    """
    Returns the configuration of the model.
    """
    if "model_config" not in session:
        return jsonify({"error": "No configuration found"}), 400
    return jsonify(session["model_config"])


def train_model(data_handler, config):
    """
    Background task for model training
    """
    global training_state

    try:
        # Prepare data
        training_state["logs"].append("Preparing data...")
        training_state["progress"] = 10

        X_train, X_test, y_train, y_test = data_handler.prepare_data(
            features=config["features"],
            target=config["target"],
            train_size=config["train_size"],
            normalize=config["normalize"],
        )

        # Lets say its %30 of the total time
        training_state["progress"] = 30
        training_state["logs"].append("Data preparation completed")

        # Create and train model
        training_state["logs"].append("Initializing model...")
        model_handler = ModelHandler(
            model_type=config["model_type"],
            features=config["features"],
            target=config["target"],
            train_size=config["train_size"],
            normalize=config["normalize"],
        )

        # Lets say its %50 of the total time
        training_state["progress"] = 50
        training_state["logs"].append("Starting model training...")

        # Train model
        success = model_handler.train(X_train, X_test, y_train, y_test)

        if success:
            training_state["progress"] = 100
            results = model_handler.get_results()
            print("Model training results:", results)  # Debug log
            training_state["results"] = results
            training_state["model_handler"] = model_handler
            training_state["logs"].append("Training completed successfully!")
        else:
            training_state["logs"].append("Error during training")
            if "error" in model_handler.get_results():
                training_state["logs"].append(
                    f"Error: {model_handler.get_results()['error']}"
                )

    except Exception as e:
        training_state["logs"].append(f"Error during training: {str(e)}")
    finally:
        training_state["completed"] = True
        training_state["in_progress"] = False


@main.route("/start_training", methods=["POST"])
def start_training():
    """
    Starts the training of the model.
    """
    global training_state

    if training_state["in_progress"]:
        # Reset training state if it's been more than 30 seconds
        if time.time() - training_state.get("start_time", 0) > 30:
            training_state = {
                "in_progress": False,
                "progress": 0,
                "logs": [],
                "completed": False,
                "results": None,
                "start_time": 0,
            }
        else:
            return jsonify({"error": "Training already in progress"}), 400

    if "model_config" not in session or "filepath" not in session:
        return jsonify({"error": "Missing configuration or data"}), 400

    # Reset training state
    training_state.update(
        {
            "in_progress": True,
            "progress": 0,
            "logs": [],
            "completed": False,
            "results": None,
            "start_time": time.time(),
        }
    )

    # Start training in background
    data_handler = DataHandler(session["filepath"])
    thread = threading.Thread(
        target=train_model, args=(data_handler, session["model_config"])
    )
    thread.start()

    return jsonify({"success": True})


@main.route("/training_status")
def training_status():
    """
    Returns the status of the training.
    """
    # Create a copy of training state without the model handler
    status = {
        "in_progress": training_state["in_progress"],
        "progress": training_state["progress"],
        "logs": training_state["logs"],
        "completed": training_state["completed"],
        "results": training_state["results"],
    }
    print("Training status response:", status)  # Debug log
    return jsonify(status)


@main.route("/save_model", methods=["POST"])
def save_model():
    """
    Saves the trained model.
    """
    if not training_state["completed"] or not training_state["results"]:
        return jsonify({"error": "No trained model to save"}), 400

    try:
        model_name = request.json["name"]
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400

        # Use the stored model handler from training state
        if "model_handler" not in training_state:
            return jsonify({"error": "Model training data not found"}), 400

        model_handler = training_state["model_handler"]
        filepath = model_handler.save_model(model_name)

        # Now push everything to remote
        if push_to_remote():
            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "message": f"Model saved successfully at: {filepath}\nAll files pushed to remote storage.",
                }
            )
        else:
            return jsonify(
                {
                    "success": True,
                    "filepath": filepath,
                    "warning": "Failed to push to remote storage",
                    "message": f"Model saved locally at: {filepath}\nWarning: Failed to push to remote storage.",
                }
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/get_suggested_name")
def get_suggested_name():
    """Get a suggested name for the model based on its type"""
    if "model_handler" not in training_state:
        return jsonify({"error": "No trained model available"}), 400

    model_handler = training_state["model_handler"]
    suggested_name = model_handler.get_default_name()

    return jsonify({"success": True, "suggested_name": suggested_name})


def allowed_file(filename):
    """
    Checks if the file is allowed to be uploaded.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "csv",
        "xlsx",
        "xls",
    }
