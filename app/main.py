from flask import Flask
import os
from dotenv import load_dotenv
from app.routes import main
from scripts.dvc_setup import setup_dvc_gdrive

resp = load_dotenv()


def create_app():
    """
    Create the Flask app
    """
    app = Flask(__name__)

    # Configuration from environment variables
    app.config["APP_NAME"] = os.getenv("APP_NAME", "ML Web App")

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
    )

    # Set upload folder path
    app.config["UPLOAD_FOLDER"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        os.getenv("UPLOAD_FOLDER", "data"),
    )

    # Set models folder path
    app.config["MODEL_FOLDER"] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), os.getenv("MODEL_FOLDER", "models")
    )

    # Ensure required directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

    # Register blueprints
    from .routes import main

    app.register_blueprint(main)

    return app


if __name__ == "__main__":
    # Setup DVC with Google Drive
    setup_dvc_gdrive()

    app = create_app()
    app.run(debug=True)
