import os
import subprocess
from dotenv import load_dotenv
from datetime import datetime

REMOTE_NAME = "my_drive"


def log_step(step_number, step_name, status, message=""):
    """Log a step with consistent formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_icon = (
        "✅" if status == "done" else "🔄" if status == "in_progress" else "❌"
    )
    print(
        f"\n[DVC_SETUP_LOG] {timestamp} | Step {step_number}: {step_name} {status_icon}"
    )
    if message:
        print(f"   └─ {message}")


def is_dvc_initialized():
    """Check if DVC is already initialized in the current directory"""
    return os.path.exists(".dvc")


def is_gdrive_configured():
    """Check if Google Drive remote is already configured"""
    try:
        result = subprocess.run(
            ["dvc", "remote", "list"], capture_output=True, text=True
        )
        return REMOTE_NAME in result.stdout
    except subprocess.CalledProcessError:
        return False


def create_and_track_directories():
    """Create and track data and models directories if they don't exist"""
    directories = ["data", "models"]
    step_number = 2

    for directory in directories:
        log_step(step_number, f"Setting up {directory} directory", "in_progress")

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            log_step(
                step_number,
                f"Setting up {directory} directory",
                "done",
                "Created directory",
            )
        else:
            log_step(
                step_number,
                f"Setting up {directory} directory",
                "done",
                "Directory already exists",
            )

        # Check if directory is already tracked by DVC
        dvc_path = f"{directory}.dvc"
        if not os.path.exists(dvc_path):
            # Add directory to DVC tracking
            subprocess.run(["dvc", "add", directory], check=True)
            log_step(
                step_number,
                f"Setting up {directory} directory",
                "done",
                "Added to DVC tracking",
            )
        else:
            log_step(
                step_number,
                f"Setting up {directory} directory",
                "done",
                "Already tracked by DVC",
            )

        step_number += 1


def setup_dvc_gdrive():
    """Setup DVC with Google Drive remote storage"""
    load_dotenv()

    print("\n[DVC_SETUP_LOG] Starting DVC and Google Drive setup...")
    print("=" * 50)

    # Step 1: Initialize DVC
    log_step(1, "DVC Initialization", "in_progress")
    if not is_dvc_initialized():
        subprocess.run(["dvc", "init"], check=True)
        log_step(1, "DVC Initialization", "done", "DVC initialized successfully")
    else:
        log_step(1, "DVC Initialization", "done", "DVC already initialized")

    # Step 2: Configure Google Drive
    log_step(2, "Google Drive Configuration", "in_progress")
    if not is_gdrive_configured():
        # Get credentials path from environment or use default
        credentials_path = os.getenv(
            "GOOGLE_DRIVE_CREDENTIALS", "google_credentials.json"
        )
        if not os.path.exists(credentials_path):
            log_step(
                2, "Google Drive Configuration", "error", "Credentials file not found"
            )
            raise ValueError(
                f"Google Drive credentials file not found at {credentials_path}. Please set GOOGLE_DRIVE_CREDENTIALS in .env or provide google_credentials.json"
            )

        # Initialize DVC with Google Drive
        subprocess.run(
            ["dvc", "remote", "add", "-d", REMOTE_NAME, os.getenv("DVC_REMOTE_URL")]
        )

        # Configure Google Drive credentials
        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                REMOTE_NAME,
                "gdrive_use_service_account",
                "true",
            ]
        )

        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                REMOTE_NAME,
                "gdrive_service_account_json_file_path",
                credentials_path,
            ]
        )

        log_step(
            2,
            "Google Drive Configuration",
            "done",
            "Google Drive configured successfully",
        )
    else:
        log_step(
            2, "Google Drive Configuration", "done", "Google Drive already configured"
        )

    # Step 3: Create and track default directories
    log_step(3, "Directory Setup", "in_progress")
    create_and_track_directories()
    log_step(3, "Directory Setup", "done", "All directories set up and tracked")

    print("\n[DVC_SETUP_LOG] Setup completed successfully!")
    print("=" * 50)


def add_to_dvc(filepath):
    """Add a file to DVC tracking without pushing

    Args:
        filepath (str): Path to the file to be versioned with DVC
    """
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    # Add file to DVC
    subprocess.run(["dvc", "add", filepath], check=True)
    print(f"File {os.path.basename(filepath)} added to DVC tracking.")


def push_to_remote():
    """Push all tracked files to DVC remote storage"""
    try:
        subprocess.run(["dvc", "push"], check=True)
        print("All tracked files pushed to DVC remote storage successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to push to DVC remote: {str(e)}")
        return False


def add_and_push_model(model_path):
    """Add and push a trained model to DVC remote storage"""
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    # Add model
    subprocess.run(["dvc", "add", model_path], check=True)

    # Push to remote
    subprocess.run(["dvc", "push"], check=True)

    print(f"Model {model_path} pushed to DVC remote storage successfully!")


if __name__ == "__main__":
    # Setup DVC with Google Drive
    setup_dvc_gdrive()
