import os
import subprocess
from dotenv import load_dotenv


def setup_dvc_gdrive():
    """Setup DVC with Google Drive remote storage"""
    load_dotenv()

    # Get credentials path from environment
    credentials_path = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        raise ValueError(
            "Google Drive credentials file not found. Please set GOOGLE_DRIVE_CREDENTIALS in .env"
        )

    # Initialize DVC with Google Drive
    subprocess.run(
        ["dvc", "remote", "add", "-d", "gdrive", os.getenv("DVC_REMOTE_URL")]
    )

    # Configure Google Drive credentials
    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "gdrive",
            "gdrive_credentials_file",
            credentials_path,
        ]
    )

    print("DVC Google Drive setup completed successfully!")


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
    subprocess.run(["dvc", "add", model_path])

    # Push to remote
    subprocess.run(["dvc", "push"])

    print(f"Model {model_path} pushed to DVC remote storage successfully!")


if __name__ == "__main__":
    # Setup DVC with Google Drive
    setup_dvc_gdrive()

    # Example: Add data from a specific path
    data_path = os.path.join("data", "raw", "student_performance.csv")
    if os.path.exists(data_path):
        add_to_dvc(data_path)
