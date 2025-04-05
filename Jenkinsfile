pipeline {
    agent any
    
    environment {
        VENV_PATH = 'venv'
        PYTHONPATH = "${WORKSPACE}"
        // DVC Configuration
        GOOGLE_DRIVE_CREDENTIALS = credentials('google-drive-credentials')
        DVC_REMOTE_URL = credentials('dvc-remote-url')
        // Git Configuration
        GIT_CREDENTIALS = credentials('github-credentials')
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/paulpascal/versioning_ml_web_app.git'
            }
        }
        
        stage('Setup') {
            steps {
                // Create and activate virtual environment
                sh '''
                    python -m venv ${VENV_PATH}
                    . ${VENV_PATH}/bin/activate
                    python -m pip install -r requirements.txt
                '''
            }
        }

        stage('DVC Setup') {
            steps {
                sh '''
                    # Write Google Drive credentials to a temporary file
                    echo "${GOOGLE_DRIVE_CREDENTIALS}" > google_credentials.json
                    
                    # Setup DVC with Google Drive
                    python scripts/dvc_setup.py
                    
                    # Verify DVC remote configuration
                    dvc remote list
                    
                    # Pull latest data and models
                    dvc pull
                    
                    # Clean up credentials file
                    rm google_credentials.json
                '''
            }
        }
        
        stage('Source Code Tests') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run test data handler tests with detailed logging
                    PYTHONPATH=${WORKSPACE} pytest tests/test_data_handler.py \
                        --junitxml=data-handler-test-results.xml \
                        --log-cli-level=DEBUG \
                        --capture=tee-sys \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S"
                    
                    # Run test model handler tests with detailed logging
                    PYTHONPATH=${WORKSPACE} pytest tests/test_model_handler.py \
                        --junitxml=model-handler-test-results.xml \
                        --log-cli-level=DEBUG \
                        --capture=tee-sys \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S"
                '''
            }
        }
        
        stage('Model Training Test') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run end-to-end model training test
                    PYTHONPATH=${WORKSPACE} pytest tests/test_model_training.py \
                        --junitxml=model-training-test-results.xml \
                        --log-cli-level=DEBUG \
                        --capture=tee-sys \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S" \
                        -v  # Verbose output
                '''
            }
        }

        stage('Model Validation Test') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run validation tests on the newly trained model
                    PYTHONPATH=${WORKSPACE} pytest tests/test_model_validation.py \
                        --junitxml=model-validation-test-results.xml \
                        --log-cli-level=DEBUG \
                        --capture=tee-sys \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S" \
                        -v  # Verbose output
                '''
            }
        }

        stage('Git Push') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Configure git credentials
                    git config --global user.email "paul.alogno+jenkins@gmail.com"
                    git config --global user.name "Paul@Jenkins"
                    
                    # Add and commit DVC metadata files
                    git add *.dvc
                    git commit -m "Jenkins: Update DVC metadata with new models and data" || true
                    
                    # Push changes to git
                    git push https://${GIT_CREDENTIALS}@github.com/paulpascal/versioning_ml_web_app.git main
                '''
            }
        }
    }
    
    post {
        always {
            // Archive test results - this will show them in the Jenkins UI
            junit 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml,model-validation-test-results.xml'
            
            // Archive artifacts for download
            archiveArtifacts artifacts: 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml,model-validation-test-results.xml', allowEmptyArchive: true
            
            // Clean up
            sh 'rm -rf ${VENV_PATH}'
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            echo "Pipeline failed. Check the test results for details."
        }
    }
} 