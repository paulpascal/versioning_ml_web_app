pipeline {
    agent any
    
    environment {
        VENV_PATH = 'venv'
        PYTHONPATH = "${WORKSPACE}"
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
        
        stage('Source Code Tests') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run test data handler tests with detailed logging
                    PYTHONPATH=${WORKSPACE} pytest tests/test_data_handler.py \
                        --junitxml=data-handler-test-results.xml \
                        --log-cli-level=INFO \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S"
                    
                    # Run test model handler tests with detailed logging
                    PYTHONPATH=${WORKSPACE} pytest tests/test_model_handler.py \
                        --junitxml=model-handler-test-results.xml \
                        --log-cli-level=INFO \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S"
                '''
            }
        }
        
        stage('Model Training Test') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run model training tests with detailed logging
                    PYTHONPATH=${WORKSPACE} pytest tests/test_model_training.py \
                        --junitxml=model-training-test-results.xml \
                        --log-cli-level=INFO \
                        --log-cli-format="%(asctime)s [%(levelname)8s] %(message)s" \
                        --log-cli-date-format="%Y-%m-%d %H:%M:%S" \
                        -v  # Verbose output
                '''
            }
        }
    }
    
    post {
        always {
            // Archive test results - this will show them in the Jenkins UI
            junit 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml'
            
            // Archive artifacts for download
            archiveArtifacts artifacts: 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml', allowEmptyArchive: true
            
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