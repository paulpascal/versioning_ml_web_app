pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
        VENV_PATH = 'venv'
        TEST_DATA_PATH = 'data/test_data.csv'
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
                    python${PYTHON_VERSION} -m venv ${VENV_PATH}
                    . ${VENV_PATH}/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Source Code Tests') {
            steps {
                // Activate virtual environment and run source code tests
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Run unit tests
                    pytest tests/unit/ --junitxml=unit-test-results.xml
                    # Run integration tests
                    pytest tests/integration/ --junitxml=integration-test-results.xml
                    # Run style checks
                    flake8 app/ tests/
                    black --check app/ tests/
                    # Run type checking
                    mypy app/
                    # Run security checks
                    bandit -r app/
                '''
            }
        }
        
        stage('Model Training Test') {
            steps {
                // Activate virtual environment and run model training test
                sh '''
                    . ${VENV_PATH}/bin/activate
                    # Check if test data exists
                    if [ ! -f "${TEST_DATA_PATH}" ]; then
                        echo "Test data not found at ${TEST_DATA_PATH}"
                        exit 1
                    fi
                    # Run model training test
                    python -m pytest tests/test_model_training.py --junitxml=model-test-results.xml
                '''
            }
        }
    }
    
    post {
        always {
            // Archive test results
            junit '**/test-results.xml'
            junit '**/model-test-results.xml'
            
            // Clean up
            sh 'rm -rf ${VENV_PATH}'
        }
        
        success {
            echo "All tests passed successfully!"
        }
        
        failure {
            echo "Pipeline failed. Check the test results for details."
        }
    }
} 