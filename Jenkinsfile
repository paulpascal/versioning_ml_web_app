pipeline {
    agent any
    
    environment {
        VENV_PATH = 'venv'
        TEST_DATA_PATH = 'data/test_data.csv'
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
                // Clean up any existing virtual environment first
                sh 'rm -rf ${VENV_PATH}'
                
                // Install virtualenv if needed
                sh 'pip install virtualenv || pip3 install virtualenv'
                
                // Create virtual environment using virtualenv with timeout and error handling
                sh '''
                    timeout 60 virtualenv ${VENV_PATH} || true
                    . ${VENV_PATH}/bin/activate || (sleep 5 && . ${VENV_PATH}/bin/activate)
                    pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
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
            // Archive test results - using exact filenames
            junit 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml'
            
            // Archive console output - using exact filenames
            archiveArtifacts artifacts: 'data-handler-test-results.xml,model-handler-test-results.xml,model-training-test-results.xml', allowEmptyArchive: true
            
            // Clean up
            sh 'rm -rf ${VENV_PATH}'
            
            // Print test summary
            script {
                def testResults = currentBuild.rawBuild.getAction(hudson.tasks.junit.TestResultAction.class)
                if (testResults != null) {
                    echo "Test Summary:"
                    echo "Total Tests: ${testResults.totalCount}"
                    echo "Failed Tests: ${testResults.failCount}"
                    echo "Skipped Tests: ${testResults.skipCount}"
                    echo "Passed Tests: ${testResults.totalCount - testResults.failCount - testResults.skipCount}"
                }
            }
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            echo "Pipeline failed. Check the test results for details."
            // Print failed test details
            script {
                def testResults = currentBuild.rawBuild.getAction(hudson.tasks.junit.TestResultAction.class)
                if (testResults != null && testResults.failCount > 0) {
                    echo "Failed Tests:"
                    testResults.failedTests.each { test ->
                        echo "- ${test.fullName}: ${test.errorDetails}"
                    }
                }
            }
        }
    }
} 