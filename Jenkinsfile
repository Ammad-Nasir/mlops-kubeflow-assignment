pipeline {
    agent any

    stages {
        stage('Environment Setup') {
            steps {
                echo 'Setting up Python environment'
                sh 'python --version'
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo 'Compiling Kubeflow pipeline'
                sh 'python pipeline.py'
                sh 'test -f pipeline.yaml'
            }
        }
    }

    post {
        always {
            echo 'Jenkins pipeline finished.'
        }
    }
}


