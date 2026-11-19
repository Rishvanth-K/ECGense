// ECGense CI/CD pipeline
// GitHub -> Jenkins -> Test -> Docker build -> Kubernetes (Kind) -> Verify
pipeline {
    agent any

    environment {
        // Jenkins (installed with Homebrew) doesn't see your shell's PATH,
        // so add where docker, kind and kubectl live on this Mac
        PATH = "/opt/homebrew/bin:/usr/local/bin:${env.PATH}"
        PYTHON = "/opt/homebrew/opt/python@3.12/bin/python3.12"
        IMAGE_NAME = "ecgense"
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        KIND_CLUSTER = "pbl3-cluster"
        KUBE_CONTEXT = "kind-pbl3-cluster"
        NAMESPACE = "ecgense"
    }

    // Check GitHub for new commits every 2 minutes
    triggers {
        pollSCM('H/2 * * * *')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --oneline'
            }
        }

        stage('Test') {
            steps {
                // The virtual environment is kept between builds, so TensorFlow
                // is downloaded only on the first run
                sh '''
                    [ -d .venv-ci ] || "$PYTHON" -m venv .venv-ci
                    . .venv-ci/bin/activate
                    pip install --quiet -r webapp/requirements.txt
                    pytest -v tests/
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build \
                      --build-arg BUILD_NUMBER=$BUILD_NUMBER \
                      --build-arg GIT_COMMIT=$GIT_COMMIT \
                      -t $IMAGE_NAME:$IMAGE_TAG \
                      -t $IMAGE_NAME:latest .
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh '''
                    # Copy the new image into the Kind cluster (no registry needed)
                    kind load docker-image $IMAGE_NAME:$IMAGE_TAG --name $KIND_CLUSTER

                    kubectl --context $KUBE_CONTEXT apply -f k8s/namespace.yaml
                    kubectl --context $KUBE_CONTEXT apply -f k8s/service.yaml
                    sed "s|IMAGE_TAG|$IMAGE_TAG|" k8s/deployment.yaml | kubectl --context $KUBE_CONTEXT apply -f -
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                sh '''
                    # Wait until the new pods are Ready (readiness probe = /health)
                    kubectl --context $KUBE_CONTEXT -n $NAMESPACE rollout status deployment/ecgense --timeout=300s
                    kubectl --context $KUBE_CONTEXT -n $NAMESPACE get pods,svc -o wide

                    # Call the app through the Kubernetes Service
                    kubectl --context $KUBE_CONTEXT -n $NAMESPACE port-forward svc/ecgense 8090:80 > /dev/null 2>&1 &
                    PF_PID=$!
                    trap "kill $PF_PID" EXIT
                    sleep 5
                    curl -fsS http://localhost:8090/health
                    echo
                    curl -fsS -o /dev/null -w "Home page HTTP %{http_code}\\n" http://localhost:8090/
                '''
            }
        }
    }

    post {
        success {
            echo "ECGense build #${env.BUILD_NUMBER} is running. Open it with: kubectl -n ecgense port-forward svc/ecgense 8080:80  ->  http://localhost:8080"
        }
        failure {
            echo 'Pipeline failed. Check the stage logs above.'
        }
    }
}
