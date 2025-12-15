# Exercise 04: Large-Scale Containerization (Kubernetes)

## Goal
Prepare your Dockerized CV service for deployment on Kubernetes.

## Learning Objectives
1.  **Pod & Deployment:** Define a Deployment YAML.
2.  **Service:** Expose the application via a Service.
3.  **Scaling:** Replicas and auto-scaling.

## Practical Motivation
Running one container is easy. Running 100 requires an orchestrator.

## Step-by-Step Instructions
1.  Write `deployment.yaml`.
2.  Define `replicas: 3`.
3.  Apply with `kubectl apply -f deployment.yaml` (requires Minikube or similar).

## Verification
`kubectl get pods` should show 3 running instances.
