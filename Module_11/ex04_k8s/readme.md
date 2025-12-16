# Exercise 04: Kubernetes (K8s) Basics

## Goal
Deploy your containerized application (from Ex02) to a local Kubernetes cluster using manifests.

## Learning Objectives
1.  **Deployment:** Create a `deployment.yaml` to manage pods.
2.  **Service:** Create a `service.yaml` to expose pods to the network.
3.  **ReplicaSet:** Understand how K8s scales applications by changing `replicas`.
4.  **Minikube/Kind:** Learn how to use local clusters.

## Practical Motivation
Docker runs one container. Kubernetes runs thousands, across many machines, handles crashes, restarts them, and balances traffic. It's the "Operating System" of the Cloud.

**Analogy:**
*   **Docker:** A single musician playing a violin.
*   **Kubernetes:** The Conductor of a massive orchestra. The conductor ensures there are exactly 4 violinists (replicas). If one faints (crashes), the conductor immediately signals a replacement to start playing. The conductor also tells the audience (traffic) where to look.

## Theory: Manifests
Kubernetes is "declarative". You don't say "start a pod". You say "I want 3 pods running this image", and K8s makes it happen.
*   **Deployment:** Defines the "what" (Image, Ports, Replicas).
*   **Service:** Defines the "access" (LoadBalancer, NodePort, ClusterIP).

## Step-by-Step Instructions

### Task 1: Deployment Manifest
1.  Create `deployment.yaml`.
2.  Define `apiVersion: apps/v1`, `kind: Deployment`.
3.  Set `replicas: 3`.
4.  In `spec.template.spec.containers`, use the image name from Ex02 (e.g., `my_cpp_app:latest`).
    *   *Note:* In Minikube, you might need to run `eval $(minikube docker-env)` so it sees your local images.

### Task 2: Service Manifest
1.  Create `service.yaml`.
2.  Define `apiVersion: v1`, `kind: Service`.
3.  Set `type: NodePort` (easiest for local testing).
4.  Match the `selector` to your Deployment's labels.

### Task 3: Deploy
1.  `kubectl apply -f deployment.yaml`
2.  `kubectl apply -f service.yaml`
3.  `kubectl get pods` (Watch them start).
4.  `minikube service my-cpp-service` (Opens the app in browser/terminal).

## Code Hints
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpp-app-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cpp-app
  template:
    metadata:
      labels:
        app: cpp-app
    spec:
      containers:
      - name: cpp-app
        image: my_cpp_app:latest
        imagePullPolicy: Never # Important for local Minikube
```

## Verification
`kubectl get pods` should show 2 (or 3) "Running" pods.
