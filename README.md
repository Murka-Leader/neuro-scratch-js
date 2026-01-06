# NeuroScratch: Vanilla JS Neural Network

A fundamental implementation of a Multilayer Perceptron (MLP) built entirely with **Vanilla JavaScript**, designed to demonstrate the mathematical foundations of Deep Learning.

## Academic Purpose
This project was developed to explore the underlying mechanics of Artificial Intelligence without the abstraction of libraries like TensorFlow. It serves as a proof-of-concept for:
- **Linear Algebra:** Manual matrix multiplication and transposition.
- **Calculus:** Partial derivatives used in the Backpropagation algorithm.
- **Optimization:** Stochastic Gradient Descent (SGD) for weight adjustment.

## Key Features
- **Manual Backpropagation:** No automatic differentiation; every gradient is calculated via code.
- **Interactive Digit Recognition:** Draw on a 28x28 canvas to test the model's prediction.
- **Weight Visualization:** Real-time rendering of neural connections and activation strengths.

## The Math Behind the Code
The network follows the standard weight update rule:
$$\Delta w = \eta \cdot \delta \cdot x^T$$

## Where:
- $\eta$ is the Learning Rate.
- $\delta$ is the calculated error gradient.
- $x^T$ is the transposed input.

## Tech Stack
- **Language:** JavaScript (ES6+)
- **Visuals:** HTML5 Canvas API
- **Styling:** CSS3 (Custom Neural UI)
