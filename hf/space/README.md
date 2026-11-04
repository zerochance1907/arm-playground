---
title: Arm Playground
emoji: 🦾
colorFrom: red
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
models:
- {{HF_USER}}/arm-playground
short_description: A PPO-trained robot arm running live in your browser
tags:
- reinforcement-learning
- robotics
- ppo
- imitation-learning
- threejs
- robot-arm
---

# Arm Playground

A 4-joint robot arm driven by a small neural network trained with PPO in PyTorch, running
live in your browser with plain JavaScript and three.js. No ML library, no server.

- **Reach:** drag the orange target and the arm chases it.
- **Push:** put a puck and a goal on the table and the arm pushes the puck home.
- Shove it mid-motion and it recovers. The side panel shows all 256 hidden neurons live.

Model weights (safetensors + ONNX) and the model card: [{{HF_USER}}/arm-playground](https://huggingface.co/{{HF_USER}}/arm-playground)
· Source and training code: [github.com/zerochance1907/arm-playground](https://github.com/zerochance1907/arm-playground)
