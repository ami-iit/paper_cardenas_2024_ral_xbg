<h1 align="center">
  XBG: End-to-end Imitation Learning for Autonomous Behaviour in Human-Robot Interaction and Collaboration
</h1>


<div align="center">


C. Cardenas-Perez, G. Romualdi, M. Elobaid, S. Dafarra, G. L'Erario, S. Traversaro, P. Morerio, A. Del Bue and D. Pucci. 

</div>


<p>
 <source src="https://github.com/ami-iit/xbg/blob/main/assets/videos/xbg_video.mov">
</p>
 


<div align="center">
  IEEE Robotics and Automation Letters
</div>

<div align="center">
  <a href=""><b>Installation</b></a> |
  <a href="https://arxiv.org/pdf/2406.15833"><b>Paper</b></a> |
  <a href="https://www.youtube.com/watch?v=zuFNEG62y6I"><b>Experiments Video</b></a> |
  <a href="https://ami-iit.github.io/xbg/"><b>Website</b></a>
</div>

### Abstract

This paper presents XBG (eXteroceptive Behaviour Generation), a multimodal end-to-end Imitation Learning (IL) system for a whole-body autonomous humanoid robot used in real-world Human-Robot Interaction (HRI) scenarios. The main contribution of this paper is an architecture for learning HRI behaviors using a data-driven approach. Through teleoperation, a diverse dataset is collected, comprising demonstrations across multiple HRI scenarios, including handshaking, handwaving, payload reception, walking, and walking with a payload. After synchronizing, filtering, and transforming the data, different Deep Neural Networks (DNN) models are trained. The final system integrates different modalities comprising exteroceptive and proprioceptive sources of information to provide the robot with an understanding of its environment and its own actions. The robot takes sequence of images (RGB and depth) and joints state information during the interactions and then reacts accordingly, demonstrating learned behaviors. By fusing multimodal signals in time, we encode new autonomous capabilities into the robotic platform, allowing the understanding of context changes over time. The models are deployed on ergoCub, a real-world humanoid robot, and their performance is measured by calculating the success rate of the robot's behavior under the mentioned scenarios.

### Installation

In order to reproduce the results related to this work, please configure your environment using either the requirements.txt file or the Pipfiles provided.


### Citing this work

If you find the work useful, please consider citing:

```bibtex
@ARTICLE{,
  author={Cardenas-Perez, Carlos and Romualdi, Giulio and Elobaid, Mohammed and Dafarra, Stefano and L'erario, Giuseppe and Traversaro, Silvio and Morerio, Pietro and Del Bue, Alessio and Pucci, Daniele},
  journal={Arxiv},
  title={},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}
```

### Maintainer

This repository is maintained by:

| |                                                        |
|:---:|:------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/44415073?s=400&u=d251a0443d6444920cf640f13e86f549269b25f3&v=4" width="40">](https://github.com/GitHubUserName) | [@carloscp3009](https://github.com/carloscp3009) |
