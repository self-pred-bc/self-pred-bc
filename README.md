## Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning 
[![arXiv](https://img.shields.io/badge/arXiv-2408.16228-df2a2a.svg)](https://arxiv.org/abs/2506.10137)
[![License: MIT](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/licenses/by/1.0/)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://self-pred-bc.github.io/)

<hr style="border: 2px solid gray;"></hr>

Code "Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning (BYOL-$`\gamma`$)
"
## Instructions
### Dataset
Download a dataset, e.g. antmaze-medium-stitch
```bash
python3 download_datasets.py --dataset-name=antmaze-medium-stitch-v0
```
### Train policies
We provide two implementations for doing BC with auxilliary BYOL-$`\gamma`$ in agents/byol.py and agents/byol_min.py, where the former is more customizable and the latter is a more clear reference point. We also provide an implementation for TDSR representation learning (FB-like loss for data collecting policies).



#### antmaze-medium
**byol_min**:
```bash
# byol_min
python3 main.py --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 --eval_episodes=50 --video_episodes=0 --agent=agents/byol_min.py --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.99 --agent.pred_loss_type=bdino --agent.pred_backwards=False --agent.pred_both=True --agent.action_forward=True --agent.use_obs_latent_dim=True --agent.value_latent_dim=64 --seed=0 --env_name=antmaze-medium --dataset_path=[path]/antmaze-medium-stitch-v0.npz --agent.alignment=6

# byol
python3 main.py --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 --eval_episodes=50 --video_episodes=0 --agent=agents/byol.py --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.99 --agent.pred_loss_type=bdino --agent.pred_backwards=False --agent.pred_both=True --agent.policy_repr=phi__phi --agent.action_forward=True --agent.use_obs_latent_dim=True --agent.value_latent_dim=64 --seed=0 --env_name=antmaze-medium --dataset_path=[path]/antmaze-medium-stitch-v0.npz  --agent.alignment=6

# tdsr
python3 main.py --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 --eval_episodes=50 --video_episodes=0 --agent=agents/tdsr.py --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.normalize_psi=False --agent.ortho_coef=0 --agent.discount=0.99  --agent.n_step=1 --agent.action_forward=True --seed=0 --env_name=antmaze-medium --dataset_path=[path]/antmaze-medium-stitch-v0.npz --agent.alignment=0.005
```

### other environments
We provide configurations for other environments in hyperparameters.sh


### Credit
Our code is based on <https://github.com/vivekmyers/tra-ogbench> and <https://github.com/seohongpark/ogbench>.

### Citation

```bibtex
@misc{lawson2025selfpredictiverepresentationscombinatorialgeneralization,
      title={Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning}, 
      author={Daniel Lawson and Adriana Hugessen and Charlotte Cloutier and Glen Berseth and Khimya Khetarpal},
      year={2025},
      eprint={2506.10137},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10137}, 
}
```

