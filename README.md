# Teaching AI to Play Pong: Double DQN vs. PPO

A deep reinforcement learning project that builds a custom **Pong environment** with Gymnasium and compares two agents: **Double DQN**, implemented in PyTorch, and **PPO**, trained using Stable-Baselines3.

**University project:** Deep Learning endterm, IT-2406  
**Team:** Abussagitov Bakhtiyar and Aldabergen Assel

## Project overview

- Implemented a custom Gymnasium environment (`reset`, `step`, and RGB rendering) with a scripted opponent.
- Trained a Double DQN agent with an experience replay buffer and target network.
- Trained a PPO agent with Stable-Baselines3.
- Compared agents against baselines over 20 test episodes each.
- Examined DQN discount factors γ = 0.90, 0.99, and 0.999, plus reward shaping.
- Ran a 40-game head-to-head tournament between DQN and PPO.
- Produced learning curves, evaluation tables, and gameplay GIFs.

## Results

In the recorded notebook run, Double DQN won **37 of 40** games against PPO (PPO won 3), with a mean score difference of **+3.35** in DQN's favor. These results are specific to this custom environment, training setup and run; they are not a general claim that DQN outperforms PPO.

## Tech stack

Python, Gymnasium, PyTorch, Stable-Baselines3, NumPy, pandas, Matplotlib, Pillow, Jupyter.

## Run locally

1. Install Python and Jupyter.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `pong_dqn_vs_ppo.ipynb` in Jupyter or Google Colab.
4. Run cells in sequence. Training takes time; the notebook uses 150,000 timesteps for each principal agent and also trains additional DQN variants.

The notebook creates a `results/` directory containing generated figures, evaluation tables, saved models, and GIFs when run.

## Notes

This is a **team project**. The repository preserves the original notebook's team credits. The notebook contains outputs from a previous run; results may vary if retrained.
