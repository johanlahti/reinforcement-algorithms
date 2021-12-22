# Q-learning algorithm

## Run the code

1. Install the imported dependencies
2. Execute one of these environments:
    - cartpole.py
    - mountain_car.py

```python
python mountain_car.py
```

## Apply algorithm on other environments:

The same code is used for all AI gym environments, but the following configuration can be overridden to suit each specific environment (in [q_learn_linear_func_approx.py](q_learn_linear_func_approx.py)):
- Environment name (e.g. `CartPole-v0`)
- Reward function (allowing to create tailor-made reward functions based on e.g. the state rather than the environment's own reward)
- Hyper parameters
- Other configuration (rendering, plotting)


