# ./.venv/bin/python cleanrl_ppo.py --capture-video --no-anneal-lr --learning-rate 0.0003 --total-timesteps 200000 --num-envs 1
python /app/confrontation_ppo.py --no-anneal-lr --save-checkpoints --learning-rate 0.0003 --total-timesteps 200000 --num-envs 1
