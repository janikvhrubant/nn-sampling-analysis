python notebooks/lion_tuning.py --scenario projectile --sampling halton && \
python notebooks/adam_tuning.py --scenario projectile --sampling halton && \
python notebooks/lion_tuning.py --scenario projectile --sampling sobol && \
python notebooks/adam_tuning.py --scenario projectile --sampling sobol && \
python notebooks/lion_tuning.py --scenario projectile --sampling mc && \
python notebooks/adam_tuning.py --scenario projectile --sampling mc && \
python notebooks/lion_tuning.py --scenario sumsin6d --sampling halton && \
python notebooks/adam_tuning.py --scenario sumsin6d --sampling halton && \
python notebooks/lion_tuning.py --scenario sumsin6d --sampling sobol && \
python notebooks/adam_tuning.py --scenario sumsin6d --sampling sobol && \
python notebooks/lion_tuning.py --scenario sumsin6d --sampling mc && \
python notebooks/adam_tuning.py --scenario sumsin6d --sampling mc

# python notebooks/lion_tuning.py --scenario sumsin8d --sampling halton && \
# python notebooks/adam_tuning.py --scenario sumsin8d --sampling halton && \
# python notebooks/lion_tuning.py --scenario sumsin8d --sampling sobol && \
# python notebooks/adam_tuning.py --scenario sumsin8d --sampling sobol && \
# python notebooks/lion_tuning.py --scenario sumsin8d --sampling mc && \
# python notebooks/adam_tuning.py --scenario sumsin8d --sampling mc && \

# python notebooks/lion_tuning.py --scenario sumsin10d --sampling halton && \
# python notebooks/adam_tuning.py --scenario sumsin10d --sampling halton && \
# python notebooks/lion_tuning.py --scenario sumsin10d --sampling sobol && \
# python notebooks/adam_tuning.py --scenario sumsin10d --sampling sobol && \
# python notebooks/lion_tuning.py --scenario sumsin10d --sampling mc && \
# python notebooks/adam_tuning.py --scenario sumsin10d --sampling mc