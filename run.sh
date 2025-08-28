# python notebooks/lion_tuning.py --scenario projectile --sampling halton && \
# python notebooks/adam_tuning.py --scenario projectile --sampling halton && \
# python notebooks/lion_tuning.py --scenario projectile --sampling sobol && \
# python notebooks/adam_tuning.py --scenario projectile --sampling sobol && \
# python notebooks/lion_tuning.py --scenario projectile --sampling mc && \
# python notebooks/adam_tuning.py --scenario projectile --sampling mc && \
# python notebooks/lion_tuning.py --scenario simsin6d --sampling halton && \
# python notebooks/adam_tuning.py --scenario simsin6d --sampling halton && \
# python notebooks/lion_tuning.py --scenario simsin6d --sampling sobol && \
# python notebooks/adam_tuning.py --scenario simsin6d --sampling sobol && \
# python notebooks/lion_tuning.py --scenario simsin6d --sampling mc && \
# python notebooks/adam_tuning.py --scenario simsin6d --sampling mc

python notebooks/lion_tuning.py --scenario simsin8d --sampling halton && \
python notebooks/adam_tuning.py --scenario simsin8d --sampling halton && \
python notebooks/lion_tuning.py --scenario simsin8d --sampling sobol && \
python notebooks/adam_tuning.py --scenario simsin8d --sampling sobol && \
python notebooks/lion_tuning.py --scenario simsin8d --sampling mc && \
python notebooks/adam_tuning.py --scenario simsin8d --sampling mc && \

python notebooks/lion_tuning.py --scenario simsin10d --sampling halton && \
python notebooks/adam_tuning.py --scenario simsin10d --sampling halton && \
python notebooks/lion_tuning.py --scenario simsin10d --sampling sobol && \
python notebooks/adam_tuning.py --scenario simsin10d --sampling sobol && \
python notebooks/lion_tuning.py --scenario simsin10d --sampling mc && \
python notebooks/adam_tuning.py --scenario simsin10d --sampling mc