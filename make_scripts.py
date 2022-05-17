import argparse

from pathlib import Path

def script(group, gpu, base_lr, head_lr, dann_lr):
    return f"""CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=100 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_0_100.txt 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=1234 \\
    --group='{group}'\\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_0_1234.txt 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=0 \\
    --experiment_seed=12345 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_0_12345.txt 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=100 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_1_100.txt 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=1234 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_1_1234.txt 2>&1 &
PID4=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=1 \\
    --experiment_seed=12345 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_1_12345.txt 2>&1 &
PID5=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=100 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_100_100.txt 2>&1 &
PID6=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=1234 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_100_1234.txt 2>&1 &
PID7=$!

CUDA_VISIBLE_DEVICES={gpu} \\
python3 -m experiments.run \\
    --device='cuda:0' \\
    --dataset_seed=100 \\
    --experiment_seed=12345 \\
    --group='{group}' \\
    --base_lr={base_lr} \\
    --head_lr={head_lr} \\
    --dann_lr={dann_lr} \\
    > out/logs/{group}_100_12345.txt 2>&1 &
PID8=$!

wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',
        type=str,
        default='digits',
        help='The group of experiments to run.')
    parser.add_argument('--base_lr',
        type=float,
        default=1e-3,
        help='init. lr for base feature extractor.')
    parser.add_argument('--head_lr',
        type=float,
        default=1e-3,
        help='init. lr for classifier head')
    parser.add_argument('--dann_lr',
        type=float,
        default=1e-3,
        help='init. lr for dann components')
    args = parser.parse_args()
    loc = f'scripts/groups'
    Path(loc).mkdir(parents=True, exist_ok=True)
    for gpu in [0, 1]:
        with open(f'{loc}/{args.group}-gpu={gpu}.sh', 'w') as out:
            out.write(script(args.group, gpu, args.base_lr,
                args.head_lr, args.dann_lr))
    # make some directories that experiments expects to be there
    Path(f'out/logs').mkdir(parents=True, exist_ok=True)
    Path(f'out/results').mkdir(parents=True, exist_ok=True)
