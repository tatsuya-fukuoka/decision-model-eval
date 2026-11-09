"""ベースラインの件ごとの予測を results/raw/ に書く（モデル推論は行わない）。

    uv run python -m src.run_baselines

- baseline-majority：S1MB・bench_ja ともに評価セットの事前分布（src/baselines.py）
- baseline-random：S1MB は一様分布、bench_ja は sokudan の RandomBaseline と同じ（seed 20260920）
"""

from .baselines import majority_rows, sokudan_random_rows, uniform_rows
from .data import bench_ja, s1mb
from .results import path_for, write_rows


def main() -> None:
    for t in s1mb.load_category():
        _, _, targets = s1mb.load_benchmark(t)
        write_rows(path_for("baseline-majority", "s1mb", t), majority_rows(targets))
        write_rows(path_for("baseline-random", "s1mb", t), uniform_rows(targets))
    tasks = {k: bench_ja.load_task(k)[1] for k in bench_ja.TASKS}
    for k, ts in tasks.items():
        write_rows(path_for("baseline-majority", "bench_ja", k), majority_rows(ts))
    for k, rows in sokudan_random_rows(tasks).items():
        write_rows(path_for("baseline-random", "bench_ja", k), rows)
    print("wrote results/raw/baseline-{majority,random}/")


if __name__ == "__main__":
    main()
