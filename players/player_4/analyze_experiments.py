"""Average score per team across every CSV in players/player_4/experiment_logs/.

From the repo root:

    uv run python -m players.player_4.analyze_experiments
"""

import csv
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path(__file__).parent / 'experiment_logs'


def load_scores(log_dir: Path = LOG_DIR) -> dict[str, list[float]]:
	"""Every player's score, grouped by team code."""
	scores_by_team: dict[str, list[float]] = defaultdict(list)
	for path in sorted(log_dir.glob('*.csv')):
		with path.open(newline='') as file:
			# Drop the '#' parameter comment line before handing rows to csv.
			rows = csv.DictReader(line for line in file if not line.startswith('#'))
			for row in rows:
				scores_by_team[row['team']].append(float(row['score']))
	return scores_by_team


def team_sort_key(team: str) -> tuple[int, int | str]:
	# Numbered groups first in numeric order, then greedy / random.
	return (0, int(team)) if team.isdigit() else (1, team)


def main() -> None:
	scores_by_team = load_scores()
	if not scores_by_team:
		print(f'no CSV logs found in {LOG_DIR}')
		return
	print(f'{"team":>6}  {"players":>7}  {"avg score":>9}')
	for team in sorted(scores_by_team, key=team_sort_key):
		scores = scores_by_team[team]
		print(f'{team:>6}  {len(scores):>7}  {sum(scores) / len(scores):>9.2f}')


if __name__ == '__main__':
	main()
