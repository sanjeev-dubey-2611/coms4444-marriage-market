"""Headless checks for the visualiser and the leaderboard.

The GUI is the thing the TA runs on a projector in front of the class, so the
failure that matters is not a wrong pixel, it is a crash partway through a
run. These drive the real render path under SDL's dummy driver.
"""

import os
import random

import pytest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

pygame = pytest.importorskip('pygame')

from core.engine import Engine  # noqa: E402
from players.greedy_player import GreedyPlayer  # noqa: E402
from players.random_player import RandomPlayer  # noqa: E402
from tests.fixtures.scripted import scripted  # noqa: E402
from ui.gui import HEIGHT, WIDTH, Visualizer  # noqa: E402


@pytest.fixture
def screen():
	pygame.init()
	surface = pygame.display.set_mode((WIDTH, HEIGHT))
	yield surface
	pygame.quit()


def build(**kwargs) -> Engine:
	random.seed(1)
	defaults = dict(
		players=[GreedyPlayer] * 10 + [RandomPlayer] * 10, rounds=100, seed=4444, timeout=0
	)
	defaults.update(kwargs)
	return Engine(**defaults)


def test_renders_a_full_run_without_crashing(screen):
	engine = build()
	view = Visualizer(engine)
	view.draw(screen)
	for _ in range(engine.rounds + 5):
		view.advance()
		view.draw(screen)
	assert view.finished
	assert not view.playing


def test_renders_a_large_market(screen):
	view = Visualizer(build(players=[GreedyPlayer] * 60))
	view.selected = 59
	for _ in range(5):
		view.advance()
		view.draw(screen)


def test_fault_banner_renders(screen):
	broken = scripted('Broken', project=lambda self, date: 'nope')
	view = Visualizer(build(players=[broken] * 2 + [GreedyPlayer] * 2))
	view.advance()
	assert view.fault_log
	view.draw(screen)


def test_participant_id_selects_person(screen):
	view = Visualizer(build())
	view.type_key(pygame.K_p, 'p')
	view.type_key(0, '7')
	view.type_key(pygame.K_RETURN, '\r')
	assert view.selected == 7
	view.draw(screen)
	view.type_key(0, '8')
	view.select_at(view.show_button.center)
	assert view.selected == 8
	view.select_person(7)
	view.select_person(-1)
	assert view.selected == 7 and view.person_error
	view.select_person(view.engine.people)
	assert view.selected == 7 and view.person_error


def test_jump_skips_forward_but_never_back():
	view = Visualizer(build(players=[scripted()] * 4))
	view.jump_to(25)
	assert view.engine.round == 25
	view.jump_to(10)
	assert view.engine.round == 25
	assert view.goto_error


def test_typing_a_round_number_jumps():
	view = Visualizer(build())
	view.type_key(pygame.K_g, 'g')
	for ch in '12':
		view.type_key(0, ch)
	view.type_key(pygame.K_RETURN, '\r')
	assert view.engine.round == 12
	assert view.goto_buffer == ''


def test_speed_controls_clamp():
	view = Visualizer(build())
	for _ in range(20):
		view.faster()
	top = view.speed
	view.faster()
	assert view.speed == top
	for _ in range(20):
		view.slower()
	assert view.speed == 1


def test_on_round_sees_every_round():
	seen = []
	view = Visualizer(build(rounds=7), on_round=lambda record: seen.append(record.round))
	view.jump_to(100)
	assert seen == list(range(1, 8))


def test_skip_input_ignores_nondecimal_digits():
	view = Visualizer(build())
	view.type_key(0, '²')
	view.type_key(pygame.K_RETURN, '\r')
	assert view.engine.round == 0
	assert view.goto_buffer == ''


def test_skip_input_backspace_and_length_limit():
	view = Visualizer(build())
	view.input_focus = 'round'
	for digit in '1234567890':
		view.type_key(0, digit)
	assert view.goto_buffer == '123456789'
	view.type_key(pygame.K_BACKSPACE, '')
	assert view.goto_buffer == '12345678'


@pytest.mark.parametrize('quit_key', [pygame.K_q, pygame.K_ESCAPE, None])
def test_real_event_loop_controls_and_partial_log(tmp_path, monkeypatch, quit_key):
	import json

	from core.runlog import RunLog, bind
	from ui.gui import run_gui

	engine = build(rounds=20)
	path = tmp_path / 'gui.log'
	log = RunLog.create(str(path), engine)

	def key(k, u=''):
		return pygame.event.Event(pygame.KEYDOWN, key=k, unicode=u)

	closing = pygame.event.Event(pygame.QUIT) if quit_key is None else key(quit_key)
	frames = iter(
		[
			[
				key(pygame.K_RIGHT),
				key(pygame.K_UP),
				key(pygame.K_DOWN),
				pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(760, 100)),
			],
			[key(pygame.K_3, '3'), key(pygame.K_RETURN, '\r')],
			[key(pygame.K_SPACE)],
			[closing],
		]
	)
	monkeypatch.setattr(pygame.event, 'get', lambda: next(frames, [closing]))
	run_gui(engine, on_round=bind(log, engine))
	log.finish(engine)
	assert engine.round == 3
	text = path.read_text()
	assert text.count('\nround ') == 3
	assert json.loads(text.split('========\n', 1)[1]) == engine.results()
	assert not pygame.get_init()


def test_leaderboard_draws_and_closes():
	from ui.leaderboard import Leaderboard

	board = Leaderboard(title='t', rank_by='positive_rate', total=2, entrants=['g', 'r'])
	row = {
		'code': 'g',
		'rank': 1,
		'positive_rate': {'mean': 0.8},
		'mean_happiness': {'mean': 40.0},
		'married_rate': {'mean': 0.9},
	}
	board.update([row], 1, ['r'], last_match='gx6  seed 1')
	board.update([row], 2, [])
	board.wait_to_close()
	assert not board.open


def test_worker_gui_keeps_processing_controls_during_slow_round(tmp_path, monkeypatch):
	import time

	from core.gui_worker import SimulationWorker
	from tests.fixtures.worker_players import SlowPlayer
	from ui.gui import run_gui

	worker = SimulationWorker(
		dict(players=[SlowPlayer] * 2, rounds=3, timeout=0), str(tmp_path / 'responsive.log')
	)
	position = Visualizer(Engine([SlowPlayer] * 2, rounds=3, timeout=0)).person_box.center
	draws = []
	original_draw = Visualizer.draw

	def draw(self, surface):
		draws.append((self.engine.round, self.selected, self.speed, self.jump_target))
		original_draw(self, surface)

	monkeypatch.setattr(Visualizer, 'draw', draw)
	started = False
	pending_frames = 0
	deadline = time.monotonic() + 15

	def key(k, u=''):
		return pygame.event.Event(pygame.KEYDOWN, key=k, unicode=u)

	def events():
		nonlocal started, pending_frames
		assert time.monotonic() < deadline
		if worker.ready and not started:
			started = True
			return [key(pygame.K_RIGHT)]
		if worker.pending:
			pending_frames += 1
			if pending_frames == 1:
				return [key(pygame.K_g, 'g'), key(pygame.K_2, '2'), key(pygame.K_RETURN, '\r')]
			if pending_frames == 2:
				return [
					key(pygame.K_SPACE),
					key(pygame.K_UP),
					pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=position),
				]
		if started and not worker.pending:
			return [pygame.event.Event(pygame.QUIT)]
		return []

	monkeypatch.setattr(pygame.event, 'get', events)
	try:
		run_gui(worker=worker)
	finally:
		worker.close()
	assert pending_frames >= 5
	assert (0, 0, 4, None) in draws  # controls handled before the round finishes
	assert not worker.process.is_alive()
	import json

	result = json.loads((tmp_path / 'responsive.log').read_text().split('========\n', 1)[1])
	assert result['rounds_played'] == 1  # canceling skip did not queue round 2


def test_event_loop_reuses_unchanged_frames(tmp_path, monkeypatch):
	from ui.gui import run_gui

	drawn = []
	original = Visualizer.draw

	def draw(self, surface):
		drawn.append(self.render_key())
		original(self, surface)

	monkeypatch.setattr(Visualizer, 'draw', draw)
	frames = iter(
		[
			[],
			[],
			[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, unicode='')],
			[],
			[pygame.event.Event(pygame.QUIT)],
		]
	)
	monkeypatch.setattr(pygame.event, 'get', lambda: next(frames))
	run_gui(build())
	assert len(drawn) == 2


def test_all_final_happiness_values_are_explicit(screen, monkeypatch):
	from tests.fixtures.scripted import propose_when_allowed

	for player, rounds in ((scripted(), 1), (scripted(decide=propose_when_allowed), 2)):
		engine = build(players=[player] * 4, rounds=rounds)
		engine.run()
		view = Visualizer(engine)
		written = []
		original = view.text

		def text(surface, value, *args, written=written, original=original, **kwargs):
			written.append(str(value))
			return original(surface, value, *args, **kwargs)

		monkeypatch.setattr(view, 'text', text)
		for person in range(engine.people):
			view.select_person(person)
			written.clear()
			view.draw(screen)
			assert f'{engine.happiness(person):,.2f}' in written
			assert 'Happiness' in written


def test_profile_and_history_navigation(screen):
	view = Visualizer(build(players=[scripted()] * 4, dims=30, rounds=16))
	view.jump_to(16)
	assert len(view.history()) == 16
	view.scroll_history(100)
	assert view.history_offset == 4
	view.select_at((view.history_area.x + 10, view.history_area.y + 10))
	assert view.history_selected == 4
	view.type_key(pygame.K_PAGEDOWN, '')
	assert view.profile_offset == 10
	view.draw(screen)
	view.select_person(1)
	assert view.profile_offset == view.history_offset == 0


def test_date_story_explains_actions_and_pairing():
	from types import SimpleNamespace

	from models.player import Decision
	from ui.gui import decision_text, pairing_text, result_text

	assert decision_text(Decision()) == 'Took no action.'
	assert 'Proposed marriage' in decision_text(Decision(again=True, propose=True))
	assert 'incompatible' in decision_text(Decision(incompatible=True))
	assert 'Both listed' in pairing_text('mutual')
	assert 'Only one listed' in pairing_text('one-sided')
	assert 'Neither listed' in pairing_text('random')
	assert 'previous date' in pairing_text('repeat')
	entry = SimpleNamespace(
		married=False, again=False, decision=Decision(), partner_decision=Decision(again=True)
	)
	assert 'Only one asked again' in result_text(entry)
	assert 'simulation ended' in result_text(entry, ended=True)
	entry.partner_decision = Decision(propose=True)
	assert 'Only one proposed' in result_text(entry)
	entry.partner_decision = Decision(incompatible=True)
	assert 'never be paired' in result_text(entry)
	entry.married = True
	assert 'married and left' in result_text(entry)


def test_trait_penalty_colors():
	from ui.gui import GOOD, penalty_color, trait_penalty

	assert trait_penalty(8, 4, True) == 0
	assert trait_penalty(2, 4, True) == 4
	assert trait_penalty(8, 4, False) == 16
	assert penalty_color(0) == GOOD
	assert penalty_color(1) != penalty_color(100)
	assert penalty_color(100) == penalty_color(200)


def test_group_label_from_player_name():
	from ui.gui import group_label

	assert group_label('Player4') == '4'
	assert group_label('Player10') == '10'
	assert group_label('GreedyPlayer') == 'g'
	assert group_label('RandomPlayer') == 'r'


def test_summary_tab_lists_married_by_happiness(screen):
	view = Visualizer(build())
	view.jump_to(40)
	view.tab = 'summary'
	rows = view.marriage_summary()
	assert rows
	assert all(partner is not None for _, _, partner, _, _, _ in rows)
	assert all(partner_group for _, _, _, partner_group, _, _ in rows)
	happiness = [score for *_, score, _ in rows]
	assert happiness == sorted(happiness, reverse=True)
	view.draw(screen)
	view.type_key(pygame.K_t, 't')
	assert view.tab == 'participant'
	view.type_key(pygame.K_t, 't')
	assert view.tab == 'summary'
	view.select_at(view.tab_participant.center)
	assert view.tab == 'participant'


def test_summary_singles_and_group_averages(screen):
	view = Visualizer(build(rounds=5))
	view.jump_to(5)
	view.tab = 'summary'
	view.set_summary_mode('singles')
	singles = view.singles_summary()
	assert {person for person, *_ in singles} == set(view.engine.singles())
	assert singles == sorted(singles, key=lambda row: (row[1], row[0]))
	view.draw(screen)
	view.type_key(pygame.K_m, 'm')
	assert view.summary_mode == 'groups'
	groups = view.group_summary()
	assert groups
	assert groups == sorted(groups, key=lambda row: (-row[4], row[0]))
	by_label = {
		group: (people, married, left, avg) for group, people, married, left, avg in groups
	}
	# GreedyPlayer -> 'g', RandomPlayer -> 'r' in the default build roster.
	assert by_label['g'][0] == 10
	assert by_label['r'][0] == 10
	assert by_label['g'][1] + by_label['g'][2] == 10
	assert by_label['r'][1] + by_label['r'][2] == 10
	view.draw(screen)
	view.select_at(view.summary_mode_rects['marriages'].center)
	assert view.summary_mode == 'marriages'


def test_profile_shows_all_ten_dimensions(screen):
	view = Visualizer(build(dims=10))
	assert view.engine.dims == 10
	written = []
	original = view.text

	def capture(surface, value, *args, **kwargs):
		written.append(str(value))
		return original(surface, value, *args, **kwargs)

	view.text = capture
	view.draw(screen)
	assert '0' in written and '9' in written
	view.scroll_profile(1)
	assert view.profile_offset == 0


def test_profile_switches_from_date_to_true_spouse(screen):
	view = Visualizer(build())
	view.advance()
	written = []
	original = view.text

	def capture(surface, value, *args, **kwargs):
		written.append(str(value))
		return original(surface, value, *args, **kwargs)

	view.text = capture
	view.draw(screen)
	assert 'Shown' in written and 'seen' in written
	assert 'Round 1 · Partner 17' in written
	view.jump_to(20)
	written.clear()
	view.draw(screen)
	assert 'Married partner 12' in written
	assert 'wants' in written and 'true' in written
	assert 'Shown' not in written
