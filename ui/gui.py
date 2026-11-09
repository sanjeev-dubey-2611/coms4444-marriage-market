"""Participant-focused visualizer; player execution remains in the worker process."""

import re
from collections import defaultdict
from collections.abc import Callable

import pygame

from core.engine import Engine, RoundRecord
from core.gui_worker import DisplayState, ParticipantDate, SimulationWorker

WIDTH, HEIGHT = 1360, 880
PAD, HEADER_H, FOOTER_H = 18, 76, 46
BACKDROP = (248, 244, 235)
BACKDROP_DARK = (238, 226, 207)
PANEL = (255, 253, 248)
PANEL_EDGE = (222, 204, 176)
INK, INK_DIM = (43, 34, 26), (83, 66, 49)
GOLD, GOOD, WARN = (133, 66, 18), (20, 112, 62), (151, 78, 0)
ALERT, ALERT_TEXT = (198, 48, 48), (255, 226, 226)
SPEEDS = (1, 2, 4, 8, 16, 32)
FAULT_LOG_LIMIT, FAULT_FLASH_ROUNDS = 4, 3
PROFILE_ROWS, HISTORY_ROWS = 10, 12
PROFILE_ROW_H = 26
HISTORY_ROW_H = 32
SUMMARY_ROWS = 16
SUMMARY_ROW_H = 30
SUMMARY_MODES = ('marriages', 'singles', 'groups')
STRIPE = (249, 243, 232)
_GROUP_FROM_NAME = re.compile(r'(?:Player)?(\d+)$', re.IGNORECASE)


def group_label(name: str) -> str:
	"""Group number/code shown in the summary tab (e.g. Player4 -> 4, GreedyPlayer -> g)."""
	match = _GROUP_FROM_NAME.search(name.replace('_', ''))
	if match:
		return match.group(1)
	lower = name.lower()
	if 'greedy' in lower:
		return 'g'
	if 'random' in lower:
		return 'r'
	return name


def trait_penalty(actual, desired, monotonic):
	gap = max(0, desired - actual) if monotonic else desired - actual
	return gap * gap


def penalty_color(penalty):
	if penalty == 0:
		return GOOD
	# All reds remain dark enough to read on the cream background.
	strength = min(100, penalty) / 100
	return (int(165 - 55 * strength), int(65 - 45 * strength), int(55 - 35 * strength))


def short_choice(decision):
	if decision.incompatible:
		return 'Never again'
	if decision.propose:
		return 'Propose + another date' if decision.again else 'Propose'
	return 'Another date' if decision.again else 'No action'


def decision_text(decision):
	if decision.incompatible:
		return 'Marked the other person incompatible.'
	if decision.propose and decision.again:
		return 'Proposed marriage and asked to meet again.'
	if decision.propose:
		return 'Proposed marriage.'
	if decision.again:
		return 'Asked to meet again.'
	return 'Took no action.'


def pairing_text(how):
	return {
		'repeat': 'Both asked to meet again after their previous date.',
		'mutual': 'Both listed each other before this round.',
		'one-sided': 'Only one listed the other before this round.',
		'random': 'Neither listed the other; the engine paired those left.',
	}[how]


def result_text(entry, ended=False):
	if entry.married:
		return 'Both proposed. They married and left the dating pool.'
	if entry.decision.incompatible or entry.partner_decision.incompatible:
		return 'They are incompatible and will never be paired again.'
	if ended:
		return 'The simulation ended before another date could happen.'
	if entry.again:
		return 'Both asked to meet again. A repeat date was scheduled.'
	if entry.decision.again or entry.partner_decision.again:
		return 'Only one asked again. There is no guaranteed repeat.'
	if entry.decision.propose or entry.partner_decision.propose:
		return 'Only one proposed. No marriage or guaranteed repeat.'
	return 'Neither asked again. Both stay in the dating pool.'


class Visualizer:
	def __init__(
		self,
		engine: Engine | DisplayState,
		on_round: Callable[[RoundRecord], None] | None = None,
		worker: SimulationWorker | None = None,
	):
		self.engine, self.on_round, self.worker = engine, on_round, worker
		self.record = None
		self.playing, self.finished = False, engine.finished
		self.speed_index, self.selected = 1, 0
		self.tab = 'participant'  # or 'summary'
		self.summary_mode = 'marriages'  # marriages | singles | groups
		self.input_focus = 'participant'
		self.person_buffer = self.goto_buffer = ''
		self.person_error = self.goto_error = self.jump_target = None
		self.profile_offset = self.history_offset = self.history_selected = 0
		self.summary_offset = 0
		self.fault_log, self.last_fault_round, self.fonts = [], -10, {}
		self.person_box = pygame.Rect(152, 86, 100, 30)
		self.show_button = pygame.Rect(264, 86, 65, 30)
		self.round_box = pygame.Rect(720, 86, 100, 30)
		self.tab_participant = pygame.Rect(1080, 86, 120, 30)
		self.tab_summary = pygame.Rect(1210, 86, 110, 30)
		self.participant_panel = pygame.Rect(18, 148, 1324, 668)
		self.profile_area = pygame.Rect(34, 286, 310, 510)
		self.history_area = pygame.Rect(380, 350, 946, HISTORY_ROWS * HISTORY_ROW_H)
		self.summary_area = pygame.Rect(34, 268, 1288, SUMMARY_ROWS * SUMMARY_ROW_H)
		self.summary_mode_rects = {
			mode: pygame.Rect(34 + i * 150, 200, 140, 28) for i, mode in enumerate(SUMMARY_MODES)
		}
		self.histories = [[] for _ in range(engine.people)] if isinstance(engine, Engine) else None
		if self.histories is not None:
			for record in engine.records:
				self.remember(record)

	def remember(self, record):
		if self.histories is not None:
			for date in record.dates:
				for person in (date.a, date.b):
					self.histories[person].append(ParticipantDate.from_record(record, date, person))

	def history(self):
		if self.histories is not None:
			return self.histories[self.selected]
		return self.engine.history if self.engine.participant == self.selected else []

	def select_person(self, person):
		if not 0 <= person < self.engine.people:
			self.person_error = f'Use ID 0 to {self.engine.people - 1}'
			return
		changed = person != self.selected
		self.selected = person
		self.person_error = None
		self.person_buffer = ''
		self.profile_offset = self.history_offset = self.history_selected = 0
		if changed and self.worker is not None:
			self.worker.inspect(person)

	def type_key(self, key, unicode):
		if key == pygame.K_t:
			self.tab = 'summary' if self.tab == 'participant' else 'participant'
			return
		if key == pygame.K_m and self.tab == 'summary':
			index = SUMMARY_MODES.index(self.summary_mode)
			self.set_summary_mode(SUMMARY_MODES[(index + 1) % len(SUMMARY_MODES)])
			return
		if key == pygame.K_TAB:
			self.input_focus = 'round' if self.input_focus == 'participant' else 'participant'
			return
		if key in (pygame.K_p, pygame.K_g):
			self.input_focus = 'participant' if key == pygame.K_p else 'round'
			return
		name = 'person_buffer' if self.input_focus == 'participant' else 'goto_buffer'
		value = getattr(self, name)
		if len(unicode) == 1 and unicode in '0123456789' and len(value) < 9:
			setattr(self, name, value + unicode)
		elif key == pygame.K_BACKSPACE:
			setattr(self, name, value[:-1])
		elif key in (pygame.K_RETURN, pygame.K_KP_ENTER) and value:
			if self.input_focus == 'participant':
				self.select_person(int(value))
				self.tab = 'participant'
			else:
				self.jump_to(int(value))
			setattr(self, name, '')
		elif key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
			direction = -1 if key == pygame.K_PAGEUP else 1
			if self.tab == 'summary':
				self.scroll_summary(direction * SUMMARY_ROWS)
				return
			self.scroll_profile(direction * PROFILE_ROWS)

	def select_at(self, pos):
		if self.tab_participant.collidepoint(pos):
			self.tab = 'participant'
			return
		if self.tab_summary.collidepoint(pos):
			self.tab = 'summary'
			return
		if self.person_box.collidepoint(pos):
			self.input_focus = 'participant'
		elif self.round_box.collidepoint(pos):
			self.input_focus = 'round'
		elif self.show_button.collidepoint(pos) and self.person_buffer:
			self.select_person(int(self.person_buffer))
			self.tab = 'participant'
		elif self.tab == 'summary':
			for mode, rect in self.summary_mode_rects.items():
				if rect.collidepoint(pos):
					self.set_summary_mode(mode)
					return
			if self.summary_area.collidepoint(pos):
				index = self.summary_offset + (pos[1] - self.summary_area.y) // SUMMARY_ROW_H
				person = self.summary_person_at(index)
				if person is not None:
					self.select_person(person)
					self.tab = 'participant'
		elif self.tab == 'participant' and self.history_area.collidepoint(pos):
			index = self.history_offset + (pos[1] - self.history_area.y) // HISTORY_ROW_H
			if index < len(self.history()):
				self.history_selected = index

	def scroll_history(self, direction):
		self.history_offset = max(
			0, min(self.history_offset + direction, max(0, len(self.history()) - HISTORY_ROWS))
		)

	def scroll_profile(self, direction):
		self.profile_offset = max(
			0,
			min(
				self.profile_offset + direction,
				max(0, self.engine.dims - PROFILE_ROWS),
			),
		)

	def scroll_summary(self, direction):
		self.summary_offset = max(
			0,
			min(
				self.summary_offset + direction,
				max(0, len(self.summary_rows()) - SUMMARY_ROWS),
			),
		)

	def set_summary_mode(self, mode: str) -> None:
		if mode not in SUMMARY_MODES or mode == self.summary_mode:
			return
		self.summary_mode = mode
		self.summary_offset = 0

	def marriage_summary(self) -> list[tuple[int, str, int, str, float, int | None]]:
		"""Married people as (id, group, partner_id, partner_group, happiness, married_on)."""
		e = self.engine
		rows = []
		for i, partner in enumerate(e.spouse):
			if partner is None:
				continue
			rows.append(
				(
					i,
					group_label(e.player_names[i]),
					partner,
					group_label(e.player_names[partner]),
					e.happiness(i),
					e.married_on[i],
				)
			)
		rows.sort(key=lambda row: (-row[4], row[0]))
		return rows

	def singles_summary(self) -> list[tuple[int, str, int, int, int]]:
		"""Singles as (id, group, dates, proposals_made, proposals_received)."""
		e = self.engine
		rows = [
			(
				i,
				group_label(e.player_names[i]),
				e.dates_had[i],
				e.proposals_made[i],
				e.proposals_received[i],
			)
			for i in e.singles()
		]
		rows.sort(key=lambda row: (row[1], row[0]))
		return rows

	def group_summary(self) -> list[tuple[str, int, int, int, float]]:
		"""Groups as (group, people, married, singles, avg happiness)."""
		e = self.engine
		buckets: dict[str, dict[str, list | int]] = defaultdict(
			lambda: {'happy': [], 'married': 0, 'singles': 0}
		)
		for i, name in enumerate(e.player_names):
			group = group_label(name)
			buckets[group]['happy'].append(e.happiness(i))
			if e.spouse[i] is None:
				buckets[group]['singles'] += 1
			else:
				buckets[group]['married'] += 1
		rows = []
		for group, data in buckets.items():
			happy = data['happy']
			rows.append(
				(
					group,
					len(happy),
					data['married'],
					data['singles'],
					sum(happy) / len(happy),
				)
			)
		rows.sort(key=lambda row: (-row[4], row[0]))
		return rows

	def summary_rows(self):
		if self.summary_mode == 'singles':
			return self.singles_summary()
		if self.summary_mode == 'groups':
			return self.group_summary()
		return self.marriage_summary()

	def summary_person_at(self, index: int) -> int | None:
		rows = self.summary_rows()
		if not 0 <= index < len(rows):
			return None
		if self.summary_mode == 'groups':
			return None
		return rows[index][0]

	def render_key(self):
		return (
			self.engine.round,
			self.selected,
			self.tab,
			self.summary_mode,
			self.playing,
			self.speed_index,
			self.finished,
			self.jump_target,
			self.goto_buffer,
			self.goto_error,
			self.person_buffer,
			self.person_error,
			self.input_focus,
			self.profile_offset,
			self.history_offset,
			self.history_selected,
			self.summary_offset,
			len(self.summary_rows()),
			getattr(self.engine, 'participant', self.selected),
			len(self.history()),
		)

	def text(self, surface, value, pos, name='small', color=None, anchor='topleft', width=None):
		color = INK if color is None else color
		value = str(value)
		font = self.font(name)
		if width is not None and font.size(value)[0] > width:
			while value and font.size(value + '...')[0] > width:
				value = value[:-1]
			value += '...'
		image = font.render(value, True, color)
		rect = image.get_rect()
		setattr(rect, anchor, pos)
		surface.blit(image, rect)
		return rect

	def draw_controls(self, surface):
		self.text(surface, 'Participant ID', (18, 94), color=INK_DIM)
		self.text(surface, 'Skip to round', (584, 94), color=INK_DIM)
		for kind, rect, value in [
			('participant', self.person_box, self.person_buffer or str(self.selected)),
			('round', self.round_box, self.goto_buffer or '#'),
		]:
			pygame.draw.rect(surface, PANEL, rect, border_radius=5)
			pygame.draw.rect(
				surface,
				GOLD if self.input_focus == kind else PANEL_EDGE,
				rect,
				2,
				border_radius=5,
			)
			self.text(surface, value, (rect.x + 8, rect.y + 7), width=rect.w - 16)
		pygame.draw.rect(surface, PANEL_EDGE, self.show_button, border_radius=5)
		self.text(surface, 'Show', self.show_button.center, 'tiny', anchor='center')
		hint = (
			f'Skipping to {self.jump_target} - SPACE cancels'
			if self.jump_target is not None
			else 'Enter to jump'
		)
		self.text(
			surface,
			self.goto_error or hint,
			(834, 94),
			'tiny',
			WARN if self.goto_error else INK_DIM,
			width=230,
		)
		if self.person_error:
			self.text(surface, self.person_error, (18, 123), 'tiny', WARN)
		self.draw_tabs(surface)

	def draw_tabs(self, surface):
		for label, rect, name in (
			('Participant', self.tab_participant, 'participant'),
			('Summary', self.tab_summary, 'summary'),
		):
			active = self.tab == name
			pygame.draw.rect(surface, PANEL if active else BACKDROP_DARK, rect, border_radius=6)
			pygame.draw.rect(
				surface, GOLD if active else PANEL_EDGE, rect, 2 if active else 1, border_radius=6
			)
			self.text(
				surface,
				label,
				rect.center,
				'small',
				GOLD if active else INK_DIM,
				anchor='center',
			)

	def draw_summary(self, surface):
		self.text(surface, 'Summary', (34, 170), 'mid')
		hints = {
			'marriages': 'Married people by happiness · Click a row to open that participant',
			'singles': 'People still single, by group · Click a row to open that participant',
			'groups': 'Per-group averages · Singles count as 0 happiness (same as engine mean)',
		}
		self.text(surface, hints[self.summary_mode], (140, 176), 'small', INK_DIM, width=1100)
		labels = {'marriages': 'Marriages', 'singles': 'Singles', 'groups': 'Groups'}
		for mode, rect in self.summary_mode_rects.items():
			active = self.summary_mode == mode
			pygame.draw.rect(surface, PANEL if active else BACKDROP_DARK, rect, border_radius=6)
			pygame.draw.rect(
				surface, GOLD if active else PANEL_EDGE, rect, 2 if active else 1, border_radius=6
			)
			self.text(
				surface,
				labels[mode],
				rect.center,
				'small',
				GOLD if active else INK_DIM,
				anchor='center',
			)
		self.text(surface, 'M cycles view · T back', (500, 206), 'tiny', INK_DIM)

		if self.summary_mode == 'singles':
			self.draw_singles_summary(surface)
		elif self.summary_mode == 'groups':
			self.draw_group_summary(surface)
		else:
			self.draw_marriage_summary(surface)

	def draw_marriage_summary(self, surface):
		rows = self.marriage_summary()
		columns = [
			(34, 70, 'Player'),
			(120, 90, 'Group'),
			(230, 100, 'Partner'),
			(350, 110, 'Partner group'),
			(480, 140, 'Happiness'),
			(650, 120, 'Married round'),
		]
		self.draw_summary_table(
			surface,
			columns,
			[
				(
					person,
					group,
					partner,
					partner_group,
					f'{score:,.2f}',
					married_on if married_on is not None else '—',
					GOOD if score > 0 else (ALERT if score < 0 else INK),
					480,
				)
				for person, group, partner, partner_group, score, married_on in rows
			],
			empty='No marriages yet.',
		)

	def draw_singles_summary(self, surface):
		rows = self.singles_summary()
		# Compact per-group counts along the right edge of the table header area.
		by_group: dict[str, int] = defaultdict(int)
		for _, group, *_ in rows:
			by_group[group] += 1
		counts = ', '.join(f'{group}: {count}' for group, count in sorted(by_group.items()))
		self.text(
			surface,
			f'Left single by group — {counts}' if counts else 'No singles left.',
			(700, 206),
			'tiny',
			INK_DIM,
			width=600,
		)
		columns = [
			(34, 70, 'Player'),
			(120, 90, 'Group'),
			(230, 100, 'Dates'),
			(350, 140, 'Proposals made'),
			(520, 160, 'Proposals received'),
		]
		self.draw_summary_table(
			surface,
			columns,
			[
				(person, group, dates, made, received, INK, None)
				for person, group, dates, made, received in rows
			],
			empty='Everyone is married.',
		)

	def draw_group_summary(self, surface):
		rows = self.group_summary()
		columns = [
			(34, 90, 'Group'),
			(140, 90, 'People'),
			(250, 100, 'Married'),
			(370, 100, 'Singles'),
			(500, 150, 'Avg happiness'),
		]
		self.draw_summary_table(
			surface,
			columns,
			[
				(
					group,
					people,
					married,
					singles,
					f'{avg:,.2f}',
					GOOD if avg > 0 else (ALERT if avg < 0 else INK),
					500,
				)
				for group, people, married, singles, avg in rows
			],
			empty='No groups yet.',
		)

	def draw_summary_table(self, surface, columns, rows, empty: str):
		for x, width, label in columns:
			self.text(surface, label, (x, self.summary_area.y - 22), 'small', INK_DIM, width=width)
		if not rows:
			self.text(surface, empty, (34, 320), 'mid', INK_DIM)
			return
		visible = rows[self.summary_offset : self.summary_offset + SUMMARY_ROWS]
		for row, entry in enumerate(visible):
			*values, accent, accent_x = entry
			line_y = self.summary_area.y + row * SUMMARY_ROW_H
			pygame.draw.rect(
				surface,
				STRIPE if row % 2 == 0 else PANEL,
				(self.summary_area.x - 3, line_y - 2, self.summary_area.w, SUMMARY_ROW_H),
				border_radius=3,
			)
			for value, (x, width, _) in zip(values, columns, strict=True):
				self.text(
					surface,
					value,
					(x, line_y + 4),
					'small',
					accent if accent_x is not None and x == accent_x else INK,
					width=width,
				)
		if len(rows) > SUMMARY_ROWS:
			self.text(
				surface,
				f'Showing {self.summary_offset + 1}–{min(self.summary_offset + SUMMARY_ROWS, len(rows))} of {len(rows)} · Scroll / Page Up/Down',
				(34, self.summary_area.bottom + 12),
				'tiny',
				INK_DIM,
			)

	def draw(self, surface):
		surface.fill(BACKDROP)
		self.draw_header(surface)
		self.draw_controls(surface)
		pygame.draw.rect(surface, PANEL, self.participant_panel, border_radius=10)
		pygame.draw.rect(surface, PANEL_EDGE, self.participant_panel, 2, border_radius=10)
		if self.tab == 'summary':
			self.draw_summary(surface)
		else:
			self.draw_participant(surface)
		pygame.draw.rect(surface, BACKDROP_DARK, (0, HEIGHT - FOOTER_H, WIDTH, FOOTER_H))
		self.text(
			surface,
			'SPACE play/pause   RIGHT/S step   UP/DOWN speed   T summary   M summary view   P select ID   G skip   Q quit',
			(18, HEIGHT - 32),
			'tiny',
			INK_DIM,
		)
		state = 'finished' if self.finished else ('playing' if self.playing else 'paused')
		self.text(
			surface,
			f'{state} | {self.speed} rounds/s',
			(WIDTH - PAD, HEIGHT - 32),
			'tiny',
			INK_DIM,
			anchor='topright',
		)
		if self.fault_log and self.engine.round - self.last_fault_round < FAULT_FLASH_ROUNDS:
			pygame.draw.rect(surface, ALERT, (0, 144, WIDTH, 18))
			self.text(
				surface,
				self.fault_log[-1][1],
				(18, 146),
				'tiny',
				ALERT_TEXT,
				width=WIDTH - 36,
			)

	def draw_participant(self, surface):
		e, i = self.engine, self.selected
		x = 34
		self.text(surface, f'Participant {i} | {e.player_names[i]}', (x, 170), 'mid')
		partner = e.spouse[i]
		status = (
			f'Married to {partner} in round {e.married_on[i]}' if partner is not None else 'Single'
		)
		self.text(
			surface,
			f'Status: {status}',
			(x, 212),
			color=GOLD if partner is not None else INK,
		)
		score = e.happiness(i)
		self.text(surface, 'Happiness', (1030, 212), color=INK_DIM)
		self.text(
			surface,
			f'{score:,.2f}',
			(1160, 207),
			'mid',
			GOOD if score > 0 else (ALERT if score < 0 else INK),
		)
		self.text(
			surface,
			f'Dates: {e.dates_had[i]}    Proposals made: {e.proposals_made[i]}    Received: {e.proposals_received[i]}',
			(x, 248),
			'small',
			INK_DIM,
		)
		ad = e.ads[i]
		self.text(
			surface,
			f'Advertisement: own trait {ad.own_dim} = {ad.own_value}; wants trait {ad.seek_dim} = {ad.seek_value}',
			(600, 248),
			'small',
			INK_DIM,
			width=720,
		)
		# Keep the static profile compact; most of the panel belongs to dates.
		pygame.draw.line(surface, PANEL_EDGE, (354, 286), (354, 795), 2)
		self.text(surface, 'Profile', (x, 290), 'mid')
		history = list(reversed(self.history()))
		entry = history[min(self.history_selected, len(history) - 1)] if history else None
		married = partner is not None
		context = (
			f'Married partner {partner}'
			if married
			else f'Round {entry.round} · Partner {entry.partner}'
			if entry
			else 'No date yet'
		)
		self.text(surface, context, (x, 320), 'small', INK_DIM)
		# Each pair of adjacent columns compares the value with its target.
		labels = [
			('Trait', ''),
			('Actual', ''),
			('Partner', 'wants') if married else ('Shown', 'by you'),
			('Desired', ''),
			('Partner', 'true') if married else ('Partner', 'seen'),
		]
		cols = [x, 88, 150, 218, 284]
		for (top, bottom), col in zip(labels, cols, strict=True):
			self.text(surface, top, (col, 352), 'tiny', INK_DIM)
			self.text(surface, bottom, (col, 372), 'tiny', INK_DIM)
		for row, dim in enumerate(
			range(self.profile_offset, min(e.dims, self.profile_offset + PROFILE_ROWS))
		):
			y = 400 + row * PROFILE_ROW_H
			actual, desired = e.profiles[i][dim], e.ideals[i][dim]
			other_wants = (
				e.ideals[partner][dim]
				if married
				else (entry.partner_stated[dim] if entry else None)
			)
			shown = e.ideals[partner][dim] if married else (entry.shown[dim] if entry else None)
			seen = e.profiles[partner][dim] if married else (entry.observed[dim] if entry else None)
			own_color = (
				penalty_color(
					trait_penalty(actual if married else shown, other_wants, dim < e.monotonic)
				)
				if other_wants is not None
				else INK
			)
			partner_color = (
				penalty_color(trait_penalty(seen, desired, dim < e.monotonic))
				if seen is not None
				else INK
			)
			values = [
				f'{dim}+' if dim < e.monotonic else str(dim),
				actual,
				shown if shown is not None else '—',
				desired,
				seen if seen is not None else '—',
			]
			colors = [INK, own_color if married else INK, own_color, partner_color, partner_color]
			for value, col, color in zip(values, cols, colors, strict=True):
				self.text(surface, value, (col, y), 'small', color)
		legend_y = 400 + min(PROFILE_ROWS, e.dims) * PROFILE_ROW_H + 18
		self.text(surface, 'Green: zero penalty', (x, legend_y), 'small', GOOD)
		self.text(surface, 'Darker red: larger penalty', (x, legend_y + 24), 'small', penalty_color(60))
		self.text(
			surface, '+ higher is better; others: closer', (x, legend_y + 48), 'tiny', INK_DIM
		)
		if e.dims > PROFILE_ROWS:
			self.text(
				surface,
				'Scroll / Page Up-Down: more traits',
				(x, legend_y + 70),
				'tiny',
				INK_DIM,
			)

		hx = self.history_area.x
		self.text(surface, 'Dating history', (hx, 286), 'mid')
		self.text(
			surface,
			'Newest first · Scroll for older dates · Click to explain',
			(620, 292),
			'small',
			INK_DIM,
		)
		columns = [(hx, 65), (450, 85), (540, 120), (665, 65), (735, 235), (975, 235), (1215, 105)]
		for label, (col, width) in zip(
			['Round', 'Partner', 'Pairing', 'Date #', 'This person', 'Their partner', 'Married'],
			columns,
			strict=True,
		):
			self.text(surface, label, (col, 326), 'small', INK_DIM, width=width)
		history = list(reversed(self.history()))
		for row, entry in enumerate(
			history[self.history_offset : self.history_offset + HISTORY_ROWS]
		):
			line_y = self.history_area.y + row * HISTORY_ROW_H
			selected = self.history_offset + row == self.history_selected
			pygame.draw.rect(
				surface,
				PANEL_EDGE if selected else (STRIPE if row % 2 == 0 else PANEL),
				(hx - 3, line_y - 2, self.history_area.w, HISTORY_ROW_H),
				border_radius=3,
			)
			values = [
				entry.round,
				entry.partner,
				entry.how,
				entry.number,
				short_choice(entry.decision),
				short_choice(entry.partner_decision),
				'Yes' if entry.married else '—',
			]
			for value, (col, width) in zip(values, columns, strict=True):
				self.text(
					surface,
					value,
					(col, line_y + 3),
					'small',
					GOOD if entry.married else INK,
					width=width,
				)
		if not history:
			loading = isinstance(e, DisplayState) and e.participant != i
			self.text(
				surface,
				'Loading history...' if loading else 'No dates recorded.',
				(hx, 370),
				'mid',
				INK_DIM,
			)
		else:
			entry = history[min(self.history_selected, len(history) - 1)]
			self.text(
				surface,
				f'Round {entry.round}, partner {entry.partner}: {pairing_text(entry.how)}',
				(hx, 752),
				'small',
				INK_DIM,
				width=946,
			)
			self.text(
				surface,
				result_text(entry, ended=e.finished and entry.round == e.round),
				(hx, 781),
				'small',
				INK_DIM,
				width=946,
			)

	@property
	def speed(self) -> int:
		return SPEEDS[self.speed_index]

	def advance(self) -> None:
		"""Play one round. The engine owns every rule; this only records what
		came back so the next frame can draw it."""
		if self.worker is not None:
			if not self.finished:
				self.worker.request(self.jump_target)
			return
		record = self.engine.step()
		self.accept_round(record)

	def accept_round(self, record: RoundRecord | None) -> None:
		if record is None:
			self.finished = True
			self.playing = False
			return
		self.record = record
		self.remember(record)
		for fault in record.faults:
			self.fault_log.append((record.round, fault))
			self.last_fault_round = record.round
		del self.fault_log[:-FAULT_LOG_LIMIT]
		if self.on_round is not None:
			self.on_round(record)
		if self.engine.finished:
			self.finished = True
			self.playing = False

	def toggle(self) -> None:
		if self.jump_target is not None:
			if self.worker is not None:
				self.worker.cancel_jump.set()
			self.jump_target = None
			self.playing = False
			return
		if not self.finished:
			self.playing = not self.playing

	def faster(self) -> None:
		self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)

	def slower(self) -> None:
		self.speed_index = max(0, self.speed_index - 1)

	def jump_to(self, target: int) -> None:
		"""Advance until ``target`` rounds have been played, or the run ends."""
		if self.finished:
			self.goto_error = f'simulation finished at round {self.engine.round}'
			return
		if target <= self.engine.round:
			self.goto_error = f'already past round {target}'
			return
		self.goto_error = None
		if self.worker is not None:
			self.playing = False
			self.jump_target = min(target, self.engine.rounds)
			self.advance()
			return
		while not self.finished and self.engine.round < target:
			self.advance()

	def font(self, name: str) -> pygame.font.Font:
		if not self.fonts:
			self.fonts = {
				'big': pygame.font.SysFont('arial,helvetica,dejavusans', 30, bold=True),
				'mid': pygame.font.SysFont('arial,helvetica,dejavusans', 22, bold=True),
				'small': pygame.font.SysFont('arial,helvetica,dejavusans', 18),
				'tiny': pygame.font.SysFont('arial,helvetica,dejavusans', 15),
			}
		return self.fonts[name]

	def draw_header(self, surface: pygame.Surface) -> None:
		engine = self.engine
		pygame.draw.rect(surface, BACKDROP_DARK, pygame.Rect(0, 0, WIDTH, HEADER_H))
		self.text(surface, 'Marriage Market', (PAD, 14), 'big')
		self.text(
			surface,
			f'People: {engine.people}    Singles: {len(engine.singles())}    Married couples: {len(engine.marriages)}',
			(PAD, 50),
			'small',
			INK_DIM,
		)
		self.text(
			surface,
			f'round {engine.round} / {engine.rounds}',
			(WIDTH - PAD, 14),
			'big',
			INK,
			'topright',
		)
		frac = engine.round / engine.rounds
		pygame.draw.rect(surface, PANEL_EDGE, pygame.Rect(0, HEADER_H - 5, WIDTH, 5))
		pygame.draw.rect(surface, GOLD, pygame.Rect(0, HEADER_H - 5, int(WIDTH * frac), 5))


def run_gui(
	engine: Engine | None = None,
	on_round: Callable[[RoundRecord], None] | None = None,
	*,
	worker: SimulationWorker | None = None,
) -> None:
	pygame.init()
	pygame.display.set_caption('COMS W4444 - Marriage Market')
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
	clock = pygame.time.Clock()
	view = Visualizer(engine, on_round=on_round) if engine is not None else None
	carry = 0.0
	frame_key = None

	running = True
	try:
		while running:
			if worker is not None:
				message = worker.poll()
				if message is not None:
					kind, state, record = message
					if view is None:
						view = Visualizer(state, worker=worker)
					else:
						view.engine = state
						view.finished = state.finished
						if kind == 'round':
							view.accept_round(record)
						if state.participant != view.selected:
							worker.inspect(view.selected)
					if view.finished or (
						view.jump_target is not None and state.round >= view.jump_target
					):
						view.jump_target = None
			# The event loop below is intentionally limited to one worker request.
			dt = clock.tick(60) / 1000.0
			for event in pygame.event.get():
				if event.type == pygame.WINDOWEXPOSED:
					frame_key = None
				if event.type == pygame.QUIT or (
					event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)
				):
					running = False
					break
				if view is None:
					continue
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_SPACE:
						view.toggle()
					elif event.key in (pygame.K_RIGHT, pygame.K_s):
						view.advance()
					elif event.key == pygame.K_UP:
						view.faster()
					elif event.key == pygame.K_DOWN:
						view.slower()
					else:
						view.type_key(event.key, event.unicode)
				elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
					view.select_at(event.pos)
				elif event.type == pygame.MOUSEWHEEL:
					if view.tab == 'summary':
						view.scroll_summary(-event.y)
					elif view.profile_area.collidepoint(pygame.mouse.get_pos()):
						view.scroll_profile(-event.y)
					else:
						view.scroll_history(-event.y)
			if not running:
				break
			if view is None:
				if frame_key != 'loading':
					screen.fill(BACKDROP)
					font = pygame.font.Font(None, 32)
					screen.blit(
						font.render('Preparing simulation... Q to quit', True, INK), (PAD, PAD)
					)
					pygame.display.flip()
					frame_key = 'loading'
			else:
				if view.jump_target is not None:
					view.advance()
				elif view.playing:
					carry = min(1.0, carry + dt * view.speed)
					if carry >= 1.0 and (worker is None or not worker.pending):
						carry = 0.0
						view.advance()
				else:
					carry = 0.0
				key = view.render_key()
				if key != frame_key:
					view.draw(screen)
					pygame.display.flip()
					frame_key = key
	finally:
		# Close the window before waiting for any in-flight round to finish.
		pygame.quit()
		if worker is not None:
			worker.close()
