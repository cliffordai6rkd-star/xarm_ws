"""Two terminal rows shared by the independent GELLO reset workers."""
import shutil
import sys
import threading
import time


class ResetDisplay:
    def __init__(self, sides, stream=None):
        self.sides = list(sides)
        self.stream = stream if stream is not None else sys.stdout
        self.tty = self.stream.isatty()
        self.lock = threading.Lock()
        self.samples = {side: None for side in self.sides}
        self.states = {side: 'waiting' for side in self.sides}
        self.last_draw = float('-inf')
        self.started = time.monotonic()
        self.closed = False

    def __enter__(self):
        self.stream.write('GELLO reset: J1..Jn; E=actual-command(deg), R=actual-reference(deg); '
                          'I=measured current(mA, XL330), PWM=measured output, NOT torque(Nm).\n'
                          'Narrow terminal: E/R/I/PWM alternate in the same two rows.\n')
        if self.tty:
            self.stream.write('\n' * len(self.sides))
        self.stream.flush()
        return self

    def _line(self, side, width, now):
        prefix = f'{side} {self.states[side]}'
        sample = self.samples[side]
        if sample is None:
            return prefix
        parts = []
        for key, label in (('tracking', 'Edeg'), ('remaining', 'Rdeg'),
                           ('current', 'I_mA'), ('pwm', 'PWM')):
            values = sample[key]
            numbers = ' '.join(f'{x:+.1f}' if key in ('tracking', 'remaining') else f'{x:+d}' for x in values)
            parts.append(f'{label}[{numbers}]')
        line = prefix + ' ' + ' '.join(parts)
        if len(line) >= width:
            # Show every axis for one quantity instead of hiding the last axes.
            line = prefix + ' ' + parts[int(now - self.started) % len(parts)]
        return line[:max(0, width - 1)]

    def _draw(self, now, final=False):
        if not self.tty and not final:
            return
        width = shutil.get_terminal_size((120, 24)).columns if self.tty else 10000
        if self.tty:
            self.stream.write(f'\x1b[{len(self.sides)}A')
        for side in self.sides:
            self.stream.write(('\r\x1b[2K' if self.tty else '') + self._line(side, width, now) + '\n')
        self.stream.flush()
        self.last_draw = now

    def update(self, side, sample):
        with self.lock:
            if self.closed:
                return
            self.samples[side] = sample
            self.states[side] = 'return'
            now = time.monotonic()
            if now - self.last_draw >= 0.25:
                self._draw(now)

    def finish(self, side, state):
        with self.lock:
            if self.closed:
                return
            self.states[side] = state
            self._draw(time.monotonic())

    def __exit__(self, exc_type, *_):
        with self.lock:
            self.closed = True
            if exc_type is not None:
                for side in self.sides:
                    if self.states[side] in ('waiting', 'return'):
                        self.states[side] = 'cancelled'
            self._draw(time.monotonic(), final=True)
