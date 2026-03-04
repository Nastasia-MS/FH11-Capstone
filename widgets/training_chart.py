from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont


class TrainingChartWidget(QWidget):
    """Custom widget to draw training/validation line charts."""

    def __init__(self, title="Chart", y_label="Value", cap_at_one=None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 250)
        self.title = title
        self.y_label = y_label
        # cap_at_one=True  -> y-axis always 0..1  (accuracy)
        # cap_at_one=False -> y-axis scales to data max (loss)
        # cap_at_one=None  -> auto-detect: cap if y_label contains "Acc"
        if cap_at_one is None:
            self._cap = "acc" in y_label.lower()
        else:
            self._cap = bool(cap_at_one)

        self.training_data   = []
        self.validation_data = []

        self.margin_left   = 58   # room for y-axis labels
        self.margin_right  = 16
        self.margin_top    = 28
        self.margin_bottom = 38   # room for x-axis labels

        self.training_color   = QColor(59,  130, 246)   # blue
        self.validation_color = QColor(251, 146,  60)   # orange

    # ------------------------------------------------------------------ data
    def add_data_point(self, training_value, validation_value):
        self.training_data.append(training_value)
        self.validation_data.append(validation_value)
        self.update()

    def clear_data(self):
        self.training_data   = []
        self.validation_data = []
        self.update()

    # ------------------------------------------------------------ paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Use the widget palette so we respect light/dark themes
        bg    = self.palette().window().color()
        fg    = self.palette().windowText().color()
        grid  = QColor(fg.red(), fg.green(), fg.blue(), 40)   # 16% opacity
        axis  = QColor(fg.red(), fg.green(), fg.blue(), 120)  # 47% opacity

        painter.fillRect(self.rect(), bg)

        W = self.width()
        H = self.height()
        ml = self.margin_left
        mr = self.margin_right
        mt = self.margin_top
        mb = self.margin_bottom

        cw = W - ml - mr   # chart width
        ch = H - mt - mb   # chart height

        # Determine y range
        all_vals = self.training_data + self.validation_data
        if all_vals:
            data_max = max(all_vals)
            data_min = min(all_vals)
        else:
            data_max, data_min = 1.0, 0.0

        if self._cap:
            y_max = 1.0
            y_min = 0.0
        else:
            # round up to a nice ceiling, always start from 0
            import math
            y_max = max(data_max * 1.05, 0.01)
            magnitude = 10 ** math.floor(math.log10(y_max)) if y_max > 0 else 1
            y_max = math.ceil(y_max / magnitude) * magnitude
            y_min = 0.0

        y_range = y_max - y_min if y_max != y_min else 1.0

        def to_px(val, idx, n):
            x = ml + cw * idx / max(n - 1, 1)
            y = mt + ch * (1.0 - (val - y_min) / y_range)
            return QPointF(x, y)

        # ── title ──────────────────────────────────────────────────────────
        title_font = QFont()
        title_font.setPointSize(9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(fg))
        painter.drawText(ml, 0, cw, mt, 0x0084, self.title)  # AlignVCenter|AlignHCenter

        # ── grid + y-axis tick labels ───────────────────────────────────────
        tick_font = QFont()
        tick_font.setPointSize(8)
        painter.setFont(tick_font)

        N_TICKS = 5
        for i in range(N_TICKS + 1):
            val = y_min + (y_max - y_min) * i / N_TICKS
            y   = mt + ch * (1.0 - i / N_TICKS)

            # grid line
            painter.setPen(QPen(grid, 1))
            painter.drawLine(ml, int(y), W - mr, int(y))

            # tick label — right-aligned in the left margin
            painter.setPen(QPen(fg))
            label = f"{val:.2f}" if y_max < 10 else f"{val:.1f}"
            painter.drawText(0, int(y) - 10, ml - 6, 20, 0x0082, label)  # AlignRight|AlignVCenter

        # ── x-axis tick labels ─────────────────────────────────────────────
        n = max(len(self.training_data), len(self.validation_data))
        if n > 1:
            step = max(1, n // 6)
            for i in range(0, n, step):
                x = ml + cw * i / max(n - 1, 1)
                painter.setPen(QPen(fg))
                painter.drawText(int(x) - 20, H - mb + 4, 40, mb - 4, 0x0024, str(i + 1))  # AlignHCenter|AlignTop

        # ── axes ───────────────────────────────────────────────────────────
        painter.setPen(QPen(axis, 1))
        painter.drawLine(ml, mt,      ml,      H - mb)   # y-axis
        painter.drawLine(ml, H - mb,  W - mr,  H - mb)   # x-axis

        # ── y-axis label (rotated) ─────────────────────────────────────────
        painter.save()
        painter.setPen(QPen(fg))
        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        painter.translate(12, mt + ch / 2)
        painter.rotate(-90)
        painter.drawText(-40, -8, 80, 16, 0x0024, self.y_label)  # AlignHCenter
        painter.restore()

        # ── data lines ─────────────────────────────────────────────────────
        for data, color in (
            (self.training_data,   self.training_color),
            (self.validation_data, self.validation_color),
        ):
            if len(data) < 2:
                continue
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            for i, val in enumerate(data):
                pt = to_px(val, i, len(data))
                if i == 0:
                    path.moveTo(pt)
                else:
                    path.lineTo(pt)
            painter.drawPath(path)