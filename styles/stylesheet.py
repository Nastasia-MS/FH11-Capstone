def get_stylesheet():
    """Returns the complete application stylesheet"""
    return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QLabel {
            color: #1f2937;
        }
        .title {
            font-size: 16px;
            font-weight: bold;
            color: #111827;
        }
        .subtitle {
            font-size: 11px;
            color: #6b7280;
        }
        .section-title {
            font-size: 13px;
            font-weight: 600;
            color: #111827;
        }
        .section-subtitle {
            font-size: 11px;
            color: #6b7280;
        }
        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: #111827;
        }
        .stat-value {
            font-size: 14px;
            font-weight: 600;
            color: #111827;
        }
        .stat-label {
            font-size: 12px;
            color: #6b7280;
        }
        QPushButton {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 7px 14px;
            font-size: 12px;
            color: #374151;
        }
        QPushButton:hover {
            background-color: #f9fafb;
            border-color: #d1d5db;
        }
        QPushButton#primaryButton {
            background-color: #111827;
            color: white;
            border: none;
        }
        QPushButton#primaryButton:hover {
            background-color: #1f2937;
        }
        QPushButton#tabButton {
            background-color: white;
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0px;
            padding: 12px 12px;
            font-size: 13px;
            font-weight: 500;
        }
        QPushButton#tabButton:checked {
            border-bottom: 2px solid #111827;
            color: #111827;
        }
        QPushButton#subtabPill {
            background-color: #f5f5f5;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 500;
            color: #6b7280;
        }
        QPushButton#subtabPill:checked {
            background-color: #111827;
            color: white;
            border: 1px solid #111827;
        }
        QPushButton#subtabPill:hover:!checked {
            background-color: #e5e7eb;
            color: #374151;
        }
        /* Combo box / dropdown improvements for higher contrast */
        QComboBox {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 12px;
            color: #111827;
            min-height: 16px;
        }
        QComboBox:hover, QComboBox:focus {
            border-color: #111827;
            background-color: #ffffff;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #6b7280;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            selection-background-color: #111827;
            selection-color: #ffffff;
            color: #111827;
            padding: 4px;
        }
        QComboBox QAbstractItemView::item {
            padding: 6px 10px;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #111827;
            color: #ffffff;
        }
        /* Menus and list views (covering other option widgets) */
        QMenu, QListView {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #111827;
        }
        QMenu::item, QListView::item {
            padding: 6px 12px;
            color: #111827;
        }
        QMenu::item:selected, QListView::item:selected {
            background-color: #111827;
            color: #ffffff;
        }
        QDoubleSpinBox {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 11px;
            color: #1f2937;
            min-height: 12px;
        }
        QSpinBox {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 11px;
            color: #1f2937;
            min-height: 12px;
        }
        QSpinBox::up-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid #6b7280;
        }
        QSpinBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #6b7280;
        }
        QSlider::groove:horizontal {
            border: none;
            height: 4px;
            background: #e5e7eb;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #111827;
            border: none;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QFrame#card {
            background-color: white;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
        }
        QFrame#infoBox {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 12px;
        }
        QCheckBox {
            spacing: 8px;
            font-size: 13px;
            color: #1f2937;
        }
        QCheckBox::indicator {
            width: 40px;
            height: 20px;
            border-radius: 10px;
            background-color: #d1d5db;
        }
        QCheckBox::indicator:checked {
            background-color: #111827;
        }
        QTableWidget {
            background-color: white;
            border: none;
            gridline-color: transparent;
            font-size: 13px;
        }
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #f3f4f6;
            border-right: none;
        }
        QHeaderView::section {
            background-color: white;
            padding: 12px 8px;
            border: none;
            border-bottom: 1px solid #e5e7eb;
            font-weight: 600;
            font-size: 12px;
            color: #6b7280;
        }
        .badge {
            border-radius: 12px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 600;
        }
        .badge-black {
            background-color: #111827;
            color: white;
        }
        .badge-red {
            background-color: #ef4444;
            color: white;
        }
        QProgressBar {
            border: none;
            border-radius: 4px;
            background-color: #e5e7eb;
            height: 8px;
        }
        QProgressBar::chunk {
            background-color: #111827;
            border-radius: 4px;
        }
        QTabBar::tab {
            background: #2b2b2b;
            color: #dddddd;
            padding: 8px 14px;
            border-top-left-radius: 1px;
            border-top-right-radius: 1px;
        }
        QTabBar::tab:selected {
            background: #5a5a5a;
            color: white;
        }
        QTabWidget::pane {
            border: none;
            top: -1px;
        }
    """