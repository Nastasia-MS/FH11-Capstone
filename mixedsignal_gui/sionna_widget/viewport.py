"""Simplified 3D viewport for the Sionna widget.

Adapted from ``sionna-gui2/gui/viewport_3d.py``.  Key changes:
- Two dedicated signals ``tx_placed`` / ``rx_placed`` instead of generic
  ``placement_clicked(role, pos)``.
- No ``transceiver_delete_requested`` signal (fixed TX/RX pair).
- No ``transceiver_moved`` signal (positions managed via control panel).
- Convenience ``set_tx_rx_positions()`` wrapper.
- Toolbar placement actions removed (handled by control panel buttons).
"""

import math
import numpy as np
from enum import Enum

from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolBar, QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QVector3D, QAction
from OpenGL.GL import *


# ═══════════════════════════════════════════════════════════════════
# Placement mode enum
# ═══════════════════════════════════════════════════════════════════

class PlacementMode(Enum):
    NONE = 0
    PLACE_TX = 1
    PLACE_RX = 2


# ═══════════════════════════════════════════════════════════════════
# Core OpenGL widget
# ═══════════════════════════════════════════════════════════════════

class SceneGLWidget(QOpenGLWidget):
    """OpenGL widget with ray-casting, placement, and 3D transceiver markers."""

    tx_placed = Signal(list)
    rx_placed = Signal(list)
    transceiver_selected = Signal(str)
    hover_position_changed = Signal(list)
    placement_mode_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

        # Camera (Z-up to match Sionna)
        self.camera_target = QVector3D(0, 0, 0)
        self.camera_up = QVector3D(0, 0, 1)
        self.camera_yaw = -45.0
        self.camera_pitch = 30.0
        self.camera_distance = 500.0

        # Mouse state
        self._last_mouse_pos = QPoint()
        self._mouse_press_pos = QPoint()
        self._mouse_button = None
        self._click_threshold = 5

        # Scene meshes: list of (vertices_Nx3, faces_Mx3, color_tuple)
        self._meshes = []
        self._mesh_bboxes = []
        self._scene_center = np.array([0.0, 0.0, 0.0])
        self._scene_radius = 100.0

        # Transceivers
        self._transceiver_positions = {}
        self._transceiver_roles = {}
        self._selected_transceiver = None

        # Placement
        self._placement_mode = PlacementMode.NONE
        self._hover_world_pos = None

        # Paths
        self._path_vertices = None

        self.setFocusPolicy(Qt.StrongFocus)

    # ── Camera helpers ──────────────────────────────────

    def _get_camera_pos(self) -> QVector3D:
        yaw = math.radians(self.camera_yaw)
        pitch = math.radians(self.camera_pitch)
        x = self.camera_distance * math.cos(pitch) * math.cos(yaw)
        y = self.camera_distance * math.cos(pitch) * math.sin(yaw)
        z = self.camera_distance * math.sin(pitch)
        return self.camera_target + QVector3D(x, y, z)

    @property
    def _marker_radius(self) -> float:
        return max(2.0, self._scene_radius * 0.012)

    # ── OpenGL lifecycle ────────────────────────────────

    def initializeGL(self):
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_FLAT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.7, 0.7, 0.7, 1])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        aspect = self.width() / max(self.height(), 1)
        near = max(1.0, self.camera_distance * 0.01)
        far = self.camera_distance * 10 + self._scene_radius * 2

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self._gl_perspective(45.0, aspect, near, far)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cp = self._get_camera_pos()
        ct = self.camera_target
        cu = self.camera_up
        self._gl_look_at(
            cp.x(), cp.y(), cp.z(),
            ct.x(), ct.y(), ct.z(),
            cu.x(), cu.y(), cu.z(),
        )

        glLightfv(GL_LIGHT0, GL_POSITION, [cp.x(), cp.y(), cp.z() + 100, 1])

        self._draw_grid()
        self._draw_meshes()
        self._draw_transceivers()
        self._draw_ray_paths()
        self._draw_hover_preview()

    # ── GL matrix helpers ────────────────────────────────

    def _gl_perspective(self, fovy, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fovy) / 2.0)
        glMultMatrixf([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2 * far * near) / (near - far), 0,
        ])

    def _gl_look_at(self, ex, ey, ez, cx, cy, cz, ux, uy, uz):
        f = np.array([cx - ex, cy - ey, cz - ez], dtype=np.float64)
        f /= np.linalg.norm(f)
        u = np.array([ux, uy, uz], dtype=np.float64)
        s = np.cross(f, u)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        m = np.eye(4)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        glMultMatrixf(m.T.flatten().astype(np.float32))
        glTranslatef(-ex, -ey, -ez)

    # ── Numpy matrix builders (for ray unprojection) ────

    def _build_projection_matrix(self):
        aspect = self.width() / max(self.height(), 1)
        near = max(1.0, self.camera_distance * 0.01)
        far = self.camera_distance * 10 + self._scene_radius * 2
        f = 1.0 / math.tan(math.radians(45.0) / 2.0)

        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = (2 * far * near) / (near - far)
        P[3, 2] = -1.0
        return P

    def _build_view_matrix(self):
        cp = self._get_camera_pos()
        ex, ey, ez = cp.x(), cp.y(), cp.z()
        ct = self.camera_target
        cx, cy, cz = ct.x(), ct.y(), ct.z()
        ux, uy, uz = self.camera_up.x(), self.camera_up.y(), self.camera_up.z()

        f = np.array([cx - ex, cy - ey, cz - ez], dtype=np.float64)
        f /= np.linalg.norm(f)
        u = np.array([ux, uy, uz], dtype=np.float64)
        s = np.cross(f, u)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        rot = np.eye(4, dtype=np.float64)
        rot[0, :3] = s
        rot[1, :3] = u
        rot[2, :3] = -f

        T = np.eye(4, dtype=np.float64)
        T[0, 3] = -ex
        T[1, 3] = -ey
        T[2, 3] = -ez

        return rot @ T

    # ── Ray casting ─────────────────────────────────────

    def _screen_to_ray(self, sx, sy):
        """Return (ray_origin, ray_direction) in world space."""
        w, h = self.width(), self.height()
        ndc_x = 2.0 * sx / w - 1.0
        ndc_y = 1.0 - 2.0 * sy / h

        PV = self._build_projection_matrix() @ self._build_view_matrix()
        inv_PV = np.linalg.inv(PV)

        near_h = inv_PV @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_h  = inv_PV @ np.array([ndc_x, ndc_y,  1.0, 1.0])
        near_w = near_h[:3] / near_h[3]
        far_w  = far_h[:3]  / far_h[3]

        d = far_w - near_w
        d /= np.linalg.norm(d)
        return near_w, d

    @staticmethod
    def _ray_aabb(origin, inv_dir, bmin, bmax):
        t1 = (bmin - origin) * inv_dir
        t2 = (bmax - origin) * inv_dir
        tmin = np.max(np.minimum(t1, t2))
        tmax = np.min(np.maximum(t1, t2))
        return tmax >= max(tmin, 0.0)

    @staticmethod
    def _ray_mesh(origin, direction, verts, faces):
        """Vectorised Moller-Trumbore. Returns closest t or None."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0

        h = np.cross(direction, e2)
        a = np.sum(e1 * h, axis=1)

        EPS = 1e-7
        valid = np.abs(a) > EPS
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]

        s = origin - v0
        u = f * np.sum(s * h, axis=1)
        valid &= (u >= 0.0) & (u <= 1.0)

        q = np.cross(s, e1)
        v = f * np.sum(direction * q, axis=1)
        valid &= (v >= 0.0) & (u + v <= 1.0)

        t = f * np.sum(e2 * q, axis=1)
        valid &= t > EPS

        if not np.any(valid):
            return None
        return float(np.min(t[valid]))

    def _ray_cast_scene(self, screen_point: QPoint):
        origin, direction = self._screen_to_ray(
            screen_point.x(), screen_point.y()
        )
        inv_dir = np.where(
            np.abs(direction) > 1e-10,
            1.0 / direction,
            np.sign(direction) * 1e10,
        )

        best_t = float("inf")

        for (verts, faces, _), (bmin, bmax) in zip(
            self._meshes, self._mesh_bboxes
        ):
            if not self._ray_aabb(origin, inv_dir, bmin, bmax):
                continue
            t = self._ray_mesh(origin, direction, verts, faces)
            if t is not None and t < best_t:
                best_t = t

        # Ground-plane fallback (z = 0)
        if abs(direction[2]) > 1e-7:
            t_ground = -origin[2] / direction[2]
            if 0 < t_ground < best_t:
                best_t = t_ground

        if best_t < float("inf"):
            return origin + direction * best_t
        return None

    def _pick_transceiver(self, screen_point: QPoint, threshold_px=20):
        if not self._transceiver_positions:
            return None

        origin, direction = self._screen_to_ray(
            screen_point.x(), screen_point.y()
        )

        best_name = None
        best_dist = float("inf")
        angle = math.radians(45.0) * threshold_px / max(self.height(), 1)
        world_thresh = self.camera_distance * math.tan(angle)

        for name, pos in self._transceiver_positions.items():
            pos_arr = np.asarray(pos, dtype=np.float64)
            v = pos_arr - origin
            proj_len = float(np.dot(v, direction))
            if proj_len < 0:
                continue
            closest = origin + direction * proj_len
            dist = float(np.linalg.norm(pos_arr - closest))
            if dist < world_thresh and dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name

    # ── Drawing helpers ─────────────────────────────────

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor4f(0.3, 0.3, 0.3, 0.5)
        glBegin(GL_LINES)
        gs = max(500, int(self._scene_radius * 1.5))
        step = max(10, gs // 20)
        cx, cy = self._scene_center[0], self._scene_center[1]
        for i in range(-gs, gs + 1, step):
            glVertex3f(cx + i, cy - gs, 0)
            glVertex3f(cx + i, cy + gs, 0)
            glVertex3f(cx - gs, cy + i, 0)
            glVertex3f(cx + gs, cy + i, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_meshes(self):
        for vertices, faces, color in self._meshes:
            glColor3f(*color)
            glBegin(GL_TRIANGLES)
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nl = np.linalg.norm(n)
                if nl > 0:
                    n /= nl
                glNormal3f(*n)
                for idx in face:
                    glVertex3f(*vertices[idx])
            glEnd()

    # ── Transceiver 3D markers ──────────────────────────

    def _draw_sphere(self, centre, radius, color, alpha=1.0,
                     slices=14, stacks=10):
        glColor4f(*color, alpha)
        cx, cy, cz = centre
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + i / stacks)
            lat1 = math.pi * (-0.5 + (i + 1) / stacks)
            z0, r0 = math.sin(lat0), math.cos(lat0)
            z1, r1 = math.sin(lat1), math.cos(lat1)

            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * j / slices
                cj, sj = math.cos(lng), math.sin(lng)

                glNormal3f(cj * r0, sj * r0, z0)
                glVertex3f(cx + radius * cj * r0,
                           cy + radius * sj * r0,
                           cz + radius * z0)
                glNormal3f(cj * r1, sj * r1, z1)
                glVertex3f(cx + radius * cj * r1,
                           cy + radius * sj * r1,
                           cz + radius * z1)
            glEnd()

    def _draw_ring(self, centre, radius, color, segments=32):
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        glColor4f(*color, 0.9)
        glBegin(GL_LINE_LOOP)
        cx, cy, cz = centre
        for i in range(segments):
            a = 2 * math.pi * i / segments
            glVertex3f(cx + radius * math.cos(a),
                       cy + radius * math.sin(a),
                       cz)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_pole(self, top, ground_z=0.0, color=(0.6, 0.6, 0.6)):
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor4f(*color, 0.6)
        glBegin(GL_LINES)
        glVertex3f(top[0], top[1], ground_z)
        glVertex3f(*top)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_ground_shadow(self, pos, radius, color, ground_z=0.0):
        glDisable(GL_LIGHTING)
        glColor4f(*color, 0.25)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(pos[0], pos[1], ground_z + 0.05)
        for i in range(33):
            a = 2 * math.pi * i / 32
            glVertex3f(pos[0] + radius * math.cos(a),
                       pos[1] + radius * math.sin(a),
                       ground_z + 0.05)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_transceivers(self):
        r = self._marker_radius

        for name, pos in self._transceiver_positions.items():
            pos_arr = np.asarray(pos)
            is_tx = self._transceiver_roles.get(name, "tx") == "tx"
            base_color = (1.0, 0.25, 0.25) if is_tx else (0.25, 0.5, 1.0)
            selected = (name == self._selected_transceiver)

            self._draw_pole(pos_arr, ground_z=0.0, color=base_color)
            self._draw_ground_shadow(pos_arr, r * 1.2, base_color)
            self._draw_sphere(pos_arr, r, base_color)

            if is_tx:
                self._draw_cone_tip(pos_arr, r, base_color)

            if selected:
                self._draw_ring(pos_arr, r * 2.0, (1.0, 1.0, 0.3))
                self._draw_ring(pos_arr, r * 2.4, (1.0, 1.0, 0.3))

    def _draw_cone_tip(self, centre, radius, color):
        tip = np.array(centre) + np.array([0, 0, radius * 2.5])
        base_r = radius * 0.5
        bz = centre[2] + radius
        segs = 12
        glColor3f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 0, 1)
        glVertex3f(*tip)
        for i in range(segs + 1):
            a = 2 * math.pi * i / segs
            glVertex3f(centre[0] + base_r * math.cos(a),
                       centre[1] + base_r * math.sin(a),
                       bz)
        glEnd()

    # ── Hover preview (ghost marker) ───────────────────

    def _draw_hover_preview(self):
        if (self._placement_mode == PlacementMode.NONE
                or self._hover_world_pos is None):
            return

        r = self._marker_radius
        is_tx = self._placement_mode == PlacementMode.PLACE_TX
        color = (1.0, 0.5, 0.5) if is_tx else (0.5, 0.7, 1.0)
        pos = self._hover_world_pos

        glDepthMask(GL_FALSE)
        self._draw_pole(pos, ground_z=0.0, color=color)
        self._draw_sphere(pos, r, color, alpha=0.45)
        if is_tx:
            self._draw_cone_tip(pos, r, color)
        self._draw_crosshair(pos, r * 2.5, color)
        glDepthMask(GL_TRUE)

    def _draw_crosshair(self, pos, size, color):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor4f(*color, 0.8)
        glBegin(GL_LINES)
        glVertex3f(pos[0] - size, pos[1], pos[2] + 0.1)
        glVertex3f(pos[0] + size, pos[1], pos[2] + 0.1)
        glVertex3f(pos[0], pos[1] - size, pos[2] + 0.1)
        glVertex3f(pos[0], pos[1] + size, pos[2] + 0.1)
        glEnd()
        glEnable(GL_LIGHTING)

    # ── Path drawing ────────────────────────────────────

    def _draw_ray_paths(self):
        if self._path_vertices is None:
            return
        glDisable(GL_LIGHTING)
        glLineWidth(1.5)
        glColor4f(1.0, 0.9, 0.2, 0.7)
        for path in self._path_vertices:
            glBegin(GL_LINE_STRIP)
            for v in path:
                pt = np.asarray(v, dtype=np.float32).flatten()
                if pt.size >= 3:
                    glVertex3f(float(pt[0]), float(pt[1]), float(pt[2]))
            glEnd()
        glEnable(GL_LIGHTING)

    # ── Public API ──────────────────────────────────────

    def set_meshes(self, meshes, bbox_min, bbox_max):
        self._meshes = meshes
        self._mesh_bboxes = []
        for verts, faces, _ in meshes:
            self._mesh_bboxes.append((verts.min(axis=0), verts.max(axis=0)))

        self._scene_center = (bbox_min + bbox_max) / 2
        self._scene_radius = float(np.linalg.norm(bbox_max - bbox_min) / 2)
        self.camera_target = QVector3D(*self._scene_center)
        self.camera_distance = self._scene_radius * 2.5
        self.update()

    def set_transceivers(self, positions: dict, roles: dict):
        """positions: name->[x,y,z]  roles: name->'tx'|'rx'"""
        self._transceiver_positions = {
            k: np.asarray(v, dtype=np.float64) for k, v in positions.items()
        }
        self._transceiver_roles = dict(roles)
        if self._selected_transceiver not in self._transceiver_positions:
            self._selected_transceiver = None
        self.update()

    def set_tx_rx_positions(self, tx_pos, rx_pos):
        """Convenience: update the fixed TX/RX pair markers."""
        from .engine import SimpleSimulationEngine
        positions = {
            SimpleSimulationEngine.TX_NAME: tx_pos,
            SimpleSimulationEngine.RX_NAME: rx_pos,
        }
        roles = {
            SimpleSimulationEngine.TX_NAME: "tx",
            SimpleSimulationEngine.RX_NAME: "rx",
        }
        self.set_transceivers(positions, roles)

    def set_path_vertices(self, vertices):
        self._path_vertices = vertices
        self.update()

    def clear_paths(self):
        self._path_vertices = None
        self.update()

    # ── Placement mode ──────────────────────────────────

    def enter_placement_mode(self, role: str):
        self._placement_mode = (
            PlacementMode.PLACE_TX if role == "tx"
            else PlacementMode.PLACE_RX
        )
        self.setCursor(Qt.CrossCursor)
        self._hover_world_pos = None
        self.placement_mode_changed.emit(True)
        self.update()

    def exit_placement_mode(self):
        self._placement_mode = PlacementMode.NONE
        self.setCursor(Qt.ArrowCursor)
        self._hover_world_pos = None
        self.placement_mode_changed.emit(False)
        self.update()

    @property
    def in_placement_mode(self) -> bool:
        return self._placement_mode != PlacementMode.NONE

    # ── Mouse handling ──────────────────────────────────

    def mousePressEvent(self, event):
        self._last_mouse_pos = event.position().toPoint()
        self._mouse_press_pos = event.position().toPoint()
        self._mouse_button = event.button()

        if self.in_placement_mode and event.button() == Qt.RightButton:
            self.exit_placement_mode()

    def mouseReleaseEvent(self, event):
        release = event.position().toPoint()
        drag = (release - self._mouse_press_pos).manhattanLength()
        is_click = drag < self._click_threshold

        if is_click and event.button() == Qt.LeftButton:
            if self.in_placement_mode:
                hit = self._ray_cast_scene(release)
                if hit is not None:
                    if self._placement_mode == PlacementMode.PLACE_TX:
                        self.tx_placed.emit(hit.tolist())
                    else:
                        self.rx_placed.emit(hit.tolist())
            else:
                name = self._pick_transceiver(release)
                self._selected_transceiver = name
                self.transceiver_selected.emit(name or "")
                self.update()

        self._mouse_button = None

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        buttons = event.buttons()

        if self.in_placement_mode:
            if buttons == Qt.NoButton:
                hit = self._ray_cast_scene(pos)
                self._hover_world_pos = hit
                if hit is not None:
                    self.hover_position_changed.emit(hit.tolist())
                self.update()
                self._last_mouse_pos = pos
                return
            if buttons & Qt.MiddleButton:
                self._do_pan(pos)
                self._last_mouse_pos = pos
                self.update()
                return
            self._last_mouse_pos = pos
            return

        if buttons & Qt.LeftButton:
            self._do_orbit(pos)
        elif buttons & Qt.MiddleButton:
            self._do_pan(pos)

        self._last_mouse_pos = pos
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        factor = 0.9 if delta > 0 else 1.1
        self.camera_distance = max(
            self._scene_radius * 0.1,
            min(self._scene_radius * 20,
                self.camera_distance * factor),
        )
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.in_placement_mode:
                self.exit_placement_mode()
            else:
                self._selected_transceiver = None
                self.transceiver_selected.emit("")
                self.update()
        super().keyPressEvent(event)

    # ── Orbit / pan helpers ─────────────────────────────

    def _do_orbit(self, pos):
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self.camera_yaw -= dx * 0.3
        self.camera_pitch = max(-89, min(89, self.camera_pitch + dy * 0.3))

    def _do_pan(self, pos):
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        cp = self._get_camera_pos()
        fwd = self.camera_target - cp
        fwd_flat = QVector3D(fwd.x(), fwd.y(), 0).normalized()
        right = QVector3D.crossProduct(fwd_flat, self.camera_up).normalized()
        speed = self.camera_distance * 0.002
        pan = right * (-dx * speed) + self.camera_up * (dy * speed)
        self.camera_target += pan


# ═══════════════════════════════════════════════════════════════════
# Wrapper widget with toolbar
# ═══════════════════════════════════════════════════════════════════

class SimpleViewport(QWidget):
    """Toolbar + OpenGL view.  Placement actions are in the control panel."""

    tx_placed = Signal(list)
    rx_placed = Signal(list)
    hover_position = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Toolbar ──────────────────────────────────────
        toolbar = QToolBar()
        toolbar.addAction("Reset View", self._reset_camera)
        toolbar.addAction("Top", self._top_view)
        toolbar.addAction("Front", self._front_view)
        toolbar.addAction("Side", self._side_view)
        toolbar.addSeparator()
        toolbar.addAction("Clear Paths", self._clear_paths)

        self._coord_label = QLabel("  ---")
        self._coord_label.setStyleSheet("color: #aaa; font-family: 'Courier New', Courier, monospace;")
        toolbar.addWidget(self._coord_label)

        layout.addWidget(toolbar)

        # ── GL widget ────────────────────────────────────
        self.gl_widget = SceneGLWidget()
        layout.addWidget(self.gl_widget)

        # Forward signals
        self.gl_widget.tx_placed.connect(self.tx_placed)
        self.gl_widget.rx_placed.connect(self.rx_placed)
        self.gl_widget.hover_position_changed.connect(self._on_hover)
        self.gl_widget.placement_mode_changed.connect(self._on_mode_changed)

    # ── Hover / mode feedback ────────────────────────────

    def _on_hover(self, pos):
        self._coord_label.setText(
            f"  X:{pos[0]:+.1f}  Y:{pos[1]:+.1f}  Z:{pos[2]:+.1f}"
        )
        self.hover_position.emit(pos)

    def _on_mode_changed(self, active: bool):
        if not active:
            self._coord_label.setText("  ---")

    # ── Placement passthrough ────────────────────────────

    def enter_placement_mode(self, role: str):
        self.gl_widget.enter_placement_mode(role)

    def exit_placement_mode(self):
        self.gl_widget.exit_placement_mode()

    # ── Scene management ─────────────────────────────────

    def refresh_scene_meshes(self, scene):
        """Extract meshes from a Sionna scene object and load into GL."""
        if scene is None:
            return

        meshes = []
        all_vertices = []
        colors = [
            (0.70, 0.70, 0.75),
            (0.60, 0.65, 0.70),
            (0.65, 0.60, 0.55),
            (0.55, 0.60, 0.65),
        ]

        try:
            for i, (name, obj) in enumerate(scene.objects.items()):
                verts = np.array(
                    obj.mi_mesh.vertex_positions_buffer()
                ).reshape(-1, 3)
                faces = np.array(
                    obj.mi_mesh.faces_buffer()
                ).reshape(-1, 3).astype(np.int32)
                meshes.append((verts, faces, colors[i % len(colors)]))
                all_vertices.append(verts)

            if all_vertices:
                combined = np.vstack(all_vertices)
                self.gl_widget.set_meshes(
                    meshes, combined.min(axis=0), combined.max(axis=0)
                )
        except Exception as e:
            import traceback
            print(f"Error loading meshes: {e}")
            traceback.print_exc()

    def update_tx_rx(self, tx_pos, rx_pos):
        """Convenience: update the TX/RX marker positions."""
        self.gl_widget.set_tx_rx_positions(tx_pos, rx_pos)

    def draw_paths(self, vertices):
        if vertices is not None:
            self.gl_widget.set_path_vertices(vertices)

    # ── Camera presets ───────────────────────────────────

    def _reset_camera(self):
        g = self.gl_widget
        g.camera_yaw = -45.0
        g.camera_pitch = 30.0
        g.camera_distance = g._scene_radius * 2.5
        g.camera_target = QVector3D(*g._scene_center)
        g.update()

    def _top_view(self):
        self.gl_widget.camera_yaw = 0
        self.gl_widget.camera_pitch = 89
        self.gl_widget.update()

    def _front_view(self):
        self.gl_widget.camera_yaw = 0
        self.gl_widget.camera_pitch = 0
        self.gl_widget.update()

    def _side_view(self):
        self.gl_widget.camera_yaw = 90
        self.gl_widget.camera_pitch = 0
        self.gl_widget.update()

    def _clear_paths(self):
        self.gl_widget.clear_paths()
