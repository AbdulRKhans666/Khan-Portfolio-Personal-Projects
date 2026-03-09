"""
=============================================================
  SPACECRAFT ATTITUDE CONTROL VISUALIZER
  Runs directly in Spyder (Python only )

  Dependencies (install once):
      pip install PyQt5 PyOpenGL matplotlib scipy numpy

  How to run in Spyder:
      1. Open this file
      2. Press F5  (or Run > Run File)
      3. A GUI window will appear with:
           - 3D spacecraft viewer (OpenGL)
           - Live telemetry readout
           - Attitude / rate / torque charts
           - PID vs LQR comparison chart
           - Playback controls
=============================================================
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# ── Qt + OpenGL ──────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QComboBox, QGroupBox,
    QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

# ── Matplotlib (embedded) ────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# ═══════════════════════════════════════════════════════════════
#  PHYSICS & SIMULATION
# ═══════════════════════════════════════════════════════════════

I_tensor = np.diag([100.0, 85.0, 120.0]) #inverse of the inertia tensor
I_inv    = np.linalg.inv(I_tensor)
TAU_MAX  = 20.0 #maximum torque actuators can produce N·m

roll = 25 # degs
pitch = 20 # degs
yaw = 30 #degs

roll_rate = 0.05 #rad/s
pitch_rate = -0.03 #rad/s
yaw_rate = 0.04 #rad/s


def quat_mult(p, q):
    px, py, pz, pw = p;  qx, qy, qz, qw = q
    return np.array([pw*qx+px*qw+py*qz-pz*qy, pw*qy-px*qz+py*qw+pz*qx,
                     pw*qz+px*qy-py*qx+pz*qw, pw*qw-px*qx-py*qy-pz*qz])

def quat_conj(q):  return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_error(q, q_des):  return quat_mult(quat_conj(q_des), q)

def normalize_quat(q):  return q / np.linalg.norm(q)

def quat_kinematics(q, omega):
    qx, qy, qz, qw = q
    Xi = 0.5 * np.array([[ qw,-qz, qy],[ qz, qw,-qx],[-qy, qx, qw],[-qx,-qy,-qz]])
    return Xi @ omega

def euler_to_quat(roll, pitch, yaw):
    cr,sr = np.cos(roll/2), np.sin(roll/2)
    cp,sp = np.cos(pitch/2),np.sin(pitch/2)
    cy,sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])

def quat_to_euler(q):
    qx,qy,qz,qw = q
    roll  = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))
    pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))
    yaw   = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))
    return np.array([roll, pitch, yaw])

def spacecraft_dynamics(t, state, torque_func):
    q, omega = state[:4], state[4:]
    tau = np.clip(torque_func(t, state), -TAU_MAX, TAU_MAX)
    omega_dot = I_inv @ (tau - np.cross(omega, I_tensor @ omega))
    q_dot = quat_kinematics(normalize_quat(q), omega)
    return np.concatenate([q_dot, omega_dot])


class PIDController:
    def __init__(self, q_des=None):
        self.q_des = q_des if q_des is not None else np.array([0.,0.,0.,1.])
        self.Kp = np.array([16.0, 13.6, 19.2])
        self.Ki = np.array([0.05, 0.05, 0.05])
        self.Kd = np.array([72.0, 61.2, 86.4])
        self._integral = np.zeros(3)
        self._prev_t = 0.0

    def __call__(self, t, state):
        q, omega = normalize_quat(state[:4]), state[4:]
        q_err = quat_error(q, self.q_des)
        if q_err[3] < 0: q_err = -q_err
        e_p = q_err[:3]
        dt = max(t - self._prev_t, 1e-4);  self._prev_t = t
        self._integral = np.clip(self._integral + e_p * dt, -1.0, 1.0)
        return -self.Kp*e_p - self.Ki*self._integral - self.Kd*omega


def build_lqr_gain():
    A = np.zeros((6,6));  A[0:3,3:6] = 0.5*np.eye(3)
    B = np.zeros((6,3));  B[3:6,:] = I_inv
    Q = np.diag([80.,80.,80.,5.,5.,5.])
    R = np.diag([0.1,0.1,0.1])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


class LQRController:
    def __init__(self, q_des=None):
        self.q_des = q_des if q_des is not None else np.array([0.,0.,0.,1.])
        self.K = build_lqr_gain()

    def __call__(self, t, state):
        q, omega = normalize_quat(state[:4]), state[4:]
        q_err = quat_error(q, self.q_des)
        if q_err[3] < 0: q_err = -q_err
        return -(self.K @ np.concatenate([q_err[:3], omega]))


def run_simulation(controller):
    q0 = euler_to_quat(np.radians(roll), np.radians(pitch), np.radians(yaw)) 
    #initial atitude (roll, pitch, yaw)
    omega0 = np.array([roll_rate, pitch_rate, yaw_rate])
    #initial spin (roll rate, pitch rate, yaw rate)
    state0 = np.concatenate([q0, omega0])
    t_eval = np.arange(0, 60.0, 0.1)
    #time duration (secs)
    sol = solve_ivp(lambda t,s: spacecraft_dynamics(t,s,controller),
                    (0,60), state0, t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)
    return sol


def extract_frames(sol, controller):
    q_des = controller.q_des
    frames = []
    for i in range(len(sol.t)):
        q     = normalize_quat(sol.y[:4, i])
        omega = sol.y[4:, i]
        euler = np.degrees(quat_to_euler(q))
        qe    = quat_error(q, q_des)
        if qe[3] < 0: qe = -qe
        err   = 2*np.degrees(np.arcsin(np.clip(np.linalg.norm(qe[:3]),0,1)))
        tau   = np.clip(controller(sol.t[i], sol.y[:,i]), -TAU_MAX, TAU_MAX)
        frames.append({'t': sol.t[i], 'q': q.copy(), 'euler': euler,
                       'omega': np.degrees(omega), 'error': err, 'torque': tau.copy()})
    return frames


# ═══════════════════════════════════════════════════════════════
#  3D OPENGL SPACECRAFT WIDGET
# ═══════════════════════════════════════════════════════════════

class SpacecraftGL(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.quaternion = np.array([0., 0., 0., 1.])
        self.setMinimumSize(380, 320)

    def set_quaternion(self, q):
        self.quaternion = q
        self.update()

    def initializeGL(self):
        glClearColor(0.02, 0.04, 0.08, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)

        # Sunlight
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 8.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.95, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.1, 0.12, 0.15, 1.0])

        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, [-4.0, -2.0, -4.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.1, 0.15, 0.3, 1.0])

    def resizeGL(self, w, h):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera
        gluLookAt(4.5, 3.0, 4.5,  0, 0, 0,  0, 1, 0)

        # Draw stars (small white dots)
        glDisable(GL_LIGHTING)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-40, 40, (200, 3))
        glPointSize(1.5)
        glBegin(GL_POINTS)
        for p in pts:
            brightness = rng.uniform(0.4, 1.0)
            glColor3f(brightness, brightness, brightness)
            glVertex3fv(p)
        glEnd()
        glEnable(GL_LIGHTING)

        # Apply spacecraft rotation from quaternion [qx,qy,qz,qw]
        qx, qy, qz, qw = self.quaternion
        # Convert quaternion to axis-angle for glRotatef
        angle = 2 * np.degrees(np.arccos(np.clip(qw, -1, 1)))
        s = np.sqrt(max(1 - qw*qw, 0))
        if s < 1e-6:
            ax, ay, az = 0, 1, 0
        else:
            ax, ay, az = qx/s, qy/s, qz/s
        glRotatef(angle, ax, ay, az)

        # ── Draw reference axes (body frame) ──
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        axis_len = 2.2
        glBegin(GL_LINES)
        glColor3f(1.0, 0.25, 0.25); glVertex3f(0,0,0); glVertex3f(axis_len,0,0)  # X roll
        glColor3f(0.25, 1.0, 0.4);  glVertex3f(0,0,0); glVertex3f(0,axis_len,0)  # Y pitch
        glColor3f(0.3, 0.55, 1.0);  glVertex3f(0,0,0); glVertex3f(0,0,axis_len)  # Z yaw
        glEnd()
        self._draw_axis_labels(axis_len)
        glEnable(GL_LIGHTING)

        # ── Main satellite bus ──
        glColor3f(0.18, 0.28, 0.48)
        self._draw_box(0, 0, 0, 0.8, 0.5, 1.2)

        # ── Solar panel arms ──
        glColor3f(0.22, 0.32, 0.52)
        self._draw_box(-1.0, 0, 0, 0.4, 0.06, 0.5)
        self._draw_box( 1.0, 0, 0, 0.4, 0.06, 0.5)

        # ── Solar panels ──
        glColor3f(0.06, 0.12, 0.32)
        self._draw_box(-1.9, 0, 0, 1.4, 0.03, 0.65)
        self._draw_box( 1.9, 0, 0, 1.4, 0.03, 0.65)
        # Panel highlight (cell pattern suggestion)
        glColor3f(0.1, 0.2, 0.5)
        for side in [-1, 1]:
            for row in [-1, 0, 1]:
                self._draw_box(side*1.9, 0.02, row*0.18, 1.3, 0.01, 0.12)

        # ── Antenna dish ──
        glColor3f(0.75, 0.75, 0.78)
        self._draw_disk(0, 0.36, -0.55, 0.32, tilt_x=-0.5)

        # Antenna stick
        glColor3f(0.5, 0.5, 0.55)
        glLineWidth(3)
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glVertex3f(0, 0.25, -0.4); glVertex3f(0, 0.42, -0.55)
        glEnd()
        glEnable(GL_LIGHTING)

        # ── Star tracker ──
        glColor3f(0.08, 0.08, 0.1)
        self._draw_box(0.22, 0.35, 0.45, 0.16, 0.16, 0.14)
        glColor3f(0.3, 0.3, 0.35)
        self._draw_box(0.22, 0.43, 0.45, 0.10, 0.02, 0.10)

        # ── Thruster nozzles (corners) ──
        glColor3f(0.45, 0.45, 0.5)
        for x in [-0.45, 0.45]:
            for z in [-0.65, 0.65]:
                self._draw_box(x, -0.28, z, 0.08, 0.1, 0.08)

    def _draw_box(self, cx, cy, cz, w, h, d):
        """Draw a solid box centered at (cx,cy,cz) with dimensions w×h×d."""
        x0,x1 = cx-w/2, cx+w/2
        y0,y1 = cy-h/2, cy+h/2
        z0,z1 = cz-d/2, cz+d/2
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0,0,1); glVertex3f(x0,y0,z1); glVertex3f(x1,y0,z1); glVertex3f(x1,y1,z1); glVertex3f(x0,y1,z1)
        # Back
        glNormal3f(0,0,-1);glVertex3f(x1,y0,z0); glVertex3f(x0,y0,z0); glVertex3f(x0,y1,z0); glVertex3f(x1,y1,z0)
        # Top
        glNormal3f(0,1,0); glVertex3f(x0,y1,z0); glVertex3f(x1,y1,z0); glVertex3f(x1,y1,z1); glVertex3f(x0,y1,z1)
        # Bottom
        glNormal3f(0,-1,0);glVertex3f(x0,y0,z1); glVertex3f(x1,y0,z1); glVertex3f(x1,y0,z0); glVertex3f(x0,y0,z0)
        # Right
        glNormal3f(1,0,0); glVertex3f(x1,y0,z0); glVertex3f(x1,y1,z0); glVertex3f(x1,y1,z1); glVertex3f(x1,y0,z1)
        # Left
        glNormal3f(-1,0,0);glVertex3f(x0,y0,z1); glVertex3f(x0,y1,z1); glVertex3f(x0,y1,z0); glVertex3f(x0,y0,z0)
        glEnd()

    def _draw_disk(self, cx, cy, cz, r, tilt_x=0, segments=24):
        """Draw a simple disk (antenna dish face)."""
        glPushMatrix()
        glTranslatef(cx, cy, cz)
        glRotatef(np.degrees(tilt_x), 1, 0, 0)
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        for i in range(segments + 1):
            a = 2 * np.pi * i / segments
            glVertex3f(r * np.cos(a), 0, r * np.sin(a))
        glEnd()
        glPopMatrix()

    def _draw_axis_labels(self, length):
        """Draw X/Y/Z text labels at axis tips using raster positions."""
        # Use OpenGL bitmap text via GLUT-style rendering is complex,
        # so we draw simple +/- markers instead as colored lines
        glLineWidth(1.5)
        tip = length + 0.15
        labels = [
            ((1,0,0), tip, 0, 0, (1.0,0.3,0.3)),   # X
            ((0,1,0), 0, tip, 0, (0.3,1.0,0.5)),   # Y
            ((0,0,1), 0, 0, tip, (0.3,0.6,1.0)),   # Z
        ]
        for (nx,ny,nz), x, y, z, col in labels:
            glColor3f(*col)
            s = 0.12
            glBegin(GL_LINES)
            # Small perpendicular tick
            px = ny * s if ny else nz * s if nz else 0
            py = nx * s if nx else nz * s if nz else 0
            pz = 0 if nz else nx * s
            glVertex3f(x-px, y-py, z-pz)
            glVertex3f(x+px, y+py, z+pz)
            glEnd()


# ═══════════════════════════════════════════════════════════════
#  MATPLOTLIB CHART WIDGET
# ═══════════════════════════════════════════════════════════════

DARK_BG   = "#040d1a"
PANEL_BG  = "#071020"
ACCENT1   = "#ef4444"   # PID red
ACCENT2   = "#3b82f6"   # LQR blue
GRID_COL  = "#0f2030"
TEXT_COL  = "#94a3b8"


class ChartWidget(FigureCanvas):
    def __init__(self, pid_frames, lqr_frames, parent=None):
        self.fig = Figure(figsize=(5, 7), facecolor=DARK_BG, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.pid_frames = pid_frames
        self.lqr_frames = lqr_frames
        self._build_axes()
        self._plot_full_traces()
        self.pid_markers = []
        self.lqr_markers = []
        self._init_markers()

    def _build_axes(self):
        gs = self.fig.add_gridspec(4, 1, hspace=0.55,
                                   top=0.95, bottom=0.07, left=0.14, right=0.95)
        self.ax_err   = self.fig.add_subplot(gs[0])
        self.ax_euler = self.fig.add_subplot(gs[1])
        self.ax_omega = self.fig.add_subplot(gs[2])
        self.ax_torq  = self.fig.add_subplot(gs[3])

        titles = ["Attitude Error (°)", "Euler Angles (°)", "Angular Rate (°/s)", "Torque (N·m)"]
        for ax, title in zip([self.ax_err, self.ax_euler, self.ax_omega, self.ax_torq], titles):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_COL, labelsize=7)
            ax.set_title(title, color=TEXT_COL, fontsize=8, pad=3)
            for spine in ax.spines.values(): spine.set_color(GRID_COL)
            ax.grid(color=GRID_COL, linewidth=0.5)
            ax.set_xlim(0, 60)

        self.ax_err.axhline(2, color="#22c55e", lw=0.8, ls="--", alpha=0.6, label="2° threshold")

    def _plot_full_traces(self):
        t = [f['t'] for f in self.pid_frames]

        # Error
        self.ax_err.plot(t, [f['error'] for f in self.pid_frames], color=ACCENT1, lw=1.2, label="PID", alpha=0.85)
        self.ax_err.plot(t, [f['error'] for f in self.lqr_frames], color=ACCENT2, lw=1.2, label="LQR", alpha=0.85)
        self.ax_err.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)

        # Euler angles (PID only, dotted for LQR)
        euler_colors = ["#f87171", "#c084fc", "#60a5fa"]
        euler_labels = ["Roll", "Pitch", "Yaw"]
        for i, (col, lbl) in enumerate(zip(euler_colors, euler_labels)):
            self.ax_euler.plot(t, [f['euler'][i] for f in self.pid_frames], color=col, lw=1.2, label=lbl)
        self.ax_euler.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)

        # Angular rate magnitude
        self.ax_omega.plot(t, [np.linalg.norm(f['omega']) for f in self.pid_frames], color=ACCENT1, lw=1.2, label="PID")
        self.ax_omega.plot(t, [np.linalg.norm(f['omega']) for f in self.lqr_frames], color=ACCENT2, lw=1.2, label="LQR")
        self.ax_omega.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)

        # Torque magnitude
        self.ax_torq.plot(t, [np.linalg.norm(f['torque']) for f in self.pid_frames], color=ACCENT1, lw=1.2, label="PID")
        self.ax_torq.plot(t, [np.linalg.norm(f['torque']) for f in self.lqr_frames], color=ACCENT2, lw=1.2, label="LQR")
        self.ax_torq.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)

    def _init_markers(self):
        """Vertical time cursor lines."""
        for ax in [self.ax_err, self.ax_euler, self.ax_omega, self.ax_torq]:
            lp, = ax.plot([0, 0], ax.get_ylim(), color=ACCENT1, lw=0.8, alpha=0.5, ls=":")
            ll, = ax.plot([0, 0], ax.get_ylim(), color=ACCENT2, lw=0.8, alpha=0.5, ls=":")
            self.pid_markers.append(lp)
            self.lqr_markers.append(ll)

    def update_cursor(self, t, controller):
        """Move the time cursor to time t for the given controller."""
        markers = self.pid_markers if controller == "PID" else self.lqr_markers
        for ax, line in zip([self.ax_err, self.ax_euler, self.ax_omega, self.ax_torq], markers):
            ymin, ymax = ax.get_ylim()
            line.set_xdata([t, t])
            line.set_ydata([ymin, ymax])
        self.draw_idle()


# ═══════════════════════════════════════════════════════════════
#  TELEMETRY PANEL
# ═══════════════════════════════════════════════════════════════

class TelemetryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT_COL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        font_mono = QFont("Courier New", 9)
        font_val  = QFont("Courier New", 11)
        font_val.setBold(True)

        def section(title):
            lbl = QLabel(f"── {title} ──────────────")
            lbl.setStyleSheet(f"color: #1e3a5f; font-size: 9px;")
            lbl.setFont(font_mono)
            layout.addWidget(lbl)

        def row(label):
            h = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(font_mono)
            lbl.setStyleSheet("color: #4a5568; font-size: 9px;")
            val = QLabel("---")
            val.setFont(font_val)
            val.setAlignment(Qt.AlignRight)
            h.addWidget(lbl); h.addWidget(val)
            layout.addLayout(h)
            return val

        section("CONTROLLER")
        self.lbl_ctrl = QLabel("---")
        self.lbl_ctrl.setFont(QFont("Courier New", 14, QFont.Bold))
        self.lbl_ctrl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_ctrl)

        section("ATTITUDE")
        self.lbl_roll  = row("ROLL")
        self.lbl_pitch = row("PITCH")
        self.lbl_yaw   = row("YAW")

        section("ANGULAR RATE")
        self.lbl_wx = row("ωX")
        self.lbl_wy = row("ωY")
        self.lbl_wz = row("ωZ")

        section("TORQUE")
        self.lbl_tx = row("τX")
        self.lbl_ty = row("τY")
        self.lbl_tz = row("τZ")

        section("TIME / ERROR")
        self.lbl_time  = row("TIME")
        self.lbl_error = row("ERROR")

        layout.addStretch()

    def update_frame(self, frame, controller):
        color = ACCENT1 if controller == "PID" else ACCENT2
        self.lbl_ctrl.setText(controller)
        self.lbl_ctrl.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

        r, p, y = frame['euler']
        self.lbl_roll.setText(f"{r:+.2f}°")
        self.lbl_pitch.setText(f"{p:+.2f}°")
        self.lbl_yaw.setText(f"{y:+.2f}°")

        ox, oy, oz = frame['omega']
        self.lbl_wx.setText(f"{ox:+.3f}°/s")
        self.lbl_wy.setText(f"{oy:+.3f}°/s")
        self.lbl_wz.setText(f"{oz:+.3f}°/s")

        tx, ty, tz = frame['torque']
        self.lbl_tx.setText(f"{tx:+.2f} N·m")
        self.lbl_ty.setText(f"{ty:+.2f} N·m")
        self.lbl_tz.setText(f"{tz:+.2f} N·m")

        self.lbl_time.setText(f"{frame['t']:.1f} s")

        err = frame['error']
        ecol = "#22c55e" if err < 2.0 else color
        self.lbl_error.setText(f"{err:.3f}°")
        self.lbl_error.setStyleSheet(f"color: {ecol}; font-size: 11px; font-weight: bold;")


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, pid_frames, lqr_frames):
        super().__init__()
        self.pid_frames = pid_frames
        self.lqr_frames = lqr_frames
        self.current_frame_idx = 0
        self.controller = "LQR"
        self.playing = False
        self.speed = 1

        self.setWindowTitle("Spacecraft Attitude Control — PID vs LQR")
        self.setMinimumSize(1100, 700)
        self._apply_dark_theme()
        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self._update_display()

    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {DARK_BG}; color: {TEXT_COL}; }}
            QPushButton {{
                background-color: #0f172a; color: {TEXT_COL};
                border: 1px solid #1e293b; border-radius: 5px;
                padding: 5px 12px; font-family: 'Courier New'; font-size: 11px;
            }}
            QPushButton:hover {{ background-color: #1e293b; color: #e2e8f0; }}
            QPushButton:pressed {{ background-color: #1e3a5f; }}
            QSlider::groove:horizontal {{ background: #0f172a; height: 4px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #3b82f6; width: 14px; height: 14px;
                                          margin: -5px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: #1e40af; border-radius: 2px; }}
            QComboBox {{ background-color: #0f172a; color: {TEXT_COL}; border: 1px solid #1e293b;
                         border-radius: 4px; padding: 3px 8px; font-family: 'Courier New'; font-size: 11px; }}
            QLabel {{ color: {TEXT_COL}; }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("SPACECRAFT ATTITUDE CONTROL  ·  3-AXIS STABILIZATION")
        title.setStyleSheet("color: #3b82f6; font-size: 12px; font-family: 'Courier New'; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        # Controller toggle
        self.btn_pid = QPushButton("PID")
        self.btn_lqr = QPushButton("LQR")
        self.btn_pid.clicked.connect(lambda: self._set_controller("PID"))
        self.btn_lqr.clicked.connect(lambda: self._set_controller("LQR"))
        header.addWidget(QLabel("Controller:"))
        header.addWidget(self.btn_pid)
        header.addWidget(self.btn_lqr)

        # Speed
        header.addWidget(QLabel("  Speed:"))
        self.combo_speed = QComboBox()
        for s in ["0.5×", "1×", "2×", "5×"]:
            self.combo_speed.addItem(s)
        self.combo_speed.setCurrentIndex(1)
        self.combo_speed.currentIndexChanged.connect(self._set_speed)
        header.addWidget(self.combo_speed)

        main_layout.addLayout(header)

        # ── Body ──
        body = QSplitter(Qt.Horizontal)

        # Left: Telemetry
        self.telemetry = TelemetryPanel()
        self.telemetry.setFixedWidth(185)
        body.addWidget(self.telemetry)

        # Centre: 3D view
        self.gl_widget = SpacecraftGL()
        body.addWidget(self.gl_widget)

        # Right: Charts
        self.chart = ChartWidget(self.pid_frames, self.lqr_frames)
        self.chart.setMinimumWidth(280)
        body.addWidget(self.chart)

        body.setSizes([185, 520, 340])
        main_layout.addWidget(body, stretch=1)

        # ── Playback bar ──
        playback = QHBoxLayout()

        self.btn_reset = QPushButton("⟪ Reset")
        self.btn_play  = QPushButton("▶ Play")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_play.clicked.connect(self._toggle_play)
        playback.addWidget(self.btn_reset)
        playback.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.pid_frames) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._slider_moved)
        playback.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("T = 0.0s")
        self.lbl_time.setStyleSheet("font-family: 'Courier New'; font-size: 11px; min-width: 80px;")
        playback.addWidget(self.lbl_time)

        main_layout.addLayout(playback)
        self._highlight_active_controller()

    def _highlight_active_controller(self):
        pid_style = f"background-color: {ACCENT1}; color: white; border: none;" if self.controller == "PID" else ""
        lqr_style = f"background-color: {ACCENT2}; color: white; border: none;" if self.controller == "LQR" else ""
        self.btn_pid.setStyleSheet(pid_style)
        self.btn_lqr.setStyleSheet(lqr_style)

    def _set_controller(self, ctrl):
        self.controller = ctrl
        self._highlight_active_controller()
        self._update_display()

    def _set_speed(self, idx):
        speeds = [0.5, 1, 2, 5]
        self.speed = speeds[idx]
        if self.playing:
            self.timer.start(int(100 / self.speed))

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("⏸ Pause")
            self.timer.start(int(100 / self.speed))
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def _reset(self):
        self.playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()
        self.current_frame_idx = 0
        self.slider.setValue(0)
        self._update_display()

    def _tick(self):
        if self.current_frame_idx >= len(self.pid_frames) - 1:
            self._toggle_play()
            return
        self.current_frame_idx += 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self._update_display()

    def _slider_moved(self, val):
        self.current_frame_idx = val
        self._update_display()

    def _update_display(self):
        idx = self.current_frame_idx
        frames = self.pid_frames if self.controller == "PID" else self.lqr_frames
        frame = frames[idx]

        # 3D
        self.gl_widget.set_quaternion(frame['q'])

        # Telemetry
        self.telemetry.update_frame(frame, self.controller)

        # Charts cursor
        self.chart.update_cursor(frame['t'], self.controller)

        # Time label
        self.lbl_time.setText(f"T = {frame['t']:.1f}s")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  SPACECRAFT ATTITUDE CONTROL VISUALIZER")
    print("=" * 55)
    print("\nRunning simulations (this takes ~10 seconds)...")

    q_des = np.array([0., 0., 0., 1.])
    pid_ctrl = PIDController(q_des=q_des)
    lqr_ctrl = LQRController(q_des=q_des)

    print("  [1/2] PID simulation...")
    pid_sol = run_simulation(pid_ctrl)
    pid_frames = extract_frames(pid_sol, pid_ctrl)

    print("  [2/2] LQR simulation...")
    lqr_sol = run_simulation(lqr_ctrl)
    lqr_frames = extract_frames(lqr_sol, lqr_ctrl)

    print(f"\n  PID settling time: {next((f['t'] for f in reversed(pid_frames) if f['error']>2), 0):.1f}s")
    print(f"  LQR settling time: {next((f['t'] for f in reversed(lqr_frames) if f['error']>2), 0):.1f}s")
    print("\nLaunching visualizer window...")

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(pid_frames, lqr_frames)
    win.show()

    # In Spyder: exec_() blocks until window is closed
    # In terminal: use sys.exit(app.exec_())
    try:
        app.exec_()
    except SystemExit:
        pass

    print("Window closed.")


if __name__ == "__main__":
    main()