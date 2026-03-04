import sys
import re
import math
import random
import colorsys
import uuid
import numpy as np
import sympy as sp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QGridLayout, QMessageBox, QScrollArea,
    QLabel, QSizePolicy, QSplitter, QFrame, QTabWidget, QSlider,
    QInputDialog,
)
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QFont, QColor

import pyvista as pv
from pyvistaqt import QtInteractor  # pip install pyvistaqt
import vtk

from skimage.measure import marching_cubes

# ══════════════════════════════════════════════════════
#  全局配色
# ══════════════════════════════════════════════════════
BG_MAIN = "#F0F4F8"      # 浅灰背景
BG_PANEL = "#FFFFFF"     # 面板背景
BG_INPUT = "#FFFFFF"     # 输入框背景
BORDER = "#DCE4E8"       # 边框颜色 (更淡的灰)
ACCENT = "#2980B9"       # 强调色
BTN_ADD = "#3498DB"      # 蓝色按钮
BTN_PLOT = "#2ECC71"     # 绿色按钮
BTN_CLEAR = "#E74C3C"    # 红色按钮
TEXT_MAIN = "#2C3E50"    # 深色文字
TEXT_SUB = "#7F8C8D"     # 次要文字
TEXT_HINT = "#95A5A6"    # 提示文字
CLR_X = "#CC2222"        # X轴 红色 (深红)
CLR_Y = "#228822"        # Y轴 绿色 (深绿)
CLR_Z = "#1144CC"        # Z轴 蓝色 (深蓝)
SCENE_BG = "#F0F4F8"     # 3D场景背景

AXIS_LEN = 10
AXIS_NEG = 10
TICK_HALF = 0.1

# 柔和的专业配色方案
PALETTE_3D = [
    "#F1C40F", # 柔和黄
    "#E67E22", # 橙色
    "#E74C3C", # 红色
    "#9B59B6", # 紫色
    "#3498DB", # 蓝色
    "#1ABC9C", # 青色
    "#2ECC71", # 绿色
    "#34495E", # 深蓝灰
]


def _rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_random_color(existing_colors=None):
    """生成随机颜色，并尽量避开已存在的颜色（RGB空间距离检查）"""
    if existing_colors is None:
        existing_colors = []

    existing_rgb = []
    for c in existing_colors:
        try:
            if not c: continue
            c_str = c.lstrip('#')
            if len(c_str) == 6:
                r = int(c_str[0:2], 16) / 255.0
                g = int(c_str[2:4], 16) / 255.0
                b = int(c_str[4:6], 16) / 255.0
                existing_rgb.append((r, g, b))
        except:
            pass

    best_color = None
    max_min_dist = -1

    # 尝试 50 次，寻找与现有颜色距离最大的颜色
    # 如果现有颜色为空，这就相当于随机生成
    for _ in range(50):
        h = random.random()
        s = 0.6 + random.random() * 0.4  # 饱和度 0.6-1.0
        v = 0.7 + random.random() * 0.3  # 亮度 0.7-1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        if not existing_rgb:
            return _rgb_to_hex((r, g, b))

        min_dist = float('inf')
        for er, eg, eb in existing_rgb:
            dist = math.sqrt((r-er)**2 + (g-eg)**2 + (b-eb)**2)
            if dist < min_dist:
                min_dist = dist

        if min_dist > 0.25: # 距离足够远，直接接受
            return _rgb_to_hex((r, g, b))

        if min_dist > max_min_dist:
            max_min_dist = min_dist
            best_color = (r, g, b)

    # 如果找不到足够远的，返回目前找到的最远的
    if best_color:
        return _rgb_to_hex(best_color)

    # Fallback
    return _rgb_to_hex((r, g, b))


def sympy_to_label(eq_text):
    s = eq_text.replace('**', '^')

    # 构造上标映射表（只对数字做上标，+、= 等仍保持在基线）
    sup_map = str.maketrans({
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    })

    def repl(match):
        # match.group(1) 是 ^ 后面的一串数字
        txt = match.group(1)
        return txt.translate(sup_map)

    # 只把 ^ + 纯数字 变成上标，避免把 +、= 也抬高
    s = re.sub(r'\^([0-9]+)', repl, s)
    return s


def _nice_step(dist):
    """根据相机距离返回 nice 刻度步长，保证可见范围内 4~10 个刻度。"""
    visible = max(dist * 0.45, 1e-9)
    raw = visible / 6.0
    mag = 10 ** math.floor(math.log10(raw))
    n = raw / mag
    if n < 1.5:
        return 1.0 * mag
    elif n < 3.5:
        return 2.0 * mag
    elif n < 7.5:
        return 5.0 * mag
    else:
        return 10.0 * mag




# ══════════════════════════════════════════════════════
#  Qt 浮动图例（不依赖 PyVista 内置图例）
# ══════════════════════════════════════════════════════
class LegendOverlay(QFrame):
    MARGIN = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LegendOverlay")
        self.setAttribute(Qt.WA_NativeWindow, True) # 必须作为原生窗口才能覆盖在 VTK 上
        self.setAttribute(Qt.WA_TranslucentBackground, True) # 支持半透明

        # 使用纯绘制模式，不使用子控件 Layout
        self._entries = []
        self.hide()

    def update_entries(self, entries):
        self._entries = entries
        if not entries:
            self.hide()
            return

        # 计算所需尺寸
        from PyQt5.QtGui import QFontMetrics
        fm_title = QFontMetrics(QFont("Segoe UI", 10, QFont.Bold))
        fm_item = QFontMetrics(QFont("Consolas", 10))

        width = fm_title.width("方程列表") + 32
        height = 12 + fm_title.height() + 12

        for text, _ in entries:
            w = 16 + 16 + 7 + fm_item.width(text) + 16
            if w > width:
                width = w
            height += fm_item.height() + 6

        height += 6 # padding bottom

        self.resize(width, height)
        self._reposition()
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event):
        """手动绘制半透明背景和所有内容，确保 NativeWindow 下可见"""
        from PyQt5.QtGui import QPainter, QBrush, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 背景
        painter.setBrush(QBrush(QColor(255, 255, 255, 235)))
        painter.setPen(QPen(QColor("#BDC3C7"), 1.0))
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

        # 2. 标题
        painter.setPen(QColor(TEXT_SUB))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(16, 12 + painter.fontMetrics().ascent(), "方程列表")

        # 3. 条目
        y = 12 + painter.fontMetrics().height() + 12
        painter.setFont(QFont("Consolas", 10))
        fm = painter.fontMetrics()
        ascent = fm.ascent()

        for text, color_hex in self._entries:
            # 颜色块
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color_hex)))
            painter.drawRoundedRect(16, y + 2, 14, 14, 3, 3)

            # 文本
            painter.setPen(QColor(TEXT_MAIN))
            painter.drawText(16 + 14 + 8, y + ascent, text)

            y += fm.height() + 6

    def _reposition(self):
        # 固定右上角，不可拖动
        if self.parent():
            margin = 20
            x = self.parent().width() - self.width() - margin
            y = margin
            self.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # self._reposition()


# ══════════════════════════════════════════════════════
#  3D 画布（全部崩溃修复）
# ══════════════════════════════════════════════════════
class Canvas3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── 渲染锁：防止定时器与 draw_equations 竞争 ──
        # 【Fix-3】draw_equations 执行期间置 True，定时器回调检查后跳过
        self._busy = False

        # ── 直接持有 actor 引用（而非字符串名称查找）──
        # 【Fix-4】避免 name 查找不可靠的问题
        self._eq_actors = {}  # dict {eq_id: vtkActor}
        self._eq_colors = {}  # dict {eq_id: hex_color} 用于图例显示的实际颜色
        self._eq_meshes = {}  # dict {eq_id: pv.PolyData} 用于交线计算（Fix 崩溃根因）
        self._tick_geom_actor = None  # 刻度线 actor
        self._tick_label_actor = None  # 刻度文字 actor

        self._tick_step = -1.0  # 上次刻度步长，-1 强制首次刷新

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 1. VTK 渲染器 (占据所有空间)
        # multi_samples 设为 8 以开启抗锯齿，显著提升线条平滑度，同时在现代 GPU 上保持高性能
        self.plotter = QtInteractor(parent=self, auto_update=False, multi_samples=8)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 【Feature】自定义滚轮缩放逻辑 (以鼠标位置为中心)
        self.plotter.wheelEvent = self._handle_wheel_event
        # 监听 Resize 事件以重新定位浮层 (图例/提示)
        self.plotter.installEventFilter(self)
        outer.addWidget(self.plotter, 1)

        # 2. 浮动图例 (作为 Canvas3D 的直接子窗口，确保层级正确)
        # 【Fix】改为 self (Canvas3D) 的子控件，避免被 VTK 内部机制遮挡
        self._legend = LegendOverlay(self)

        # 3. 右下角视角提示 (已删除)
        # (Deleted as per user request)

        # 浮动图例
        # self._legend = LegendOverlay(self) # 已移至上方
        # 确保图例在 VTK 窗口之上
        self._legend.raise_()

        self.plotter.set_background(SCENE_BG)

        # ── 定时器：创建但不启动 ─────────────────────
        # 【Fix-1】定时器通过 showEvent/hideEvent 管理，只在可见时运行
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._refresh_cam_info)

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._update_ticks)

        # 初始化静态场景元素（不含刻度，刻度由定时器管理）
        self._init_scene()

        # ──【New】动态视图范围更新定时器 ──
        # 避免滚轮缩放时频繁重绘，使用防抖 (Debounce)
        self._view_radius = 12.0  # 初始视图半径
        self._last_focal_point = np.array([0.0, 0.0, 0.0]) # 【New】记录上一次的焦点位置
        self._view_update_timer = QTimer(self)
        self._view_update_timer.setSingleShot(True)
        self._view_update_timer.timeout.connect(self._on_view_update_timeout)

        # ──【LOD 两级质量】─────────────────────────────
        # _is_lod=True  : 滚轮滚动中，使用极低分辨率（快速预览）
        # _is_lod=False : 滚轮停止后，恢复全精度重建
        self._is_lod = False
        # 滚轮停止 600ms 后触发全精度重建
        self._zoom_end_timer = QTimer(self)
        self._zoom_end_timer.setSingleShot(True)
        self._zoom_end_timer.timeout.connect(self._on_zoom_end)

        # 缓存最后一次绘制的方程数据，用于缩放重绘
        self._last_parsed_equations = []

    # ── 可见性控制定时器生命周期（Fix-1 核心）─────
    def showEvent(self, event):
        """Canvas3D 变为可见时启动定时器（如 QStackedWidget 切换到此页）"""
        super().showEvent(event)
        if not self._cam_timer.isActive():
            self._cam_timer.start(150)
        if not self._tick_timer.isActive():
            self._tick_timer.start(250)

    def hideEvent(self, event):
        """Canvas3D 被隐藏时停止定时器，避免对不可见 VTK 窗口渲染"""
        super().hideEvent(event)
        self._cam_timer.stop()
        self._tick_timer.stop()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 当窗口大小时，确保图例位置正确并置顶
        if hasattr(self, '_legend'):
            self._legend._reposition()
            self._legend.raise_()

    def eventFilter(self, obj, event):
        if obj == self.plotter and event.type() == QEvent.Resize:
            # 当 plotter 大小改变时，更新浮层位置
            if hasattr(self, '_legend'):
                self._legend._reposition()

        return super().eventFilter(obj, event)

    # ── 相机信息刷新 ──────────────────────────────
    def _refresh_cam_info(self):
        if self._busy:
            return
        try:
            # (Deleted as per user request)
            pass
        except Exception:
            pass

    # ──【New】视图范围动态更新逻辑 ──
    def _on_view_update_timeout(self):
        """
        防抖触发。
        - 滚动中 (_is_lod=True)：只更新内部视图半径，完全不重建任何 mesh。
          相机已经移动，VTK 直接重渲染已有 actor，纯 GPU，零卡顿。
        - 停止后由 _on_zoom_end 负责全量重建。
        """
        if self._busy:
            return
        try:
            pos = self.plotter.camera_position
            p1 = np.array(pos[0])
            p2 = np.array(pos[1])
            dist = float(np.linalg.norm(p1 - p2))
            self._view_radius = max(6.0, dist * 0.45)
            self._last_focal_point = np.array(self.plotter.camera.GetFocalPoint())
        except Exception as e:
            print(f"[View Radius Update Error] {e}")

    def _on_zoom_end(self):
        """
        滚轮/拖拽停止 600ms 后触发全量重建：
          1. 更新网格/坐标轴（与新 view_radius 匹配）
          2. 重建所有曲面 mesh（force_update=True，全精度 RES）
          3. 50ms 后重建交线（此时 _eq_meshes 已是全精度）
        """
        if self._busy:
            self._zoom_end_timer.start(300)
            return
        self._is_lod = False
        try:
            r = getattr(self, '_view_radius', 12.0)
            self._update_grid_dynamic(r)
            if not (hasattr(self, '_last_parsed_equations') and self._last_parsed_equations):
                return
            # Phase 1: 全精度曲面（force_update=True 走 mapper.SetInputData 快速路径）
            self.draw_equations(self._last_parsed_equations, force_update=True)
            # Phase 2: 重建交线（用全精度 mesh）
            QTimer.singleShot(80, self._rebuild_intersections_only)
        except Exception as e:
            print(f"[Zoom End Error] {e}")

    def _rebuild_intersections_only(self):
        """单独重建交线，不重建曲面（供 _on_zoom_end Phase2 使用）"""
        if self._busy or not (hasattr(self, '_last_parsed_equations') and self._last_parsed_equations):
            return
        # 清除旧交线
        for actor in getattr(self, '_intersection_actors', []):
            try: self.plotter.remove_actor(actor)
            except: pass
        self._intersection_actors = []
        # 触发一次 force_update=False → 走完整的交线计算路径
        self.draw_equations(self._last_parsed_equations, force_update=False)

    def _update_grid_dynamic(self, radius):
        """根据视图半径更新网格线范围"""
        try:
            # 网格范围：刚好覆盖视野即可，不需要更大
            # 旧值 min(120, radius*2.0) 在 radius=60 时创建 240 单位范围、数百条线
            # 新值：最多超出视野 30%，上限 60，大幅减少线数量
            grid_range = min(60.0, radius * 1.3)

            # 动态步长
            major_step = 1.0
            if radius > 15: major_step = 5.0
            if radius > 50: major_step = 10.0
            if radius > 100: major_step = 20.0

            minor_step = major_step / 2.0
            if major_step >= 5.0: minor_step = major_step / 5.0

            # 1. 主网格：深灰，加粗 (GeoGebra: #DCE4E8 for grid, but we need visible)
            # GeoGebra 主网格较深，次网格很淡
            self._add_grid_lines(step=major_step, color="#BDC3C7", opacity=0.8, lw=1.5,
                                 name="_grid_major", grid_range=grid_range)
            # 2. 次网格：浅灰，细
            self._add_grid_lines(step=minor_step, color="#DCE4E8", opacity=0.5, lw=0.5,
                                 name="_grid_minor", grid_range=grid_range)

            # 3. 更新 XOY 平面大小
            # _xoy_plane 是 pv.Plane，无法动态改大小，必须重建
            plane = pv.Plane(i_size=grid_range * 2, j_size=grid_range * 2, i_resolution=2, j_resolution=2)
            # GeoGebra 风格 XOY 平面：稍微加深颜色以便可见
            self.plotter.add_mesh(plane, color="#CBD5E1", opacity=0.4,
                                  show_edges=False, lighting=False, name="_xoy_plane")

            # 4. 动态更新坐标轴长度 (同步缩放)
            self._update_axes_dynamic(radius)

        except Exception as e:
            print(f"[Grid Update Error] {e}")

    def _update_axes_dynamic(self, radius):
        """根据视图半径动态调整坐标轴长度和箭头大小"""
        # 基础长度至少 10，或者随 radius 增长 (e.g. 1.2 * radius)
        # 这样当缩放时，坐标轴总是充斥大部分视野，不会显得太小
        axis_len = max(10.0, radius * 1.1)
        self._current_axis_len = axis_len

        # 箭头大小也需要随之缩放，否则在远视图下看不见
        # 设定为 axis_len 的 5% 左右，且有下限
        arr_sz = max(0.5, axis_len * 0.05)

        for direction, color, label in [
            ([1, 0, 0], CLR_X, 'x'),
            ([0, 1, 0], CLR_Y, 'y'),
            ([0, 0, 1], CLR_Z, 'z'),
        ]:
            d = np.array(direction, dtype=float)

            # 轴线：范围 [-axis_len, axis_len]
            line = pv.Line((-d * axis_len).tolist(),
                           (d * axis_len).tolist(), resolution=1)
            self.plotter.add_mesh(line, color=color, line_width=3.0,
                                  lighting=False, name=f"_axisline_{label}")

            # 箭头：位置随 axis_len 变化
            arrow = pv.Arrow(
                start=(d * axis_len).tolist(),
                direction=direction,
                tip_length=0.2, tip_radius=0.1,
                shaft_radius=0.03, scale=arr_sz,
            )
            self.plotter.add_mesh(arrow, color=color, lighting=True,
                                  name=f"_arrow_{label}")

            # 轴标签
            try:
                # 标签位置在箭头尖端外侧
                label_pos = (d * (axis_len + arr_sz)).tolist()
                self.plotter.add_point_labels(
                    np.array([label_pos]),
                    [label],
                    font_size=20, bold=True,
                    text_color=color,
                    always_visible=True,
                    show_points=False,
                    shape=None,
                    name=f"_label_{label}",
                )
            except TypeError:
                pass

    # ── 自定义滚轮缩放 ──────────────────────────────
    def _handle_wheel_event(self, event):
        """实现 GeoGebra 风格的以鼠标为中心缩放"""
        angle = event.angleDelta().y()
        if angle == 0: return

        # 向上滚动 (>0) 放大 -> factor > 1
        # 向下滚动 (<0) 缩小 -> factor < 1
        factor = 1.1 if angle > 0 else 0.9

        # 1. 获取 VTK 渲染器
        renderer = self.plotter.renderer
        camera = renderer.GetActiveCamera()

        # 2. 坐标转换：Qt 逻辑坐标 -> VTK 物理坐标 (适配 High-DPI)
        w_vtk, h_vtk = self.plotter.ren_win.GetSize()
        w_qt, h_qt = self.plotter.width(), self.plotter.height()
        if w_qt == 0 or h_qt == 0: return

        scale_x = w_vtk / w_qt
        scale_y = h_vtk / h_qt

        vtk_x = event.x() * scale_x
        vtk_y = (h_qt - event.y()) * scale_y

        # 3. 计算缩放中心 P (世界坐标)
        picker = vtk.vtkPropPicker()
        if picker.Pick(vtk_x, vtk_y, 0, renderer):
            # 优先拾取几何体表面点
            P = np.array(picker.GetPickPosition())
        else:
            # 未拾取到物体，使用鼠标射线与焦点平面的交点
            focal_pt = np.array(camera.GetFocalPoint())

            # 使用更稳定的平面投影方法
            try:
                # 获取焦点平面参数
                view_vec = focal_pt - np.array(camera.GetPosition())
                norm = np.linalg.norm(view_vec)
                if norm > 1e-9:
                    view_vec /= norm
                else:
                    view_vec = np.array([0, 0, -1])

                # 创建平面
                plane = vtk.vtkPlane()
                plane.SetOrigin(focal_pt)
                plane.SetNormal(view_vec)

                # 获取射线
                renderer.SetDisplayPoint(vtk_x, vtk_y, 0)
                renderer.DisplayToWorld()
                w1 = np.array(renderer.GetWorldPoint()[:3])

                renderer.SetDisplayPoint(vtk_x, vtk_y, 1)
                renderer.DisplayToWorld()
                w2 = np.array(renderer.GetWorldPoint()[:3])

                t = vtk.mutable(0.0)
                p_out = [0.0, 0.0, 0.0]
                if plane.IntersectWithLine(w1, w2, t, p_out):
                    P = np.array(p_out)
                else:
                    P = focal_pt
            except:
                P = focal_pt

        # 4. 更新相机位置
        cam_pos = np.array(camera.GetPosition())
        cam_foc = np.array(camera.GetFocalPoint())

        new_pos = P + (cam_pos - P) / factor
        new_foc = P + (cam_foc - P) / factor

        camera.SetPosition(new_pos)
        camera.SetFocalPoint(new_foc)

        renderer.ResetCameraClippingRange()
        self.plotter.render()
        self._refresh_cam_info()

        # 5. 进入 LOD 模式，启动两级定时器
        # LOD 定时器（120ms防抖）：滚动中快速用低精度预览
        # 结束定时器（600ms）：停止后触发全精度重建
        self._is_lod = True
        self._view_update_timer.start(120)
        self._zoom_end_timer.start(600)

    # ════════════════════════════════════════════
    #  场景初始化（仅在 __init__ 和 clear_canvas 调用）
    # ════════════════════════════════════════════
    def _init_scene(self):
        """初始化静态场景（坐标轴、网格等）"""
        self.plotter.clear()

        # 【Interaction】添加交互监听 (鼠标悬停 + 实时视角更新)
        if not hasattr(self, '_observer_added'):
            self.plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, self._on_mouse_move)

            def _on_interaction(obj, event):
                self._refresh_cam_info()
                # 拖拽时同样进入 LOD 模式，停止后恢复全精度
                self._is_lod = True
                self._view_update_timer.start(200)
                self._zoom_end_timer.start(600)

            self.plotter.iren.add_observer(vtk.vtkCommand.InteractionEvent, _on_interaction)
            self._observer_added = True
        self._last_picked_actor = None

        # 清空所有 actor 引用
        self._eq_actors = {}
        self._eq_colors = {}
        self._eq_meshes = {}
        self._intersection_actors = []
        self._tick_geom_actor = None
        self._tick_label_actor = None
        self._tick_step = -1.0

        # ──【Visual Optimization】背景与网格 ──
        self.plotter.set_background(SCENE_BG)

        # 1. XOY 参考平面（无限大感，半透明）
        plane = pv.Plane(i_size=40, j_size=40, i_resolution=2, j_resolution=2)
        # GeoGebra 风格 XOY 平面：稍微加深颜色以便可见
        self.plotter.add_mesh(plane, color="#CBD5E1", opacity=0.4,
                              show_edges=False, lighting=False, name="_xoy_plane")

        # 2. 主次网格线
        # 主网格：深灰，加粗
        self._add_grid_lines(step=1.0, color=BORDER, opacity=1.0, lw=1.5,
                             name="_grid_major")
        # 次网格：浅灰，细
        self._add_grid_lines(step=0.5, color=BORDER, opacity=0.4, lw=0.5,
                             name="_grid_minor")

        # 3. 坐标轴线 + 箭头 + 标签
        # 坐标轴移至原点 (0,0,0)，并根据初始视野设置长度
        self._update_axes_dynamic(10.0)

        # 5. 原点标记 O
        try:
            self.plotter.add_point_labels(
                np.array([[0.0, 0.0, 0.0]]),
                ['O'],
                font_size=14,
                text_color=TEXT_SUB,
                always_visible=True,
                show_points=True,
                point_color=TEXT_SUB,
                point_size=6,
                shape=None,
                name="_origin_label",
            )
        except Exception:
            pass

        # 6. 初始相机视角优化
        self.plotter.camera_position = [(12, -21, 14), (0, 0, 0), (0, 0, 1)]
        self.plotter.reset_camera()
        self.plotter.camera_position = [(12, -21, 14), (0, 0, 0), (0, 0, 1)]

        # 7. 左下角方位导航器 (Orientation Widget)
        # 恢复左下角的 XYZ 轴向指示器，方便用户辨识方向
        try:
            # interactive=True 允许用户点击拖拽

            self.plotter.add_axes(interactive=True, line_width=2, color='black')
        except Exception:
            pass



        self.plotter.render()

    def reset_view(self):
        """重置视角到初始状态"""
        self.plotter.camera_position = [(12, -21, 14), (0, 0, 0), (0, 0, 1)]
        self.plotter.render()

    def _on_mouse_move(self, obj, event):
        """鼠标悬停高亮几何体"""
        try:
            x, y = self.plotter.iren.GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(x, y, 0, self.plotter.renderer)
            actor = picker.GetActor()

            # 仅针对方程几何体进行高亮（忽略坐标轴、网格等）
            if actor and actor not in self._eq_actors.values():
                actor = None

            if actor == self._last_picked_actor:
                return

            # 1. 恢复上一个高亮对象
            if self._last_picked_actor:
                try:
                    prop = self._last_picked_actor.GetProperty()
                    prop.SetAmbient(self._last_ambient)
                except:
                    pass

            self._last_picked_actor = actor

            # 2. 高亮当前对象
            if actor:
                prop = actor.GetProperty()
                self._last_ambient = prop.GetAmbient()
                prop.SetAmbient(0.6) # 提高环境光使其发亮

            self.plotter.render()
        except Exception:
            pass

    def _add_grid_lines(self, step, color, opacity, lw, name, grid_range=15):
        pts, cells, ci = [], [], 0
        # 绘制 XOY 平面上的网格
        for v in np.arange(-grid_range, grid_range + step * 0.01, step):
            v = round(v, 8)
            # 平行于 X 轴的线 (y=v)
            pts += [[-grid_range, v, 0], [grid_range, v, 0]]
            cells += [2, ci, ci + 1];
            ci += 2
            # 平行于 Y 轴的线 (x=v)
            pts += [[v, -grid_range, 0], [v, grid_range, 0]]
            cells += [2, ci, ci + 1];
            ci += 2

        poly = pv.PolyData()
        poly.points = np.array(pts, dtype=float)
        poly.lines = np.array(cells, dtype=int)
        self.plotter.add_mesh(poly, color=color, opacity=opacity,
                              line_width=lw, lighting=False, name=name)

    # ════════════════════════════════════════════
    #  动态刻度（安全版）
    # ════════════════════════════════════════════
    def _update_ticks(self):
        """
        【Fix-1/3】仅在非忙状态且 widget 可见时刷新刻度。
        """
        if self._busy or not self.isVisible():
            return
        try:
            pos = self.plotter.camera_position
            dist = float(np.linalg.norm(
                np.array(pos[0], dtype=float) - np.array(pos[1], dtype=float)))
            if dist < 1e-6:
                return
        except Exception:
            return

        step = _nice_step(dist)
        if self._tick_step > 0 and abs(step - self._tick_step) / self._tick_step < 0.01:
            return  # 变化不足 1%，跳过

        self._tick_step = step
        try:
            self._rebuild_ticks(step)
        except Exception as e:
            print(f"[tick error] {e}")

    def _rebuild_ticks(self, step):
        """
        精确替换刻度几何体和标签 actor。
        【Fix-2/4】直接持有并传递 actor 对象，而非字符串查找。
        【Fix-2】所有 add_mesh/add_point_labels 不传 render 参数，最后统一 render()。
        """
        axis_len = getattr(self, '_current_axis_len', AXIS_LEN)
        extent = axis_len - 0.5
        tick_vals = []
        v = math.ceil(-extent / step) * step
        while v <= extent + 1e-9:
            val = round(v, 10)
            if abs(val) > step * 0.01:
                tick_vals.append(val)
            v += step

        if not tick_vals:
            return

        # 动态调整刻度线长度
        # 基础 TICK_HALF=0.1，当 step 很大时需要按比例放大
        TH = max(TICK_HALF, step * 0.08)
        LO = TH * 3.2

        # ── 刻度线几何体 ───────────────────────
        pts, cells, ci = [], [], 0
        for v in tick_vals:
            pts += [[v, 0, -TH], [v, 0, TH]];
            cells += [2, ci, ci + 1];
            ci += 2
            pts += [[0, v, -TH], [0, v, TH]];
            cells += [2, ci, ci + 1];
            ci += 2
            pts += [[-TH, 0, v], [TH, 0, v]];
            cells += [2, ci, ci + 1];
            ci += 2

        poly = pv.PolyData()
        poly.points = np.array(pts, dtype=float)
        poly.lines = np.array(cells, dtype=int)

        # 删除旧 actor（直接传 actor 对象，不用字符串，Fix-4）
        if self._tick_geom_actor is not None:
            try:
                self.plotter.remove_actor(self._tick_geom_actor)
            except Exception:
                pass
            self._tick_geom_actor = None

        self._tick_geom_actor = self.plotter.add_mesh(
            poly, color=TEXT_SUB, opacity=0.85,
            line_width=1.5, lighting=False,
        )

        # ── 刻度标签 ───────────────────────────
        fmt = ("{:.0f}" if step >= 1.0 - 1e-9
               else "{:.1f}" if step >= 0.1 - 1e-9
               else "{:.2f}")

        label_pts, label_texts = [], []
        for v in tick_vals:
            s = fmt.format(v)
            label_pts.append([v, -LO * 0.5, -LO]);
            label_texts.append(s)  # X 轴
            label_pts.append([LO * 0.8, v, -LO]);
            label_texts.append(s)  # Y 轴
            label_pts.append([-LO * 1.2, 0, v]);
            label_texts.append(s)  # Z 轴

        if self._tick_label_actor is not None:
            try:
                self.plotter.remove_actor(self._tick_label_actor)
            except Exception:
                pass
            self._tick_label_actor = None

        try:
            # 【Fix-2】不传 render 参数
            self._tick_label_actor = self.plotter.add_point_labels(
                np.array(label_pts, dtype=float),
                label_texts,
                font_size=16, # 字体大小适中
                bold=False,
                text_color=TEXT_MAIN,
                always_visible=False,
                show_points=False,
                shape=None,
            )
        except TypeError:
            # 降级：去掉可能不支持的参数
            try:
                self._tick_label_actor = self.plotter.add_point_labels(
                    np.array(label_pts, dtype=float),
                    label_texts,
                    font_size=14,
                    always_visible=False,
                    show_points=False,
                )
            except Exception:
                pass

        self.plotter.render()

    # ════════════════════════════════════════════
    #  交线 Actor 样式统一设置
    # ════════════════════════════════════════════
    def _style_intersection_actor(self, actor):
        """
        强制交线永远显示为纯黑色，不受任何光照/视角影响。

        PyVista add_mesh(lighting=False) 只关闭 VTK 的光照计算，
        但 actor.GetProperty() 上的 Ambient/Diffuse/Specular 系数
        仍然存在，当场景有多光源或深度渐变时会产生颜色偏移。
        这里从 VTK 属性层彻底锁定：
          - Ambient=1, Diffuse=0, Specular=0 → 颜色 100% 来自 AmbientColor
          - SetLighting(False)               → 跳过整个光照管线
          - SetColor(0,0,0)                  → 强制黑色
        """
        if actor is None:
            return
        try:
            prop = actor.GetProperty()
            prop.SetColor(0.0, 0.0, 0.0)        # 纯黑
            prop.SetAmbient(1.0)                 # 全环境光 = 不受灯光方向影响
            prop.SetDiffuse(0.0)                 # 无漫反射
            prop.SetSpecular(0.0)                # 无高光
            prop.SetLighting(False)              # 彻底关闭 VTK 光照管线
            prop.SetOpacity(1.0)                 # 完全不透明
        except Exception:
            pass
        # Z-fighting 偏移（前置到这里统一管理）
        try:
            mapper = actor.GetMapper()
            if mapper:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -66)
        except Exception:
            pass

    # ════════════════════════════════════════════
    #  方程绘制（不调用 _init_scene，不 clear）
    # ════════════════════════════════════════════
    def draw_equations(self, parsed_list_with_ids, force_update=False):
        """
        【Refactor】增量更新方程 actor，并计算几何体交线。
        parsed_list_with_ids: [(eq_id, parsed_dict), ...]
        force_update: 是否强制重新生成网格（用于视图范围更新）
        """
        self._busy = True
        self._last_parsed_equations = parsed_list_with_ids # 【New】缓存以备重绘
        try:
            current_ids = set()
            new_data_map = {}

            # 1. 解析输入数据
            for eq_id, p in parsed_list_with_ids:
                current_ids.add(eq_id)
                new_data_map[eq_id] = p

            # 2. 移除不再存在的方程 actor
            existing_ids = list(self._eq_actors.keys())
            for eid in existing_ids:
                if eid not in current_ids:
                    actor = self._eq_actors.pop(eid)
                    self._eq_meshes.pop(eid, None)   # 【Fix】同步清除 mesh 引用
                    if eid in self._eq_colors:
                        del self._eq_colors[eid]
                    if actor is not None:
                        try:
                            actor.SetVisibility(False)
                            if actor.GetMapper():
                                self.plotter.remove_actor(actor)
                        except Exception as e:
                            print(f"[remove error] {e}")

            # 3. 清除旧的交线 actor
            if hasattr(self, '_intersection_actors'):
                for actor in self._intersection_actors:
                    try:
                        self.plotter.remove_actor(actor)
                    except:
                        pass
                self._intersection_actors.clear()
            else:
                self._intersection_actors = []

            # 4. 添加/更新方程 actor
            for eq_id, p in new_data_map.items():
                # 【Optimized】如果是强制更新 (View Range Change)，则尝试更新现有 Actor 的数据
                if force_update and eq_id in self._eq_actors:
                    try:
                        # 重新生成网格 (适应新视野)
                        new_mesh = self._build_isosurface(p)
                        if new_mesh:
                            actor = self._eq_actors[eq_id]
                            mapper = actor.GetMapper()
                            if mapper:
                                # VTK SetInputData is much faster than recreating actor
                                mapper.SetInputData(new_mesh)

                            # 更新引用
                            self._eq_meshes[eq_id] = new_mesh

                            # 确保 geom_type 属性正确传递 (用于交线计算)
                            continue
                    except Exception as e:
                        print(f"[Update Mesh Error] {e}")
                        # Fallback: recreate actor below
                        if eq_id in self._eq_actors:
                            old_actor = self._eq_actors.pop(eq_id)
                            self.plotter.remove_actor(old_actor)

                if eq_id in self._eq_actors:
                    if eq_id not in self._eq_colors:
                         self._eq_colors[eq_id] = p['color']
                    continue

                mesh = self._build_isosurface(p)
                if mesh:
                    # ──【Visual Optimization】材质与颜色 ──
                    color = p['color']  # 基础颜色来自随机分配，确保唯一性

                    # 默认参数（适用于一般曲面/Marching Cubes）
                    opacity = 0.85
                    smooth_shading = True
                    # 【Visual Optimization】统一使用专业光照参数
                    specular = 0.4
                    specular_power = 30
                    ambient = 0.3
                    diffuse = 0.7

                    # 获取几何体类型标记
                    geom_type = "general"
                    if mesh.field_data and "geom_type" in mesh.field_data:
                        try:
                            # PyVista string array return handling
                            gt = mesh.field_data["geom_type"]
                            if hasattr(gt, '__len__') and len(gt) > 0:
                                geom_type = gt[0]
                        except:
                            pass

                    # 针对特定类型的优化
                    if geom_type == "plane":
                         # 平面：半透明，避免遮挡
                         opacity = 0.35 # 进一步降低透明度
                         specular = 0.1
                         ambient = 0.6
                         diffuse = 0.6

                    elif geom_type == "cylinder":
                        # 圆柱：类似 GeoGebra 的半透明风格
                        opacity = 0.6
                        specular = 0.3
                        ambient = 0.4
                        diffuse = 0.7
                        smooth_shading = True

                    elif geom_type == "sphere":
                        # 球体：柔和，强立体感
                        opacity = 0.9
                        # 使用用户指定的参数
                        specular = 0.4
                        specular_power = 30
                        ambient = 0.3
                        diffuse = 0.7

                    elif geom_type == "general":
                        # 【Fix】一般曲面 (如圆锥、抛物面)：启用半透明
                        # GeoGebra 风格：半透明以显示内部结构和交线
                        opacity = 0.65
                        specular = 0.5
                        specular_power = 40
                        ambient = 0.4
                        diffuse = 0.7
                        smooth_shading = True

                    self._eq_colors[eq_id] = color

                    actor = self.plotter.add_mesh(
                        mesh,
                        color=color,
                        opacity=opacity,
                        smooth_shading=smooth_shading,
                        specular=specular,
                        specular_power=specular_power,
                        ambient=ambient,
                        diffuse=diffuse,
                        lighting=True,
                    )
                    self._eq_actors[eq_id] = actor
                    self._eq_meshes[eq_id] = mesh   # 【Fix】保存 pv.PolyData 引用

            # 5. 计算并绘制交线
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 【终极修复】彻底放弃 VTK intersection()。
            #
            # 原因：vtkIntersectionPolyDataFilter 对平面-平面相交时，
            # 当交线恰好穿过网格顶点/边界时会触发 C++ abort()，
            # Python try/except 完全无法捕获此类 C++ 级崩溃。
            # 4 个平面产生 C(4,2)=6 次调用，必然命中此 bug。
            #
            # 解决：平面×平面 -> 纯 NumPy 线性代数解析法，零 VTK 几何调用
            #       其他组合   -> 跳过
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            import itertools

            # 【Update】根据当前视图范围动态调整交线裁剪范围
            view_r = getattr(self, '_view_radius', 12.0)
            CLIP = view_r   # 交线裁剪范围（±view_r）

            mesh_ids = [eid for eid in current_ids
                        if eid in self._eq_meshes and eid in self._eq_actors
                        and self._eq_actors[eid].GetVisibility()]

            def _geom_type(m):
                try:
                    if m.field_data and "geom_type" in m.field_data:
                        gt = m.field_data["geom_type"]
                        if len(gt) > 0:
                            return str(gt[0])
                except Exception:
                    pass
                return "general"

            for eid1, eid2 in itertools.combinations(mesh_ids, 2):
                m1 = self._eq_meshes[eid1]
                m2 = self._eq_meshes[eid2]
                try:
                    g1 = _geom_type(m1)
                    g2 = _geom_type(m2)

                    # ── 平面×平面：纯 NumPy 解析法，不调用任何 VTK ──────
                    if g1 == "plane" and g2 == "plane":
                        if ("plane_params" not in m1.field_data or
                                "plane_params" not in m2.field_data):
                            continue

                        pp1 = m1.field_data["plane_params"]  # [nx,ny,nz,cx,cy,cz]
                        pp2 = m2.field_data["plane_params"]

                        n1  = np.array(pp1[:3], dtype=float)
                        n2  = np.array(pp2[:3], dtype=float)
                        c1  = np.array(pp1[3:6], dtype=float)
                        c2  = np.array(pp2[3:6], dtype=float)
                        d1  = float(np.dot(n1, c1))
                        d2  = float(np.dot(n2, c2))

                        # 交线方向 = n1 × n2
                        direction = np.cross(n1, n2)
                        dir_norm = np.linalg.norm(direction)
                        if dir_norm < 1e-9:
                            continue   # 平行/共面，无交线

                        d_n = direction / dir_norm

                        # 求交线上一点：令最大分量轴坐标=0，解 2×2 方程组
                        idx = int(np.argmax(np.abs(direction)))
                        cols = [0, 1, 2]
                        cols.pop(idx)
                        A2 = np.array([[n1[cols[0]], n1[cols[1]]],
                                       [n2[cols[0]], n2[cols[1]]]], dtype=float)
                        b2 = np.array([d1, d2], dtype=float)
                        try:
                            sol = np.linalg.solve(A2, b2)
                        except np.linalg.LinAlgError:
                            continue
                        pt0 = np.zeros(3)
                        pt0[cols[0]] = sol[0]
                        pt0[cols[1]] = sol[1]
                        # pt0[idx] = 0 已经是 0

                        # 把直线 pt0 + t*d_n 裁剪到 [-CLIP, CLIP]^3
                        t_min_v, t_max_v = -1e18, 1e18
                        for axis in range(3):
                            if abs(d_n[axis]) < 1e-12:
                                if abs(pt0[axis]) > CLIP + 1e-9:
                                    t_min_v = t_max_v = 0.0  # 线完全在范围外
                                    break
                            else:
                                t_lo = (-CLIP - pt0[axis]) / d_n[axis]
                                t_hi = ( CLIP - pt0[axis]) / d_n[axis]
                                if t_lo > t_hi:
                                    t_lo, t_hi = t_hi, t_lo
                                t_min_v = max(t_min_v, t_lo)
                                t_max_v = min(t_max_v, t_hi)

                        if t_max_v - t_min_v < 1e-9:
                            continue

                        # 生成线段点集 -> pv.Spline (增加采样点到 400 提升平滑度)
                        ts = np.linspace(t_min_v, t_max_v, 201)
                        line_pts = pt0[np.newaxis, :] + ts[:, np.newaxis] * d_n[np.newaxis, :]
                        spline = pv.Spline(line_pts, 400)

                        actor = self.plotter.add_mesh(
                            spline, color="#000000", line_width=2.5, lighting=False,
                        )
                        self._style_intersection_actor(actor)
                        self._intersection_actors.append(actor)

                    # ── 球面×平面：解析法 ─────────────────────
                    elif (g1 == "sphere" and g2 == "plane") or (g1 == "plane" and g2 == "sphere"):
                        if g1 == "plane":
                            m1, m2 = m2, m1  # 确保 m1 是球，m2 是面

                        if ("sphere_params" not in m1.field_data or
                            "plane_params" not in m2.field_data):
                            continue

                        # 球参数
                        sp_p = m1.field_data["sphere_params"]
                        cx, cy, cz, radius = sp_p[0], sp_p[1], sp_p[2], sp_p[3]
                        C = np.array([cx, cy, cz])

                        # 面参数
                        pp = m2.field_data["plane_params"]
                        n = np.array(pp[:3]) # 法线
                        P0 = np.array(pp[3:6]) # 平面上一点

                        # 归一化法线
                        n_norm = np.linalg.norm(n)
                        if n_norm < 1e-9: continue
                        n = n / n_norm

                        # 球心到平面的有向距离 d = n·(C - P0)
                        dist = np.dot(n, C - P0)

                        # 判断相交情况
                        if abs(dist) > radius + 1e-9:
                            continue # 相离

                        # 交线圆半径
                        r_circle = math.sqrt(max(0, radius**2 - dist**2))
                        if r_circle < 1e-9:
                            continue # 相切一点，忽略

                        # 交线圆心
                        C_circle = C - dist * n

                        # 生成圆环几何体
                        # 找到圆平面上的两个正交基 u, v
                        # 选取任意非平行于 n 的辅助向量 temp
                        if abs(n[0]) > 0.9:
                            temp = np.array([0, 1, 0])
                        else:
                            temp = np.array([1, 0, 0])

                        u = np.cross(n, temp)
                        u = u / np.linalg.norm(u)
                        v = np.cross(n, u)

                        # 生成圆周点
                        # 增加采样点到 500 以获得极度平滑的曲线
                        thetas = np.linspace(0, 2*np.pi, 501) # 闭合圆
                        # circle_pts = C_circle + r_circle * (cos * u + sin * v)
                        circle_pts = (C_circle[np.newaxis, :] +
                                      r_circle * np.outer(np.cos(thetas), u) +
                                      r_circle * np.outer(np.sin(thetas), v))

                        # 创建 PolyData 线
                        spline = pv.Spline(circle_pts, 501)

                        actor = self.plotter.add_mesh(
                            spline, color="#000000", line_width=2.5, lighting=False,
                            render=False
                        )
                        self._style_intersection_actor(actor)
                        self._intersection_actors.append(actor)

                    # ── 圆柱×平面：解析法 (简化为离散采样求交) ────────
                    elif (g1 == "cylinder" and g2 == "plane") or (g1 == "plane" and g2 == "cylinder"):
                        if g1 == "plane":
                             m1, m2 = m2, m1

                        # 确保参数存在
                        if ("cylinder_params" not in m1.field_data or
                            "plane_params" not in m2.field_data):
                            continue

                        # 取出参数
                        cp = m1.field_data["cylinder_params"]
                        pp = m2.field_data["plane_params"]

                        # 圆柱参数：中心C，方向D，半径R
                        C_cyl = np.array(cp[:3])
                        D_cyl = np.array(cp[3:6])
                        R_cyl = cp[6]

                        # 平面参数：法线n，点P0
                        n_pl = np.array(pp[:3])
                        P0_pl = np.array(pp[3:6])

                        # 检查平行：n_pl 与 D_cyl 垂直 (点积为0)
                        # 如果平行，要么无交点，要么两条线，要么相切
                        if abs(np.dot(n_pl, D_cyl)) < 1e-6:
                            # 简化处理：对于平行情况暂不绘制或交由 VTK 处理
                            continue

                        # 椭圆生成逻辑
                        # 参数化圆柱面：P(theta, h) = C + R*(u*cos + v*sin) + h*D
                        # 其中 u, v 是垂直于 D 的单位向量
                        # 代入平面方程 n·(P - P0) = 0 解出 h(theta)

                        # 构建基向量 u, v
                        if abs(D_cyl[0]) > 0.9:
                            temp = np.array([0, 1, 0])
                        else:
                            temp = np.array([1, 0, 0])
                        u = np.cross(D_cyl, temp)
                        u /= np.linalg.norm(u)
                        v = np.cross(D_cyl, u)

                        # 离散化 theta (增加采样到 400 提升精度)
                        thetas = np.linspace(0, 2*np.pi, 401)
                        cos_t = np.cos(thetas)
                        sin_t = np.sin(thetas)

                        # P_circle = C + R*(u*cos + v*sin)
                        P_circle = (C_cyl[np.newaxis, :] +
                                    R_cyl * (np.outer(cos_t, u) + np.outer(sin_t, v)))

                        # h = n·(P0 - P_circle) / (n·D)
                        numer = np.dot(P0_pl - P_circle, n_pl)
                        denom = np.dot(D_cyl, n_pl)
                        h = numer / denom

                        # 最终点 P = P_circle + h*D
                        ellipse_pts = P_circle + np.outer(h, D_cyl)

                        # 裁剪：根据 view_r 裁剪过远的交线
                        view_r = getattr(self, '_view_radius', 12.0)
                        dists = np.linalg.norm(ellipse_pts, axis=1)
                        mask = dists < view_r * 1.5 # 稍微放宽

                        if np.any(mask):
                            # 如果有点被保留，则绘制。注意 Spline 需要连续点，
                            # 如果裁剪导致不连续，Spline 会乱连。
                            # GeoGebra 风格：通常画整个椭圆。

                            spline = pv.Spline(ellipse_pts, 401)
                            actor = self.plotter.add_mesh(
                                spline, color="#000000", line_width=2.5, lighting=False,
                                render=False
                            )
                            self._style_intersection_actor(actor)
                            self._intersection_actors.append(actor)

                    # ── 球面×圆柱：完全解析法，彻底避开奇点精度问题 ──
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    # 背景：VTK mesh intersection 在球面与圆柱面内切处
                    #       （如 Viviani 曲线在 (a,0,0) 处）判别式≈0，
                    #       浮点误差导致该点被完全丢弃，曲线断成两段。
                    #
                    # 解析参数化：
                    #   圆柱面上的点：P(θ,h) = C_cyl + R_c*(u·cosθ + v·sinθ) + h·D
                    #   代入球方程 |P - S_c|² = R_s²，对每个 θ 解关于 h 的二次方程：
                    #     h² + 2(Q(θ)·D)h + (|Q(θ)|² - R_s²) = 0
                    #   其中 Q(θ) = C_cyl - S_c + R_c*(u·cosθ + v·sinθ)
                    #
                    #   判别式 Δ = (Q·D)² - |Q|² + R_s²
                    #   - Δ > 0：两个交点（曲线穿过圆柱两侧）
                    #   - Δ = 0：切点（内切奇点，解析法精确覆盖）
                    #   - Δ < 0：无交点
                    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    elif (g1 == "sphere" and g2 == "cylinder") or (g1 == "cylinder" and g2 == "sphere"):
                        if g1 == "cylinder":
                            m1, m2 = m2, m1   # m1=球, m2=柱

                        if ("sphere_params" not in m1.field_data or
                            "cylinder_params" not in m2.field_data):
                            continue

                        sp  = m1.field_data["sphere_params"]   # [cx,cy,cz,r]
                        cp  = m2.field_data["cylinder_params"] # [cx,cy,cz,dx,dy,dz,r]

                        S_c  = np.array(sp[:3], dtype=float)
                        R_s  = float(sp[3])
                        C_c  = np.array(cp[:3], dtype=float)
                        D    = np.array(cp[3:6], dtype=float)
                        R_c  = float(cp[6])

                        D_norm = np.linalg.norm(D)
                        if D_norm < 1e-9: continue
                        D = D / D_norm  # 确保单位向量

                        # 构建圆柱面的正交基 u, v（均垂直于 D）
                        tmp = np.array([1,0,0]) if abs(D[0]) < 0.9 else np.array([0,1,0])
                        u = np.cross(D, tmp); u /= np.linalg.norm(u)
                        v = np.cross(D, u)

                        # 高密度 θ 采样，保证切点附近不丢失
                        N_theta = 1200
                        thetas  = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
                        cos_t   = np.cos(thetas)
                        sin_t   = np.sin(thetas)

                        # Q(θ) = C_cyl - S_c + R_c*(u·cosθ + v·sinθ)
                        rim = R_c * (np.outer(cos_t, u) + np.outer(sin_t, v))   # (N,3)
                        Q   = (C_c - S_c)[np.newaxis, :] + rim                  # (N,3)

                        QD   = Q.dot(D)                   # (N,)  Q·D
                        QQ   = (Q * Q).sum(axis=1)        # (N,)  |Q|²
                        disc = QD**2 - QQ + R_s**2        # (N,)  判别式

                        # ── 分支1（h = -QD + sqrt(disc)）
                        # ── 分支2（h = -QD - sqrt(disc)）
                        # 对每个 θ 收集所有有效交点，再做链排序 + 平滑
                        all_pts = []
                        for branch in [+1, -1]:
                            mask  = disc >= 0
                            if not np.any(mask): continue
                            sq    = np.where(mask, np.sqrt(np.maximum(disc, 0)), 0.0)
                            h_val = -QD + branch * sq          # (N,)

                            # P = C_cyl + rim + h*D，只保留判别式≥0的点
                            pts_b = (C_c[np.newaxis,:] + rim +
                                     np.outer(h_val, D))       # (N,3)
                            valid = pts_b[mask]
                            if len(valid) >= 2:
                                all_pts.append(valid)

                        if not all_pts: continue

                        # 合并两分支点，重新排序为连续曲线
                        # 策略：对每段分支分别做最近邻排序（分支内部已近似有序），
                        # 再用边遍历光滑，避免分支间错误连接。
                        def _smooth_and_draw(pts_arr):
                            """对有序点集做 Laplacian 平滑 + Spline 绘制"""
                            n = len(pts_arr)
                            if n < 2: return
                            sm = pts_arr.copy()
                            # 判断是否闭合（首尾距离 < 平均步长 * 3）
                            avg_step = np.linalg.norm(np.diff(sm, axis=0), axis=1).mean()
                            is_closed = np.linalg.norm(sm[0] - sm[-1]) < avg_step * 3
                            N_ITER = 80
                            for _ in range(N_ITER):
                                if is_closed:
                                    # 闭合曲线：循环边界
                                    prev = np.roll(sm, 1, axis=0)
                                    nxt  = np.roll(sm, -1, axis=0)
                                    sm   = 0.4 * sm + 0.3 * (prev + nxt)
                                else:
                                    # 开放曲线：端点锚定
                                    sm[1:-1] = 0.4 * sm[1:-1] + 0.3 * (sm[:-2] + sm[2:])
                            n_sp   = max(400, n * 4)
                            spline = pv.Spline(sm, n_sp)
                            actor  = self.plotter.add_mesh(
                                spline, color="#000000",
                                line_width=2.5, lighting=False, render=False)
                            self._style_intersection_actor(actor)
                            self._intersection_actors.append(actor)

                        for branch_pts in all_pts:
                            _smooth_and_draw(branch_pts)

                    # ── 其他类型组合：尝试使用 VTK intersection() ───
                    else:
                        # 简单的包围盒预检测，避免无交集的计算
                        b1 = m1.bounds
                        b2 = m2.bounds
                        eps = 1e-5
                        if (b1[0] > b2[1] + eps or b1[1] < b2[0] - eps or
                            b1[2] > b2[3] + eps or b1[3] < b2[2] - eps or
                            b1[4] > b2[5] + eps or b1[5] < b2[4] - eps):
                            continue

                        try:
                            m1_clean = m1.clean().triangulate()
                            m2_clean = m2.clean().triangulate()
                            res = m1_clean.intersection(m2_clean, split_first=False, split_second=False)

                            if isinstance(res, (tuple, list)):
                                inter = res[0]
                            else:
                                inter = res

                            if inter and inter.n_points > 1:
                                try:
                                    inter_clean = inter.clean()
                                    pts = inter_clean.points
                                    n_pts = len(pts)
                                    if n_pts < 2:
                                        raise ValueError("too few pts")

                                    # ══════════════════════════════════════════
                                    # 【边拓扑链遍历 + 折线 Laplacian 平滑】
                                    #
                                    # 为什么必须用边拓扑而非 smooth()/KDTree：
                                    # - smooth() 是三角面片算法，对折线无效
                                    #   → 调用后点坐标根本不变 → 永远锯齿
                                    # - KDTree 度修剪会误切正确连接 → 断线
                                    #
                                    # 正确流程：
                                    # 1. 读 VTK inter.lines 边集合（O(n)，最精确）
                                    # 2. 构建邻接表
                                    # 3. 用"边已访问"控制遍历，在度≠2节点
                                    #    切断链 → 自交点作为多段链的公共端点
                                    # 4. 折线 Laplacian：端点锚定，内部点平滑
                                    #    → 自交点坐标不变，两段在此精确汇合
                                    # 5. 密集 Spline 插值，消除折线感
                                    # ══════════════════════════════════════════

                                    # Step 1: 读取所有边
                                    raw_lines = inter_clean.lines
                                    edge_set = set()
                                    ridx = 0
                                    while ridx < len(raw_lines):
                                        seg_len = int(raw_lines[ridx]); ridx += 1
                                        if seg_len == 2 and ridx + 1 < len(raw_lines):
                                            a, b = int(raw_lines[ridx]), int(raw_lines[ridx + 1])
                                            edge_set.add((min(a, b), max(a, b)))
                                        ridx += seg_len

                                    # Step 2: 邻接表
                                    adj = [[] for _ in range(n_pts)]
                                    for a, b in edge_set:
                                        adj[a].append(b)
                                        adj[b].append(a)

                                    # Step 3: 基于"边已访问"的分段链提取
                                    # 关键：度≠2 的节点（端点 deg=1 / 自交点 deg=4）
                                    # 作为链的起止点，可被多段链共享，坐标完全相同
                                    used_edges = set()
                                    chains = []

                                    def _walk(start, first_nb):
                                        chain = [start, first_nb]
                                        prev, cur = start, first_nb
                                        while True:
                                            if len(adj[cur]) != 2:
                                                break   # 端点或分叉点，停止
                                            nxt = -1
                                            for nb in adj[cur]:
                                                ek = (min(cur, nb), max(cur, nb))
                                                if nb != prev and ek not in used_edges:
                                                    nxt = nb; break
                                            if nxt == -1:
                                                break
                                            used_edges.add((min(cur, nxt), max(cur, nxt)))
                                            chain.append(nxt)
                                            prev, cur = cur, nxt
                                        return chain

                                    # 优先从端点(deg=1)和分叉点(deg≥3)出发
                                    priority = sorted(range(n_pts),
                                                      key=lambda i: (0 if len(adj[i]) == 1
                                                                     else 1 if len(adj[i]) >= 3
                                                                     else 2))
                                    for start in priority:
                                        for nb in list(adj[start]):
                                            ek = (min(start, nb), max(start, nb))
                                            if ek in used_edges:
                                                continue
                                            used_edges.add(ek)
                                            chain_idxs = _walk(start, nb)
                                            if len(chain_idxs) >= 2:
                                                chains.append(
                                                    np.array([pts[i] for i in chain_idxs],
                                                             dtype=float))

                                    if not chains:
                                        raise ValueError("no chains")

                                    # Step 4: 对每段链做折线 Laplacian 平滑
                                    # - 端点严格锚定（不移动） → 自交点坐标两段完全一致
                                    # - 内部点向邻点均值收敛 → 消除 Marching Cubes 网格边锯齿
                                    # - 80 次迭代 + alpha=0.6：足够光滑且不过度收缩
                                    N_SMOOTH = 80
                                    ALPHA    = 0.6

                                    for chain_pts in chains:
                                        n_c = len(chain_pts)
                                        sm = chain_pts.copy()
                                        if n_c > 2:
                                            for _ in range(N_SMOOTH):
                                                sm[1:-1] = ((1.0 - ALPHA) * sm[1:-1] +
                                                            ALPHA * 0.5 * (sm[:-2] + sm[2:]))

                                        # Step 5: 密集 Spline 插值
                                        n_sp = max(300, n_c * 6)
                                        spline = pv.Spline(sm, n_sp)

                                        actor = self.plotter.add_mesh(
                                            spline,
                                            color="#000000",
                                            line_width=2.5,
                                            lighting=False,
                                            render=False,
                                        )
                                        self._style_intersection_actor(actor)
                                        self._intersection_actors.append(actor)

                                except Exception as _e:
                                    print(f"[Chain Fallback] {_e}")
                                    actor = self.plotter.add_mesh(
                                        inter, color="#000000",
                                        line_width=2.5, lighting=False, render=False,
                                    )
                                    self._style_intersection_actor(actor)
                                    self._intersection_actors.append(actor)

                        except Exception as e:
                            print(f"[VTK Intersection Skipped] {e}")

                except Exception as e:
                    print(f"[Intersection Error] {e}")

            # 6. 渲染与图例更新
            self.plotter.render()

        except Exception as e:
            QMessageBox.critical(self, "绘制错误", str(e))

        finally:
            # 【Fix】无论成功失败，确保更新图例
            try:
                legend_entries = []
                # 使用 parsed_list_with_ids 保持原始顺序
                for eq_id, p in parsed_list_with_ids:
                    # 只要 ID 在 _eq_actors 中即视为存在
                    if eq_id in self._eq_actors:
                        display_color = self._eq_colors.get(eq_id, p.get('color', '#000000'))
                        try:
                            label = sympy_to_label(p['raw'])
                        except:
                            label = p.get('raw', 'Equation')
                        legend_entries.append((label, display_color))

                # 即使没有条目，也要调用 update_entries 以隐藏
                self._legend.update_entries(legend_entries)

                if legend_entries:
                    self._legend.show()
                    self._legend.raise_()
                    # 强制重新定位，防止初始位置错误
                    self._legend._reposition()
                    # 多次延时 raise，对抗 VTK 渲染可能的覆盖
                    QTimer.singleShot(100, self._legend.raise_)
                    QTimer.singleShot(300, self._legend.raise_)
            except Exception as e:
                print(f"[Legend Update Error] {e}")

            self._busy = False

    def _build_isosurface(self, p):
        """
        生成等值面网格。
        【Update】支持基于当前视图范围的动态裁剪和生成。
        """
        sd = {str(s): s for s in p['syms']}
        args = [None, None, None]
        for nm, i in [('x', 0), ('y', 1), ('z', 2)]:
            if nm in sd:
                args[i] = sd[nm]
        remaining = [s for s in p['syms'] if str(s) not in ('x', 'y', 'z')]
        for i in range(3):
            if args[i] is None:
                args[i] = (remaining.pop(0) if remaining else sp.Symbol(f'_d{i}'))

        # ──【新增】标准球体检测与参数化生成 ──
        # 优先级高于 Marching Cubes，确保完美光滑度和坐标准确性
        view_r = getattr(self, '_view_radius', 12.0)

        if all(a is not None for a in args):
            try:
                x_s, y_s, z_s = args
                # 构造多项式，检查是否为球体方程：A(x²+y²+z²) + Dx + Ey + Fz + G = 0
                poly = sp.Poly(p['expr'], x_s, y_s, z_s)

                if poly.total_degree() == 2:
                    coeffs = poly.coeffs()
                    monoms = poly.monoms()

                    # 提取系数映射 {(power_x, power_y, power_z): coeff}
                    c_map = {m: c for m, c in zip(monoms, coeffs)}

                    A = float(c_map.get((2, 0, 0), 0))
                    B = float(c_map.get((0, 2, 0), 0))
                    C = float(c_map.get((0, 0, 2), 0))

                    # 检查是否为球体：x²,y²,z² 系数相等且不为0，且无交叉项
                    # 使用 epsilon 容差比较，避免浮点误差
                    eps = 1e-7
                    is_sphere = (abs(A) > eps and
                                 abs(A - B) < eps and
                                 abs(A - C) < eps and
                                 abs(float(c_map.get((1, 1, 0), 0))) < eps and
                                 abs(float(c_map.get((1, 0, 1), 0))) < eps and
                                 abs(float(c_map.get((0, 1, 1), 0))) < eps)

                    if is_sphere:

                        D = float(c_map.get((1, 0, 0), 0))
                        E = float(c_map.get((0, 1, 0), 0))
                        F = float(c_map.get((0, 0, 1), 0))
                        G = float(c_map.get((0, 0, 0), 0))

                        # 计算球心和半径
                        # (x + D/2A)² + (y + E/2A)² + (z + F/2A)² = R²
                        # R² = (D/2A)² + (E/2A)² + (F/2A)² - G/A

                        cx = -float(D) / (2 * float(A))
                        cy = -float(E) / (2 * float(A))
                        cz = -float(F) / (2 * float(A))

                        r_sq = cx**2 + cy**2 + cz**2 - float(G)/float(A)

                        if r_sq > 0:
                            radius = math.sqrt(r_sq)
                            # 生成参数化球体
                            # 降低分辨率以在缩放时保持流畅
                            mesh = pv.Sphere(radius=radius, center=(cx, cy, cz),
                                           theta_resolution=72, phi_resolution=72)
                            # 标记几何体类型，用于渲染风格控制
                            mesh.field_data["geom_type"] = ["sphere"]
                            mesh.field_data["sphere_params"] = [cx, cy, cz, radius]
                            return mesh

                # ──【新增】圆柱体检测与参数化生成 ──
                # 检查是否为圆柱方程：只有两个变量的二次方程
                # 例如 x^2 + y^2 = R^2 (z轴圆柱), x^2 + z^2 = R^2 (y轴圆柱)
                if poly.total_degree() == 2:
                    coeffs = poly.coeffs()
                    monoms = poly.monoms()
                    c_map = {m: c for m, c in zip(monoms, coeffs)}

                    # 检查变量出现情况
                    has_x = any(m[0] > 0 for m in monoms)
                    has_y = any(m[1] > 0 for m in monoms)
                    has_z = any(m[2] > 0 for m in monoms)

                    missing_vars = []
                    if not has_x: missing_vars.append('x')
                    if not has_y: missing_vars.append('y')
                    if not has_z: missing_vars.append('z')

                    if len(missing_vars) == 1:
                        axis = missing_vars[0] # 轴向

                        # 提取非轴向系数
                        if axis == 'z':
                            A = float(c_map.get((2, 0, 0), 0)) # x^2
                            B = float(c_map.get((0, 2, 0), 0)) # y^2
                            D = float(c_map.get((1, 0, 0), 0)) # x
                            E = float(c_map.get((0, 1, 0), 0)) # y
                            G = float(c_map.get((0, 0, 0), 0)) # const
                            cross = float(c_map.get((1, 1, 0), 0)) # xy
                        elif axis == 'y':
                            A = float(c_map.get((2, 0, 0), 0)) # x^2
                            B = float(c_map.get((0, 0, 2), 0)) # z^2
                            D = float(c_map.get((1, 0, 0), 0)) # x
                            E = float(c_map.get((0, 0, 1), 0)) # z
                            G = float(c_map.get((0, 0, 0), 0)) # const
                            cross = float(c_map.get((1, 0, 1), 0)) # xz
                        else: # x
                            A = float(c_map.get((0, 2, 0), 0)) # y^2
                            B = float(c_map.get((0, 0, 2), 0)) # z^2
                            D = float(c_map.get((0, 1, 0), 0)) # y
                            E = float(c_map.get((0, 0, 1), 0)) # z
                            G = float(c_map.get((0, 0, 0), 0)) # const
                            cross = float(c_map.get((0, 1, 1), 0)) # yz

                        # 检查是否为标准圆：A=B!=0, cross=0
                        eps = 1e-7
                        if abs(A) > eps and abs(A - B) < eps and abs(cross) < eps:
                            # 计算圆心和半径
                            # (u + D/2A)^2 + (v + E/2A)^2 = R^2
                            c1 = -D / (2 * A)
                            c2 = -E / (2 * A)
                            r_sq = c1**2 + c2**2 - G/A

                            if r_sq > 0:
                                radius = math.sqrt(r_sq)
                                center = [0.0, 0.0, 0.0]
                                direction = [0.0, 0.0, 0.0]

                                if axis == 'z':
                                    center = [c1, c2, 0.0]
                                    direction = [0.0, 0.0, 1.0]
                                elif axis == 'y':
                                    center = [c1, 0.0, c2]
                                    direction = [0.0, 1.0, 0.0]
                                else:
                                    center = [0.0, c1, c2]
                                    direction = [1.0, 0.0, 0.0]

                                # 根据视图范围生成圆柱
                                # GeoGebra 风格：高度有限，随视图缩放
                                height = min(view_r * 2.2, 40.0)  # 封顶 40，防止缩小时生成巨型圆柱

                                # 【New】将圆柱中心移动到相机焦点在轴线上的投影点
                                try:
                                    fp = np.array(self.plotter.camera.GetFocalPoint())
                                    d_vec = np.array(direction)
                                    c_vec = np.array(center)
                                    t_val = np.dot(fp - c_vec, d_vec)
                                    center = c_vec + t_val * d_vec
                                except:
                                    pass

                                # 降低分辨率以减小面片数量
                                mesh = pv.Cylinder(center=center, direction=direction,
                                                 radius=radius, height=height,
                                                 resolution=48, capping=False) # 不封口

                                mesh.field_data["geom_type"] = ["cylinder"]
                                mesh.field_data["cylinder_params"] = [*center, *direction, radius]
                                return mesh

                # ──【新增】圆锥体/抛物面检测与参数化生成 ──
                # 针对 z^2 = x^2 + y^2 (圆锥) 和 z = x^2 + y^2 (抛物面)
                # 实现视锥体裁剪，避免无限延伸覆盖视图
                if poly.total_degree() == 2:
                    coeffs = poly.coeffs()
                    monoms = poly.monoms()
                    c_map = {m: c for m, c in zip(monoms, coeffs)}

                    # 提取系数
                    A = float(c_map.get((2, 0, 0), 0)) # x^2
                    B = float(c_map.get((0, 2, 0), 0)) # y^2
                    C = float(c_map.get((0, 0, 2), 0)) # z^2
                    D = float(c_map.get((1, 0, 0), 0)) # x
                    E = float(c_map.get((0, 1, 0), 0)) # y
                    F = float(c_map.get((0, 0, 1), 0)) # z
                    G = float(c_map.get((0, 0, 0), 0)) # const

                    eps = 1e-7

                    # Case 1: 圆锥 z^2 = a*x^2 + b*y^2 (C < 0, A > 0, B > 0)
                    # 简化检测：A=B, C=-A (z^2 = x^2 + y^2)
                    is_cone_z = (abs(A - B) < eps and abs(A + C) < eps and abs(A) > eps)

                    if is_cone_z:
                         # 生成圆锥网格
                         # z = ±sqrt(x^2+y^2)
                         # 范围：z in [-view_r, view_r]
                         # 也就是 r in [0, view_r]

                         res = 100
                         r_max = view_r * 1.0 # 绑定到视图半径

                         # 生成点云或网格
                         # 使用参数方程: x=r*cos(t), y=r*sin(t), z=r
                         r = np.linspace(0, r_max, res)
                         theta = np.linspace(0, 2*np.pi, res)
                         r_grid, theta_grid = np.meshgrid(r, theta)

                         X = r_grid * np.cos(theta_grid)
                         Y = r_grid * np.sin(theta_grid)
                         Z = r_grid # 上半部分

                         # 创建上半部分 mesh
                         grid_up = pv.StructuredGrid(X, Y, Z)

                         # 创建下半部分 mesh
                         grid_down = pv.StructuredGrid(X, Y, -Z)

                         mesh = grid_up.merge(grid_down)

                         # 标记几何体类型
                         mesh.field_data["geom_type"] = ["general"] # 使用通用半透明渲染
                         return mesh

                    # Case 2: 抛物面 z = a*x^2 + b*y^2 (C=0, F!=0, A>0, B>0)
                    # 简化检测：A=B, C=0, F=-1 (z = x^2 + y^2)
                    # 或者 z = c*(x^2+y^2) -> A=c, B=c, F=-1
                    is_paraboloid_z = (abs(A - B) < eps and abs(C) < eps and abs(F) > eps and abs(A) > eps)

                    if is_paraboloid_z:
                         # z = -(A/F)*x^2 - (B/F)*y^2 - G/F ...
                         # 假设标准形式 z = k*(x^2+y^2)
                         k = -A/F

                         # 范围：z in [0, view_r] -> r^2 <= view_r/k
                         # 如果 k > 0 (开口向上)
                         if k > 0:
                             z_max = view_r
                             r_max = math.sqrt(z_max / k)
                         else:
                             z_min = -view_r
                             r_max = math.sqrt(z_min / k)

                         # 限制 r_max 不要太大 (避免视野极小时渲染错误)
                         r_max = min(r_max, view_r * 1.5)

                         res = 100
                         r = np.linspace(0, r_max, res)
                         theta = np.linspace(0, 2*np.pi, res)
                         r_grid, theta_grid = np.meshgrid(r, theta)

                         X = r_grid * np.cos(theta_grid)
                         Y = r_grid * np.sin(theta_grid)
                         Z = k * (r_grid**2)

                         # 添加线性项/常数项偏移 (简化处理，假设主要在原点)
                         if abs(D) > eps: X += -D/(2*A)
                         if abs(E) > eps: Y += -E/(2*B)
                         if abs(G) > eps: Z += -G/F

                         mesh = pv.StructuredGrid(X, Y, Z)
                         mesh.field_data["geom_type"] = ["general"]
                         return mesh

                    # Case 3: Cylinder (Axis Aligned)
                    # x^2 + y^2 = r^2 (along Z) -> A=B, C=0, G < 0
                    is_cyl_z = (abs(A - B) < eps and abs(C) < eps and abs(A) > eps and G < -eps)
                    is_cyl_y = (abs(A - C) < eps and abs(B) < eps and abs(A) > eps and G < -eps)
                    is_cyl_x = (abs(B - C) < eps and abs(A) < eps and abs(B) > eps and G < -eps)

                    if (is_cyl_z or is_cyl_y or is_cyl_x) and (abs(D) < eps and abs(E) < eps and abs(F) < eps):
                        # 【Update】始终以原点为中心，高度与视图半径绑定
                        # 这样圆柱体就是固定的，不会随着相机平移而滑动
                        limit = view_r * 0.6  # 控制显示范围
                        if is_cyl_z:
                            radius = math.sqrt(-G/A)
                            direction = [0, 0, 1]
                            center = [0, 0, 0]  # 固定中心
                        elif is_cyl_y:
                            radius = math.sqrt(-G/A)
                            direction = [0, 1, 0]
                            center = [0, 0, 0]
                        else:
                            radius = math.sqrt(-G/B)
                            direction = [1, 0, 0]
                            center = [0, 0, 0]

                        # Generate Cylinder
                        # Height = limit * 1.2 (更扁平，减少体积感)
                        mesh = pv.Cylinder(center=center, direction=direction,
                                         radius=radius, height=min(limit * 1.2, 40.0),
                                         resolution=40, capped=True)

                        mesh.field_data["geom_type"] = ["cylinder"]
                        # Params: [cx,cy,cz, dx,dy,dz, r]
                        mesh.field_data["cylinder_params"] = center + direction + [radius]
                        return mesh

                # ──【新增】平面检测与参数化生成 ──
                # 检查是否为一次方程：Ax + By + Cz + D = 0
                if poly.total_degree() == 1:
                    coeffs = poly.coeffs()
                    monoms = poly.monoms()
                    c_map = {m: c for m, c in zip(monoms, coeffs)}

                    A = float(c_map.get((1, 0, 0), 0))
                    B = float(c_map.get((0, 1, 0), 0))
                    C = float(c_map.get((0, 0, 1), 0))
                    D = float(c_map.get((0, 0, 0), 0))

                    # 只要 A,B,C 不全为 0 即可
                    if not (A == 0 and B == 0 and C == 0):
                        try:
                            normal = np.array([A, B, C], dtype=float)
                            n_sq = np.dot(normal, normal)
                            if n_sq < 1e-9: raise ValueError("Normal is zero")

                            # 1. 以原点为参考，计算“最近点”作为平面中心
                            #    这样所有看起来无限大的平面都会围绕坐标原点，
                            #    而不是随着相机焦点跑到视野外只剩下一小块。
                            origin = np.array([0.0, 0.0, 0.0])
                            # 最近点公式：P0 = O - (n·O + D) * n / |n|^2
                            val0 = np.dot(normal, origin) + D
                            center = origin - (val0 / n_sq) * normal

                            # 【Fix Crash】微小抖动，避免数值退化
                            center += np.random.uniform(-1e-5, 1e-5, 3)

                            raw_size = max(6.0, min(view_r * 1.8, 40.0))
                            mesh = pv.Plane(center=center, direction=normal,
                                            i_size=raw_size, j_size=raw_size,
                                            i_resolution=2, j_resolution=2)

                            # 标记几何体类型
                            mesh.field_data["geom_type"] = ["plane"]
                            mesh.field_data["plane_params"] = [normal[0], normal[1], normal[2], center[0], center[1], center[2]]

                            return mesh
                        except Exception as e:
                            pass # Fallback to marching cubes
            except Exception as e:
                # 检测失败则回退到 Marching Cubes
                pass

        try:
            f = sp.lambdify(args, p['expr'], modules=['numpy'])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"方程转换失败: {e}")
            return None

        # 【Update】根据视图范围动态调整 Marching Cubes 范围和分辨率
        view_r = getattr(self, '_view_radius', 12.0)

        # 核心优化：严格绑定到视图范围，防止无限延伸
        # 根据相机距离动态计算包围盒，确保只绘制视野内的部分
        limit = view_r * 0.6
        gmin = -limit
        gmax = limit

        # 根据 LOD 模式动态选择分辨率
        # LOD 模式（滚动中）：RES=22，22³=10648 个格点，比全精度快约 50 倍
        # 全精度模式（停止后）：RES 48-100，保证光滑细节
        if getattr(self, '_is_lod', False):
            RES = 22
        else:
            RES = int(max(48, min(100, 180 / max(0.5, view_r))))

        lin = np.linspace(gmin, gmax, RES)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
        try:
            V = np.asarray(f(X, Y, Z), dtype=float)
            if V.shape != X.shape:
                V = np.full(X.shape, float(V.flat[0]))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"方程计算失败: {e}")
            return None

        if not (V.min() < 0 < V.max()):

            return pv.PolyData()

        try:
            sp_step = (limit * 2.0) / (RES - 1)
            verts, faces, _, _ = marching_cubes(
                V, level=0, spacing=(sp_step,) * 3)
            verts += gmin
        except Exception as e:

                QMessageBox.critical(self, "错误", f"等值面生成失败: {e}")
                return None

        n = len(faces)
        pv_faces = np.empty(n * 4, dtype=np.int_)
        pv_faces[0::4] = 3
        pv_faces[1::4] = faces[:, 0]
        pv_faces[2::4] = faces[:, 1]
        pv_faces[3::4] = faces[:, 2]
        mesh = pv.PolyData(verts.astype(float), pv_faces)

        # 1. 初始法线计算
        mesh.compute_normals(cell_normals=False, point_normals=True,
                             consistent_normals=True, inplace=True)

        # 2. 网格平滑 (Taubin Smooth 保持体积，减少收缩)
        # 参数优化：n_iter 从 30 降为 15，明显降低计算量
        try:
            # 注意：PyVista 的 filter 通常返回新对象
            mesh = mesh.smooth_taubin(n_iter=15, pass_band=0.1)
        except AttributeError:
            # 兼容旧版 PyVista 可能没有 smooth_taubin
            pass
        except Exception as e:
            print(f"[smooth error] {e}")

        # 3. 平滑后重新计算法线，确保光照正确
        mesh.compute_normals(cell_normals=False, point_normals=True,
                             consistent_normals=True, inplace=True)
        # 标记几何体类型，用于渲染风格控制
        mesh.field_data["geom_type"] = ["general"]
        return mesh

    def clear_canvas(self):
        """完全清空：停止定时器 → _init_scene → 重启定时器（如果可见）"""
        self._cam_timer.stop()
        self._tick_timer.stop()
        self._legend.hide()
        self._eq_meshes = {}  # 【Fix】提前清除，避免 _init_scene 中 plotter.clear() 后 mesh 引用悬空
        self._init_scene()
        self.plotter.render()
        if self.isVisible():
            self._cam_timer.start(150)
            self._tick_timer.start(250)

    def closeEvent(self, event):
        self._cam_timer.stop()
        self._tick_timer.stop()
        self.plotter.close()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════
#  参数管理条目
# ══════════════════════════════════════════════════════
class ParamItem(QWidget):
    """
    参数项：包含名称、当前值显示、滑动条、播放按钮和删除按钮。
    """
    def __init__(self, name, val, v_min, v_max, on_change, on_delete, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.name = name
        self.v_min = v_min
        self.v_max = v_max
        self.on_change = on_change
        self.on_delete = on_delete
        self._is_playing = False

        self.setStyleSheet(f"background:{BG_PANEL}; border-bottom:1px solid {BORDER};")

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(16, 8, 12, 8)
        main_lay.setSpacing(4)

        # 第一行：名称 = 当前值
        top_lay = QHBoxLayout()
        self.label = QLabel(f"{name} = ")
        self.label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.label.setStyleSheet(f"color:{TEXT_MAIN}; border:none;")

        # 数值输入框 (GeoGebra 风格：看起来像文字，点击可编辑)
        self.val_edit = QLineEdit(f"{val:.2f}")
        self.val_edit.setFixedWidth(80)
        self.val_edit.setFont(QFont("Consolas", 12, QFont.Bold))
        self.val_edit.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                color: {TEXT_MAIN};
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 2px;
            }}
            QLineEdit:hover {{
                background: {BG_MAIN};
                border-color: {BORDER};
            }}
            QLineEdit:focus {{
                background: white;
                border-color: {ACCENT};
            }}
        """)
        self.val_edit.returnPressed.connect(self._on_text_edited)

        # 播放按钮
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(24, 24)
        self.play_btn.setCheckable(True)
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{ 
                background:transparent; color:{ACCENT}; 
                border:1px solid {ACCENT}; border-radius:12px; font-size:12px;
            }}
            QPushButton:checked {{ background:{ACCENT}; color:white; }}
        """)
        self.play_btn.clicked.connect(self._toggle_animation)

        # 删除按钮
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(24, 24)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{TEXT_SUB};
                border:none; font-size:14px; border-radius:4px;
            }}
            QPushButton:hover {{ color:white; background:#E74C3C; }}
        """)
        del_btn.clicked.connect(lambda: self.on_delete(self.name))

        top_lay.addWidget(self.label)
        top_lay.addWidget(self.val_edit)
        top_lay.addStretch()
        top_lay.addWidget(self.play_btn)
        top_lay.addWidget(del_btn)
        main_lay.addLayout(top_lay)

        # 第二行：范围显示 + 滑动条
        btm_lay = QHBoxLayout()
        min_lbl = QLabel(str(v_min))
        min_lbl.setFont(QFont("Segoe UI", 8))
        min_lbl.setStyleSheet(f"color:{TEXT_HINT}; border:none;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000) # 0 to 1000 for 0.1 precision mapping
        self.slider.setValue(int((val - v_min) / (v_max - v_min) * 1000))
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {BORDER};
                height: 4px;
                background: {BG_MAIN};
                margin: 2px 0;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #6C5CE7;
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
        """)
        self.slider.valueChanged.connect(self._on_slider_move)

        max_lbl = QLabel(str(v_max))
        max_lbl.setFont(QFont("Segoe UI", 8))
        max_lbl.setStyleSheet(f"color:{TEXT_HINT}; border:none;")

        btm_lay.addWidget(min_lbl)
        btm_lay.addWidget(self.slider)
        btm_lay.addWidget(max_lbl)
        main_lay.addLayout(btm_lay)

        # 动画定时器
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        self._anim_dir = 1

    def _on_slider_move(self, v):
        val = self.v_min + (v / 1000.0) * (self.v_max - self.v_min)
        self.val_edit.setText(f"{val:.2f}")
        self.on_change(self.name, val)

    def _on_text_edited(self):
        try:
            val = float(self.val_edit.text())
            self.set_value(val)
            self.val_edit.clearFocus()
        except:
            # 恢复旧值
            v_slider = self.slider.value()
            curr_v = self.v_min + (v_slider / 1000.0) * (self.v_max - self.v_min)
            self.val_edit.setText(f"{curr_v:.2f}")

    def set_value(self, val):
        val = max(self.v_min, min(self.v_max, val))
        v_slider = int((val - self.v_min) / (self.v_max - self.v_min) * 1000)
        self.slider.blockSignals(True)
        self.slider.setValue(v_slider)
        self.slider.blockSignals(False)
        self.val_edit.setText(f"{val:.2f}")
        self.on_change(self.name, val)

    def _toggle_animation(self):
        if self.play_btn.isChecked():
            self.play_btn.setText("⏸")
            self.anim_timer.start(50) # 20 FPS
        else:
            self.play_btn.setText("▶")
            self.anim_timer.stop()

    def _animate_step(self):
        curr_v = self.slider.value()
        new_v = curr_v + self._anim_dir * 5 # 每次移动 0.5%
        if new_v >= 1000:
            new_v = 1000
            self._anim_dir = -1
        elif new_v <= 0:
            new_v = 0
            self._anim_dir = 1
        self.slider.setValue(new_v)

# ══════════════════════════════════════════════════════
#  方程列表条目
# ══════════════════════════════════════════════════════
class EqItem(QWidget):
    def __init__(self, eq_text, color, on_delete, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(
            f"background:{BG_PANEL}; border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 12, 0)
        lay.setSpacing(12)

        self._badge = QLabel()
        self._badge.setFixedSize(16, 16)
        self._badge.setStyleSheet(f"background:{color}; border-radius:4px; border:none;")

        # 使用 SymPy 的漂亮格式，在列表中直接显示 x² + y² = 25 这样的形式
        lbl = QLabel(sympy_to_label(eq_text))
        lbl.setFont(QFont("Consolas", 13))
        lbl.setStyleSheet(
            f"color:{TEXT_MAIN}; border:none; background:transparent;")
        lbl.setWordWrap(True)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(28, 28)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background:transparent; color:{TEXT_SUB};
                border:none; font-size:14px; border-radius:4px;
            }}
            QPushButton:hover {{ color:white; background:#E74C3C; }}
        """)
        del_btn.clicked.connect(on_delete)

        lay.addWidget(self._badge)
        lay.addWidget(lbl, 1)
        lay.addWidget(del_btn)

    def set_color(self, color):
        self._badge.setStyleSheet(f"background:{color}; border-radius:3px; border:none;")


# ══════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════
class MathPlotterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数学方程可视化 Pro")
        self.resize(1400, 860)
        self._equations = []
        self._eq_widgets = []
        self._params = {}           # dict: {name: value}
        self._param_widgets = {}    # dict: {name: ParamItem}
        self._color_idx = 0
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background:{BG_MAIN}; color:{TEXT_MAIN};
                font-family:'Segoe UI',Arial,sans-serif;
            }}
            QSplitter::handle {{ background:{BORDER}; width:1px; }}
            QLineEdit {{
                background:{BG_INPUT}; border:1.5px solid {BORDER};
                border-radius:6px; color:{TEXT_MAIN};
                font-size:14px; padding:7px 10px;
            }}
            QLineEdit:focus {{ border-color:{ACCENT}; }}
            QScrollBar:vertical {{
                background:{BG_MAIN}; width:6px; border-radius:3px;
            }}
            QScrollBar::handle:vertical {{
                background:{BORDER}; border-radius:3px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height:0; }}
        """)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        self.setCentralWidget(splitter)

        # ── 左侧面板 ────────────────────────────
        left = QWidget()
        left.setFixedWidth(380)
        left.setStyleSheet(
            f"background:{BG_PANEL}; border-right:1px solid {BORDER};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        # Logo 栏
        logo_bar = QWidget()
        logo_bar.setFixedHeight(64)
        logo_bar.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #5A4ED4,stop:1 #8B5CF6);"
            "border-bottom:1px solid #4C3BBF;")
        lb = QHBoxLayout(logo_bar)
        lb.setContentsMargins(20, 0, 20, 0)
        logo_lbl = QLabel("方程可视化")
        logo_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        logo_lbl.setStyleSheet("color:white; border:none;")
        lb.addWidget(logo_lbl)
        lb.addStretch()
        ll.addWidget(logo_bar)

        # 输入区
        inp_area = QWidget()
        inp_area.setStyleSheet(
            f"background:{BG_PANEL}; border-bottom:1px solid {BORDER};")
        ia = QVBoxLayout(inp_area)
        ia.setContentsMargins(16, 16, 16, 16)
        ia.setSpacing(12)

        # 输入框和删除按钮的容器
        in_top = QHBoxLayout()
        self.eq_input = QLineEdit()
        self.eq_input.setPlaceholderText("输入方程，如 x**2+y**2+z**2=9 …")
        self.eq_input.setFont(QFont("Consolas", 16))
        self.eq_input.returnPressed.connect(self._add_eq)
        self.eq_input.installEventFilter(self)

        self.backspace_btn = QPushButton("⌫")
        self.backspace_btn.setFixedSize(44, 44)
        self.backspace_btn.setCursor(Qt.PointingHandCursor)
        self.backspace_btn.setStyleSheet(f"""
            QPushButton {{
                background:{BG_INPUT}; border:1.5px solid {BORDER};
                border-radius:6px; color:{BTN_CLEAR};
                font-size:20px; font-weight:bold;
            }}
            QPushButton:hover {{ background:#FEE2E2; border-color:{BTN_CLEAR}; }}
        """)
        self.backspace_btn.clicked.connect(self._handle_backspace)

        in_top.addWidget(self.eq_input, 1)
        in_top.addWidget(self.backspace_btn)
        ia.addLayout(in_top)
        br = QHBoxLayout()
        br.setSpacing(10)
        for txt, col, slot in [
            ("＋ 添加", BTN_ADD, self._add_eq),
            ("▶ 绘制", BTN_PLOT, self._plot_all),
            ("⟳ 复位", "#8E44AD", self._reset_view),
            ("✕ 清除", BTN_CLEAR, self._clear_all),
        ]:
            b = self._mk_btn(txt, col)
            b.clicked.connect(lambda _, s=slot: s())
            br.addWidget(b)
        ia.addLayout(br)
        ll.addWidget(inp_area)

        # 符号键盘
        # 数学符号键盘区域（初始隐藏，仅在聚焦输入框时显示）
        kb = QWidget()
        kb.setStyleSheet(
            f"background:{BG_PANEL}; border-bottom:1px solid {BORDER};")
        kbl = QVBoxLayout(kb)
        kbl.setContentsMargins(16, 12, 16, 12)
        kbl.setSpacing(10)
        kbl_title = QLabel("常用符号")
        # 放大标题字号，增强可读性
        kbl_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        kbl_title.setStyleSheet(f"color:{TEXT_SUB}; border:none;")
        kbl.addWidget(kbl_title)

        # GeoGebra 风格的多页符号键盘（数字 / 函数 / 其它）——放大按钮尺寸以便看清
        tabs = QTabWidget()
        tabs.setTabBarAutoHide(False)
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 0px;
            }}
            QTabBar::tab {{
                background: {BG_INPUT};
                border: 1px solid {BORDER};
                border-bottom: none;
                padding: 4px 10px;
                font-size: 11px;
                min-width: 40px;
            }}
            QTabBar::tab:selected {{
                background: #EEF2FF;
                color: {ACCENT};
                border-color: {ACCENT};
            }}
        """)

        def make_button(txt, display=None, cursor_offset=0):
            """创建键盘按钮。

            txt: 实际插入到输入框的字符串
            display: 按钮上显示的文本（为 None 时使用 txt）
            cursor_offset: 插入后光标相对偏移（负数可把光标移到括号内）
            """
            label = display if display is not None else txt
            b = QPushButton(label)
            # 放大按钮尺寸与字体，接近 GeoGebra 面板视觉
            b.setFixedSize(72, 48)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{BG_INPUT}; border:1px solid {BORDER};
                    border-radius:8px; color:{ACCENT}; 
                    font-family: Consolas; font-size: 18px; font-weight: bold;
                }}
                QPushButton:hover {{
                    background:#EEF2FF; border-color:{ACCENT}; color:#3B52C4;
                    margin-top: -1px;
                }}
                QPushButton:pressed {{ background:#DDE4FF; margin-top: 1px; }}
            """)
            b.clicked.connect(lambda _, t=txt, off=cursor_offset: self._insert_token(t, off))
            return b

        # --- 基本页：数字 + 变量 + 四则运算 ---
        basic_page = QWidget()
        basic_grid = QGridLayout(basic_page)
        basic_grid.setSpacing(8)
        basic_keys = [
            # (text, display, row, col, cursor_offset)
            ('x', None, 0, 0, 0), ('y', None, 0, 1, 0), ('z', None, 0, 2, 0), ('pi', 'π', 0, 3, 0),
            ('x^2', 'x²', 1, 0, 0), ('y^2', 'y²', 1, 1, 0), ('^2', 'a²', 1, 2, 0),
            ('^(', 'a^□', 1, 3, -1),  # 插入 ^( ) 并把光标移到括号内
            ('7', None, 2, 0, 0), ('8', None, 2, 1, 0), ('9', None, 2, 2, 0), ('/', '÷', 2, 3, 0),
            ('4', None, 3, 0, 0), ('5', None, 3, 1, 0), ('6', None, 3, 2, 0), ('*', '×', 3, 3, 0),
            ('1', None, 4, 0, 0), ('2', None, 4, 1, 0), ('3', None, 4, 2, 0), ('-', None, 4, 3, 0),
            ('0', None, 5, 0, 0), ('.', None, 5, 1, 0), ('=', None, 5, 2, 0), ('+', None, 5, 3, 0),
        ]
        for txt, disp, r, c, off in basic_keys:
            # 特殊处理：x^2 / y^2 直接插入为 SymPy 友好的形式
            if txt == 'x^2':
                real_txt = 'x^2'
            elif txt == 'y^2':
                real_txt = 'y^2'
            elif txt == '^(':
                real_txt = '^()'
            else:
                real_txt = txt
            basic_grid.addWidget(make_button(real_txt, disp, off), r, c)
        tabs.addTab(basic_page, "123")

        # --- 函数页：三角、指数、对数、根号等 ---
        func_page = QWidget()
        func_grid = QGridLayout(func_page)
        func_grid.setSpacing(8)
        func_keys = [
            ('sin()', 'sin', 0, 0, -1),
            ('cos()', 'cos', 0, 1, -1),
            ('tan()', 'tan', 0, 2, -1),
            ('sqrt()', '√', 0, 3, -1),

            ('exp()', 'exp', 1, 0, -1),
            ('log()', 'log', 1, 1, -1),
            ('abs()', '|a|', 1, 2, -1),
            ('E', 'e', 1, 3, 0),

            ('(', None, 2, 0, 0),
            (')', None, 2, 1, 0),
            ('**', '^', 2, 2, 0),
            (',', None, 2, 3, 0),
        ]
        for txt, disp, r, c, off in func_keys:
            func_grid.addWidget(make_button(txt, disp, off), r, c)
        tabs.addTab(func_page, "f(x)")

        # --- 其它符号页：关系运算等（均为 SymPy 支持的 ASCII） ---
        other_page = QWidget()
        other_grid = QGridLayout(other_page)
        other_grid.setSpacing(8)
        other_keys = [
            ('<', None, 0, 0, 0), ('>', None, 0, 1, 0),
            ('<=', '≤', 0, 2, 0), ('>=', '≥', 0, 3, 0),
            ('!=', '≠', 1, 0, 0), ('and', 'and', 1, 1, 0),
            ('or', 'or', 1, 2, 0), ('not', 'not', 1, 3, 0),
        ]
        for txt, disp, r, c, off in other_keys:
            other_grid.addWidget(make_button(txt, disp, off), r, c)
        tabs.addTab(other_page, "#")

        kbl.addWidget(tabs)
        ll.addWidget(kb)
        # 记录键盘组件，初始隐藏：只有在输入框获得焦点时才显示
        self._kb_widget = kb
        self._kb_widget.setVisible(False)

        # 代数区
        alg_bar = QWidget()
        alg_bar.setFixedHeight(40)
        alg_bar.setStyleSheet(
            f"background:#F6F4FF; border-bottom:1px solid {BORDER};")
        ab = QHBoxLayout(alg_bar)
        ab.setContentsMargins(16, 0, 16, 0)
        ab_lbl = QLabel("代 数 区")
        ab_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        ab_lbl.setStyleSheet(f"color:{ACCENT}; border:none;")
        self._alg_count_lbl = QLabel("0 个方程")
        self._alg_count_lbl.setFont(QFont("Segoe UI", 9))
        self._alg_count_lbl.setStyleSheet(
            f"color:{TEXT_HINT}; border:none; background:transparent;")
        ab.addWidget(ab_lbl)
        ab.addStretch()
        ab.addWidget(self._alg_count_lbl)
        ll.addWidget(alg_bar)

        self.eq_scroll = QScrollArea()
        self.eq_scroll.setWidgetResizable(True)
        self.eq_scroll.setStyleSheet(f"background:{BG_PANEL}; border:none;")
        self.eq_list_w = QWidget()
        self.eq_list_w.setStyleSheet(f"background:{BG_PANEL};")
        self.eq_list_lay = QVBoxLayout(self.eq_list_w)
        self.eq_list_lay.setContentsMargins(16, 16, 16, 16) # 增加内边距
        self.eq_list_lay.setSpacing(12) # 增加条目间距
        self.eq_list_lay.addStretch()
        self.eq_scroll.setWidget(self.eq_list_w)
        ll.addWidget(self.eq_scroll, 1)

        # ── 参数区 ────────────────────────────
        param_bar = QWidget()
        param_bar.setFixedHeight(40)
        param_bar.setStyleSheet(f"background:#F6F4FF; border-bottom:1px solid {BORDER}; border-top:1px solid {BORDER};")
        pb = QHBoxLayout(param_bar)
        pb.setContentsMargins(16, 0, 16, 0)
        pb_lbl = QLabel("参 数")
        pb_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        pb_lbl.setStyleSheet(f"color:{ACCENT}; border:none;")
        pb.addWidget(pb_lbl)
        pb.addStretch()
        ll.addWidget(param_bar)

        # 参数输入与列表
        param_area = QWidget()
        param_area.setStyleSheet(f"background:{BG_PANEL};")
        pal = QVBoxLayout(param_area)
        pal.setContentsMargins(16, 12, 16, 12)
        pal.setSpacing(8)

        # 参数列表滚动区
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setStyleSheet("border:none; background:transparent;")
        self.param_list_w = QWidget()
        self.param_list_w.setStyleSheet("background:transparent;")
        self.param_list_lay = QVBoxLayout(self.param_list_w)
        self.param_list_lay.setContentsMargins(0, 0, 0, 0)
        self.param_list_lay.setSpacing(2)
        self.param_list_lay.addStretch()
        self.param_scroll.setWidget(self.param_list_w)
        pal.addWidget(self.param_scroll, 1)

        ll.addWidget(param_area, 1)

        splitter.addWidget(left)

        # ── 右侧画布 ────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{BG_MAIN};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        self.canvas3d = Canvas3D()
        rl.addWidget(self.canvas3d)

        splitter.addWidget(right)
        splitter.setSizes([380, 1020])

    def _insert_token(self, text, cursor_offset=0):
        """
        在当前输入框中插入一段文本，并根据 cursor_offset 调整光标位置。
        例如：text='sqrt()' 且 cursor_offset=-1 时，插入后光标会落在括号内部。
        """
        le = self.eq_input
        current = le.text()
        pos = le.cursorPosition()
        new_text = current[:pos] + text + current[pos:]
        le.setText(new_text)
        # 光标移动到插入文本后的合适位置
        le.setCursorPosition(pos + len(text) + cursor_offset)
        le.setFocus()

    def _handle_backspace(self):
        """处理退格键逻辑，删除光标前的一个字符或选中部分。"""
        le = self.eq_input
        if le.hasSelectedText():
            le.del_()
        else:
            pos = le.cursorPosition()
            if pos > 0:
                text = le.text()
                new_text = text[:pos-1] + text[pos:]
                le.setText(new_text)
                le.setCursorPosition(pos - 1)
        le.setFocus()

    def _show_math_keyboard(self, visible: bool):
        """控制数学符号键盘的显隐。"""
        if hasattr(self, "_kb_widget") and self._kb_widget is not None:
            self._kb_widget.setVisible(visible)

    def eventFilter(self, obj, event):
        # 当方程输入框获得焦点或被点击时，显示数学键盘
        if obj is self.eq_input:
            if event.type() in (QEvent.FocusIn, QEvent.MouseButtonPress):
                self._show_math_keyboard(True)
        return super().eventFilter(obj, event)

    def _mk_btn(self, text, color):
        c = QColor(color)
        hov = QColor(max(c.red() - 30, 0), max(c.green() - 30, 0), max(c.blue() - 30, 0)).name()
        prs = QColor(max(c.red() - 55, 0), max(c.green() - 55, 0), max(c.blue() - 55, 0)).name()
        b = QPushButton(text)
        b.setFont(QFont("Segoe UI", 9, QFont.Bold)) # 调小字体
        b.setFixedHeight(34) # 调小高度
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{color}; border:none; border-radius:6px;
                color:white; padding:0 8px;
                text-align: center;
            }}
            QPushButton:hover {{ background:{hov}; margin-top: 1px; }}
            QPushButton:pressed {{ background:{prs}; margin-top: 2px; }}
        """)
        return b

    # ── 参数管理 ────────────────────────────────
    def _add_param(self, name=None, val=1.0, trigger_plot=True):
        """
        添加参数。如果传入 name，则直接使用；否则（如果还保留了输入框的话）从输入框读取。
        如果输入框中只有名字 (如 "a")，则默认值为 1.0。
        """
        if name is None:
            # 现在已经移除了手动输入框，此分支通常不再触发，但保留逻辑以防万一
            return

        try:
            if name in ('x', 'y', 'z'):
                return

            if name in self._params:
                # 如果已经存在，就不重复添加
                return

            # 创建参数条目
            # 默认范围 -10 到 10，除非初值超标
            v_min = min(-10.0, val - 5.0)
            v_max = max(10.0, val + 5.0)

            self._params[name] = val
            item = ParamItem(name, val, v_min, v_max, self._on_param_change, self._remove_param)
            self._param_widgets[name] = item
            self.param_list_lay.insertWidget(self.param_list_lay.count() - 1, item)

            # 如果已有方程使用该参数，则重新绘制
            if trigger_plot:
                self._plot_all(auto_active=[], force_update=True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法添加参数: {e}")

    def _remove_param(self, name):
        if name in self._params:
            del self._params[name]
            widget = self._param_widgets.pop(name)
            widget.setParent(None)
            widget.deleteLater()
            # 重新绘制，可能需要显示错误提示（如果方程引用了已删除的参数）
            self._plot_all(auto_active=[])

    def _on_param_change(self, name, val):
        self._params[name] = val
        # 实时联动：当参数变化时，强制触发重绘以更新现有 Actor 的几何数据
        self._plot_all(auto_active=[], force_update=True)

    # ── 方程管理 ────────────────────────────────
    def _add_eq(self):
        text = self.eq_input.text().strip()
        if not text:
            return

        # 获取当前所有已使用的颜色
        existing_colors = [item[2] for item in self._equations if item]
        color = generate_random_color(existing_colors)

        # 生成唯一 ID
        eq_id = str(uuid.uuid4())

        # 存入列表 (保持索引对应机制)
        idx = len(self._equations)
        self._equations.append((eq_id, text, color))

        # 创建 UI 条目，传递 ID 用于删除
        item = EqItem(text, color, lambda _=False, i=eq_id: self._remove_eq(i))
        self._eq_widgets.append(item)
        self.eq_list_lay.insertWidget(self.eq_list_lay.count() - 1, item)
        self.eq_input.clear()
        self._update_count()
        # 【UX】用户要求点击添加不立即绘制，需点击绘制按钮才开始
        # self._plot_all()

    def _remove_eq(self, eq_id):
        # 根据 ID 查找索引
        idx = -1
        for i, val in enumerate(self._equations):
            if val and val[0] == eq_id:
                idx = i
                break

        if idx != -1:
            self._equations[idx] = None
            if idx < len(self._eq_widgets) and self._eq_widgets[idx]:
                self._eq_widgets[idx].setParent(None)
                self._eq_widgets[idx] = None

        self._update_count()
        # 删除时立即重绘，确保所见即所得
        self._plot_all(auto_active=[])

    def _clear_all(self):
        self._equations.clear()
        self._eq_widgets.clear()
        self._color_idx = 0
        while self.eq_list_lay.count() > 1:
            item = self.eq_list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.canvas3d.clear_canvas()
        self._update_count()

    def _reset_view(self):
        """调用3D画布的重置视角功能"""
        if hasattr(self, 'canvas3d'):
            self.canvas3d.reset_view()

    def _update_count(self):
        n = sum(1 for e in self._equations if e)
        self._alg_count_lbl.setText(f"{n} 个方程")

    def _plot_all(self, auto_active=None, force_update=False):
        target_items = []
        # 扫描 self._equations 获取当前有效方程
        for val in self._equations:
            if val:
                # val is (eq_id, text, color)
                target_items.append(val)

        # 只有当列表为空且 auto_active 为 None (即用户点击"绘制") 时，才检查输入框
        if not target_items and auto_active is None:
            text = self.eq_input.text().strip()
            if text:
                # 输入框预览模式：生成临时 ID
                temp_id = str(uuid.uuid4())
                existing_colors = [] # 预览模式暂不考虑冲突
                color = generate_random_color(existing_colors)
                target_items.append((temp_id, text, color))
            else:
                QMessageBox.warning(self, "警告", "请先添加至少一个方程！")
                return

        parsed = []

        for eq_id, eq_text, color in target_items:
            try:
                p = self._parse(eq_text)
                if p:
                    # ──【新增】参数代入逻辑 ──
                    # 在解析后、生成网格前，将表达式中的参数替换为当前数值
                    if self._params:
                        # 构造替换映射表
                        subs_dict = {sp.Symbol(k): v for k, v in self._params.items()}
                        p['expr'] = p['expr'].subs(subs_dict)

                    p['color'] = color
                    parsed.append((eq_id, p))
            except Exception as e:
                QMessageBox.critical(self, "解析错误", f"{eq_text}\n{e}")
                return

        if not parsed:
            self.canvas3d.clear_canvas()
            return

        self.canvas3d.draw_equations(parsed, force_update=force_update)

        # 【Fix】更新颜色对应关系
        # 这里的 parsed 包含 (eq_id, p)
        for eq_id, p in parsed:
            # 从 Canvas3D 获取最终使用的颜色
            real_color = self.canvas3d._eq_colors.get(eq_id)

            # 查找对应的方程条目并更新
            if real_color:
                found_idx = -1
                for i, val in enumerate(self._equations):
                    if val and val[0] == eq_id:
                        found_idx = i
                        if val[2] != real_color:
                            # 更新数据
                            self._equations[i] = (eq_id, val[1], real_color)
                        break

                # 更新 UI
                if found_idx != -1 and found_idx < len(self._eq_widgets):
                    w = self._eq_widgets[found_idx]
                    if w:
                        w.set_color(real_color)

    def _parse(self, eq_text):
        try:
            # 1. Protect common functions: "sqrt(" -> "sqrt (" to avoid implicit multiplication
            # This prevents "sqrt(x)" becoming "sqrt*(x)" which is invalid
            funcs = ["sin", "cos", "tan", "sqrt", "exp", "log", "ln", "abs",
                     "asin", "acos", "atan", "sinh", "cosh", "tanh", "floor", "ceil"]
            for f in funcs:
                eq_text = eq_text.replace(f"{f}(", f"{f} (")

            # ──【修复】更完善的隐式乘法处理 (GeoGebra 风格) ──
            # 1. 数字后接字母: 2x -> 2*x
            eq_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq_text)

            # 2. 字母后接数字 (如果不是幂运算): x2 -> x*2 (通常建议用 x^2，这里作为防御)
            # 但不要处理已经在处理幂运算的场景
            # eq_text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', eq_text) # 暂时不开启，避免破坏 ^2

            # 3. 处理 ax, az, xyz 等隐式乘法
            # 关键：识别 x, y, z 与前后字母的关系
            # 如果字母不是函数名的一部分，就在它们之间插入 *
            # 我们先处理坐标变量 x, y, z 与其它字母的相邻情况
            eq_text = re.sub(r'([a-zA-Z])([xyz])', r'\1*\2', eq_text) # ax -> a*x, bx -> b*x
            eq_text = re.sub(r'([xyz])([a-zA-Z])', r'\1*\2', eq_text) # xa -> x*a, zb -> z*b

            # 4. 处理坐标变量之间的隐式乘法: xy -> x*y, yz -> y*z
            # 注意：上一步已经覆盖了 [a-zA-Z][xyz]，所以 xy 会被替换为 x*y。

            # 5. 括号相关的隐式乘法
            eq_text = re.sub(r'(?<=[a-zA-Z0-9])(?=\()', '*', eq_text) # a(x) -> a*(x)
            eq_text = re.sub(r'(?<=\))(?=[a-zA-Z0-9])', '*', eq_text) # (x)a -> (x)*a

            t = eq_text.replace('^', '**')

            # 【Fix】支持不等式解析 (>=, <=, >, <, !=, ==, =)
            # 将关系式转换为 f(x,y,z) = LHS - RHS，以便 Marching Cubes 绘制边界
            ops = ["==", "!=", ">=", "<=", ">", "<", "="]
            found_op = None
            for op in ops:
                if op in t:
                    found_op = op
                    break

            if found_op:
                lhs, rhs = t.split(found_op, 1)
                # print(f"[DEBUG] Parsing relation: {lhs} {found_op} {rhs}")
                expr = sp.sympify(f"({lhs})-({rhs})")
            else:
                expr = sp.sympify(t)

            # ──【新增】参数检测逻辑 ──
            all_syms = expr.free_symbols
            coords = set([sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')])
            params_needed = all_syms - coords

            for p_sym in params_needed:
                p_name = str(p_sym)
                if p_name not in self._params:
                    # ──【优化】自动创建未定义的参数 ──
                    # 模仿 GeoGebra：发现未知变量时自动创建滑动条，默认值为 1.0
                    # 设置 trigger_plot=False 避免在解析循环中产生递归重绘
                    self._add_param(name=p_name, val=1.0, trigger_plot=False)

            # 过滤出真正的坐标变量，用于 Marching Cubes / Parametric Mesh
            syms = sorted(list(all_syms & coords), key=str)
            # 如果方程没有 x, y, z（如 a=1），这虽然不是几何方程但 SymPy 会认为它是常数，这里保持原有逻辑
            if not syms and not params_needed:
                return None

            return {
                'expr': expr,
                'syms': syms,
                'sym_names': [str(s) for s in syms],
                'dims': len(syms),
                'raw': t,
            }
        except Exception as e:
            raise e


# ══════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    w = MathPlotterApp()
    w.show()
    sys.exit(app.exec_())