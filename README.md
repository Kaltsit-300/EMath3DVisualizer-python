# EMath3DVisualizer

一个用于三重积分几何分析的三维可视化工具。

本项目为学习型项目，旨在帮助理解空间几何结构及三重积分区域。

---

## 功能

- 绘制三维隐式曲面（球面、柱面、平面等）
- 同时显示多个几何体
- 计算并显示几何体交线
- 支持围成空间区域的可视化分析
- 左键用于旋转图像，右键可以拖动图像

> 注意：本项目只提供三维立体几何可视化，不包含积分数值计算，平面绘图等功能。

---

## 技术栈

- Python 3.x
- PyQt5
- PyVista
- VTK
- NumPy
- SymPy
- scikit-image

---

## 核心原理


### 整体流程

用户输入隐式方程 → SymPy 解析与符号替换 → 生成三角网格（优先解析几何，后备 Marching Cubes）→ PyVista/VTK 渲染 → 实时交线计算与显示。

### 方程解析与预处理

- **符号解析**：使用 `sympy.sympify` 将字符串转换为表达式，自动识别 `x,y,z` 及自定义参数（如 `a, b, c`）。
- **参数替换**：参数滑块改变时，将表达式中的符号替换为当前数值（`expr.subs()`）。
- **隐式化**：方程中的等号（`=`、`==`、`<=` 等）统一转换为 `F(x,y,z)=0` 形式（左减右）。
- **文本美化**：支持 `^` 转 `**`，自动补全隐式乘法（如 `2x` → `2*x`），公式显示时转换为 Unicode 上标（`x^2` → `x²`）。

### 网格生成策略（三种方法，按优先级）

#### ① 解析几何精确生成（性能最优，无锯齿）

检测方程是否为标准二次曲面，直接调用 PyVista 的参数化网格：

- **球体**：`Ax²+Ay²+Az²+Dx+Ey+Fz+G=0` → `pv.Sphere`
- **圆柱体**：`x²+y²=R²`（沿 Z 轴） → `pv.Cylinder`
- **平面**：`Ax+By+Cz+D=0` → `pv.Plane`
- **圆锥/抛物面**：`z² = x²+y²` 或 `z = x²+y²` → 参数化生成网格

#### ② 显式曲面快速生成（`z=f(x,y)` 等）

当方程可解出某一变量（如 `z = f(x,y)`）且为单值函数时：

- 在 `xy` 平面构建结构化网格，计算 `z` 值，直接生成 `pv.StructuredGrid`
- 避免 Marching Cubes 在定义域边界产生竖直侧壁伪影
- 分辨率根据视图距离动态调整（`res_uv = 64~180`）

#### ③ Marching Cubes 后备方案（通用隐式曲面）

- 使用 `skimage.measure.marching_cubes` 从体素网格提取零等值面
- 体素范围：根据相机视野动态裁剪（`limit = view_radius * 0.6`）
- 分辨率动态调整：滚动中 `RES=22`（低精度），停止后 `RES=48~100`（高精度）
- 后处理：Taubin 平滑（`smooth_taubin`）+ 法线重算，消除阶梯感

### 交线计算（解析法优先，避免 VTK 崩溃）

- **平面-平面**：解析求解直线方程，裁剪到视图盒，生成 `pv.Spline`
- **球-平面**：解析计算圆方程，参数化生成圆环线
- **圆柱-平面**：离散化参数 `θ`，代入平面方程解 `h`，生成椭圆/双曲线
- **球-圆柱**：高密度采样 `θ`，解析求解二次方程得到 `h`，合并分支并做 Laplacian 平滑（解决 Viviani 曲线奇点）
- **其他组合**：回退到 VTK 的 `vtkIntersectionPolyDataFilter`，并附加边拓扑链提取 + 折线平滑

所有交线强制设为纯黑色（`ambient=1, diffuse=0, specular=0, lighting=False`），始终可见。

### 性能优化与 LOD（Level of Detail）

- **视图半径检测**：定时器获取相机与焦点的距离，动态计算 `view_radius`
- **滚动缩放时**：设置 `_is_lod=True`，后续网格生成使用最低分辨率（`RES=22`），**仅更新相机，不重建网格** → 零卡顿
- **停止滚动 600ms 后**：`_is_lod=False`，触发全精度重建（高分辨率网格 + 坐标轴 + 交线）
- **刻度动态步长**：根据相机距离计算 `nice step`，只重建必要的刻度线
- **网格范围裁剪**：`grid_range = min(60, view_radius*1.3)`，避免绘制视野外物体

### 3D 交互与渲染

- **渲染引擎**：`pyvistaqt.QtInteractor`（基于 VTK），嵌入 PyQt5
- **交互风格**：`terrain_style`（禁用 Z 轴旋转），支持右键拖动平移
- **滚轮缩放**：重写 `wheelEvent`，实现以鼠标位置为中心的缩放（射线-平面交点）
- **抗锯齿**：`multi_samples=8` 开启 MSAA，线条平滑
- **坐标轴与网格**：动态长度、箭头大小，主次网格线（`major_step` / `minor_step`）
- **浮动图例**：`QFrame` 子窗口，手动绘制半透明背景、颜色块和方程文本，始终覆盖在 VTK 之上

### UI 与用户体验

- **符号键盘**：`QTabWidget` 包含数字/函数/运算符，点击插入光标位置
- **参数动画**：`QTimer` 驱动滑块自动往复运动（20 FPS）
- **实时联动**：参数滑动时立即重绘所有方程（`force_update=True`）
- **公式渲染**：自绘 `√` 根号横线，支持上标、希腊字母（`π`）

---

## 安装

建议使用虚拟环境：

```bash
python -m venv venv
venv\Scripts\activate
```

安装依赖：

```bash
pip install pyqt5 pyvista pyvistaqt vtk numpy sympy scikit-image
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

---

## 运行

```bash
python math_3d_visualizer.py
```

---

## 项目结构

```
math_3d_visualizer.py
```

---
## 效果展示
<img width="2559" height="1494" alt="image" src="https://github.com/user-attachments/assets/5a474147-fb67-4c35-8641-0193220ec017" />
<img width="2559" height="1494" alt="image" src="https://github.com/user-attachments/assets/e52f8df5-b64b-4cbb-a213-b0208bbc77dd" />



---
## 使用场景

- 用于辅助三重积分解题，用于分析空间中面与面和空间中体与面之间的位置关系，或分析面与面所围成的立体图形在空间里的具体位置


## 说明

本项目为学习用途开发，用于加深对三维空间曲面结构的理解。
