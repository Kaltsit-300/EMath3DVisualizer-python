# EMath3DVisualizer

一个用于三重积分几何分析的三维可视化工具。

本项目为学习型项目，旨在帮助理解空间几何结构及三重积分区域。

---

## 功能

- 绘制三维隐式曲面（球面、柱面、平面等）
- 同时显示多个几何体
- 计算并显示几何体交线
- 支持围成空间区域的可视化分析

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

- 使用 Marching Cubes 算法将隐式方程  
  `f(x, y, z) = 0`  
  转换为三角网格

- 使用 SymPy 解析数学表达式

- 使用 PyVista + VTK 进行三维渲染

- 使用分辨率控制优化渲染性能

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

## 使用场景

- 三重积分做题辅助，用于分析空间中面与面和空间中体与面之间的位置关系，或分析面与面所围成的立体图形在空间里的具体位置


## 说明

本项目为学习用途开发，用于加深对三维空间曲面结构的理解。
