# -*- coding: UTF-8 -*-
# filename: Find_Jacobian date: 2018/3/1 21:10  
# author: FD 
import sympy
from sympy.abc import x, y, r, theta
from sympy import symbols, Matrix

sympy.init_printing(use_latex="mathjax", fontsize='16pt')
theta_v, theta_a = symbols("theta_v,theta_a")
m = 0.03
x_tag0 = (r + m) * sympy.cos(theta) + x
y_tag0 = (r + m) * sympy.sin(theta) + y
angle_delta = sympy.acos((2 * r ** 2 - m * r + m ** 2 / 4) / (2 * r * sympy.sqrt(r ** 2 + m ** 2 - m * r)))
d = sympy.sqrt(r ** 2 + m ** 2 - m * r)
x_tag1 = d * sympy.cos(theta - angle_delta) + x
y_tag1 = d * sympy.sin(theta - angle_delta) + y
x_tag2 = d * sympy.cos(theta + angle_delta) + x
y_tag2 = d * sympy.cos(theta + angle_delta) + y
distance_tag0 = sympy.sqrt(x_tag0 ** 2 + y_tag0 ** 2)
distance_tag1 = sympy.sqrt(x_tag1 ** 2 + y_tag1 ** 2)
distance_tag2 = sympy.sqrt(x_tag2 ** 2 + y_tag2 ** 2)
fxu = Matrix([[distance_tag0], [distance_tag1], [distance_tag2]])
F = fxu.jacobian(Matrix([x, y, r, theta, theta_v, theta_a]))
print F