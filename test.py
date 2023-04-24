#%%
import matplotlib.pyplot as plt
import math


def delta_on_curve(omega, v, span):
    # Calculate the curvature (κ)
    curvature = omega / v

    # Calculate the radius of the curvature (R)
    radius = 1 / curvature

    # Calculate the angular displacement (θ) based on the curve_span
    theta = span * curvature

    # Calculate the change in x and y coordinates
    delta_x = radius * (1 - math.cos(theta))
    delta_y = radius * math.sin(theta)

    return delta_x, delta_y


# get list of points for a given number of steps
def get_points(x0, y0, omega, v, span, steps):
    points = []
    delta_x, delta_y = delta_on_curve(omega, v, span)
    for i in range(steps):
        x0 += delta_x
        y0 += delta_y
        points.append((x0, y0))

    return points


def round_with_scale(number, scale):
    return round(number / scale)


# rasterize the points to a grid by rounding into the nearest square
def rasterize_points(gridmap, points, grid_scale):
    x_len = len(gridmap[0])
    y_len = len(gridmap)
    for x, y in points:
        x_int, y_int = round_with_scale(x, grid_scale), round_with_scale(y, grid_scale)
        gridmap[y_int - 1, x_int - 1] = 1

    return gridmap


import numpy as np

# test above functions
omega = 0.1
v = 0.1
span = 2
grid_scale = 2
steps = 100
x0 = 0
y0 = 0
gridmap = np.zeros((100, 100))

points = get_points(x0, y0, omega, v, span, steps)
grid = rasterize_points(gridmap, points, grid_scale)

plt.imshow(grid, cmap="gray")

# %%
