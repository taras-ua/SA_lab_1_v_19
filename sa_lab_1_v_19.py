import matplotlib.pyplot as plotter
from matplotlib.patches import Polygon
import numpy as np


''' Target functions '''


def f1(x):
    #return 5 * np.log10(x + 9)
    return 1 - 1.6 * x + 7 * x ** 2


def f2(x):
    #return -12 + 4 * x
    return 6 + 8 * x - 3 * x ** 2


''' Target constants '''

f1_star = 45.0#10.0
f2_star = 5.0#2.0
x_lbound = -2.0#0.001
x_rbound = 2.0#5.0
step = 0.001
eps = 0.0001
all_interval = np.linspace(x_lbound, x_rbound, (x_rbound - x_lbound) / step)

''' Minmax/maxmin functions '''


def min_f(x):
    return min(f1(x) / f1_star, f2(x) / f2_star)


def max_f(x):
    return max(f1(x) / f1_star, f2(x) / f2_star)


def maxmin(space):
    f0 = min_f(space[0])
    x0 = space[0]
    for i in range(len(space) - 1):
        if min_f(space[i]) >= f0:
            f0 = min_f(space[i])
            x0 = space[i]
    return f0, x0


def minmax(space):
    f0 = max_f(space[0])
    x0 = space[0]
    for i in range(len(space) - 1):
        print("x=%.2f"%space[i],"max=%.2f"%max_f(space[i]))
        if max_f(space[i]) <= f0:
            f0 = max_f(space[i])
            x0 = space[i]
    print("MINMAX = %.2f"%x0)
    return f0, x0

def minmax_maxmin_region():
    space, a, b = pareto_region()
    min_delta, x_l = minmax(space)
    max_delta, x_r = maxmin(space)
    if x_r == x_l:
        x_r += step
    return np.linspace(min(x_l, x_r), max(x_l, x_r), abs(x_r - x_l) / step), min(x_l, x_r), max(x_l, x_r)


''' Pareto region searching '''


def is_in_pareto_region(x):
    return f1(x) <= f1_star and f2(x) >= f2_star


def pareto_region():
    x_lpareto = x_lbound
    while x_lpareto < x_rbound and not is_in_pareto_region(x_lpareto):
        x_lpareto += step
    x_rpareto = x_lpareto
    while x_rpareto < x_rbound and is_in_pareto_region(x_rpareto):
        x_rpareto += step
    return np.linspace(x_lpareto, x_rpareto, (x_rpareto - x_lpareto) / step), x_lpareto, x_rpareto


''' MAIN '''


def main():
    fig, ax = plotter.subplots()
    f1_plot, = plotter.plot(all_interval, f1(all_interval))
    f2_plot, = plotter.plot(all_interval, f2(all_interval))

    # Make the shaded Pareto region
    ix, a, b = pareto_region()
    print("Pareto region = [ %.3f" % a, " , %.3f" % b, " ]", sep="")
    iy = f1(ix)
    for i in range(len(ix) - 1):
        if f2(ix[i]) > f1(ix[i]):
            iy[i] = f2(ix[i])
    verts = [(a, -100)] + list(zip(ix, iy)) + [(b, -100)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax.add_patch(poly)

    # Make the shaded MM region
    ix2, a2, b2 = minmax_maxmin_region()
    print("Minmax/maxmin region = [ %.3f" % a2, " , %.3f" % b2, " ]", sep="")
    iy2 = f1(ix2)
    for i2 in range(len(ix2) - 1):
        if f2(ix2[i2]) > f1(ix2[i2]):
            iy2[i2] = f2(ix2[i2])
    verts2 = [(a2, -100)] + list(zip(ix2, iy2)) + [(b2, -100)]
    poly2 = Polygon(verts2, facecolor='0.5', edgecolor='0.2')
    ax.add_patch(poly2)

    plotter.legend((f1_plot, f2_plot, poly, poly2), ('f1(x)', 'f2(x)', 'Pareto region', 'Minmax/maxmin region'),
                   loc='lower right', shadow=True)
    plotter.xlabel('x')
    plotter.ylabel('y')
    plotter.grid(True)
    plotter.show()


''' LAUNCHER '''

main()