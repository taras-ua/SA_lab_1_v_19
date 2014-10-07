import matplotlib.pyplot as plotter
from matplotlib.patches import Polygon
import numpy as np

#############################################
###########         TASK 1        ###########
#############################################

''' Target functions '''


def f1(x):
    return 5 * np.log10(x + 9)


def f2(x):
    return -12 + 4 * x


def f1_s(x):
    return f1_star + x * 0


def f2_s(x):
    return f2_star + x * 0


''' Target constants '''

f1_star = 10.0
f2_star = 2.0
x_lbound = 0.001
x_rbound = 5.0
step = 0.001
eps = 0.0001
all_interval = np.linspace(x_lbound, x_rbound, (x_rbound - x_lbound) / step)

''' Minmax/maxmin functions '''


def min_f(x):
    return min(f1_star / f1(x), f2(x) / f2_star)


def max_f(x):
    return max(f1_star / f1(x), f2(x) / f2_star)


def maxmin(space):
    f0 = min_f(space[0])
    x0 = space[0]
    for i in range(len(space) - 1):
        if min_f(space[i]) >= f0:
            f0 = min_f(space[i])
            x0 = space[i]
    print("MAXMIN = %.3f" % f0, "in x = %.3f" % x0)
    return f0, x0


def minmax(space):
    f0 = max_f(space[0])
    x0 = space[0]
    for i in range(len(space) - 1):
        print("x = %.3f" % space[i], "f1'/f1 = %.4f" % (f1_star / f1(space[i])), sep='\t', end='\t')
        print("f2/f2' = %.4f" % (f2(space[i]) / f2_star), end='\t')
        print("max = %.4f" % max_f(space[i]), end='\t')
        print("min = %.4f" % min_f(space[i]))
        if max_f(space[i]) <= f0:
            f0 = max_f(space[i])
            x0 = space[i]
    print("\n\nMINMAX = %.3f" % f0, "in x = %.3f" % x0)
    return f0, x0


def minmax_maxmin_region():
    space, a, b = pareto_region()
    min_delta, x_l = minmax(space)
    max_delta, x_r = maxmin(space)
    if x_r == x_l:
        x_r += step / 4
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


def task_1():
    fig, ax = plotter.subplots()
    f1_plot, = plotter.plot(all_interval, f1(all_interval))
    f2_plot, = plotter.plot(all_interval, f2(all_interval))
    f1_star_plot, = plotter.plot(all_interval, f1_s(all_interval))
    f2_star_plot, = plotter.plot(all_interval, f2_s(all_interval))

    min_of_f1_f2_star = min(f1_star, f2_star)
    max_of_f1_f2_star = max(f1_star, f2_star)

    # Make the shaded Pareto region
    ix, a, b = pareto_region()
    iy = f1(ix)
    for i in range(len(ix) - 1):
        iy[i] = max_of_f1_f2_star
    verts = [(a, min_of_f1_f2_star)] + list(zip(ix, iy)) + [(b, min_of_f1_f2_star)]
    poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
    ax.add_patch(poly)

    # Make the shaded MM region
    ix2, a2, b2 = minmax_maxmin_region()
    iy2 = f1(ix2)
    for i2 in range(len(ix2) - 1):
        iy2[i2] = max_of_f1_f2_star
        iy2[i2] -= (max_of_f1_f2_star - min_of_f1_f2_star) / 2
    verts2 = [(a2, min_of_f1_f2_star)] + list(zip(ix2, iy2)) + [(b2, min_of_f1_f2_star)]
    poly2 = Polygon(verts2, facecolor='0.5', edgecolor='0.2')
    ax.add_patch(poly2)


    print("\n\nPareto region = [ %.3f" % a, " , %.3f" % b, " ]", sep="")
    print("Minmax/maxmin region = [ %.3f" % a2, " , %.3f" % b2, " ]", sep="")

    plotter.legend((f1_plot, f2_plot, f1_star_plot, f2_star_plot, poly, poly2),
                   ('f1(x)', 'f2(x)', 'f1\'', 'f2\'', 'Pareto region', 'Minmax/maxmin region'),
                   loc='lower right', shadow=True)
    plotter.xlabel('x')
    plotter.ylabel('y')
    plotter.grid(True)
    plotter.show()


#############################################
###########         TASK 2        ###########
#############################################


''' Target functions '''


def f12(x1, x2):
    return 3 * x1 ** 2 + 4 * x2 ** 2 + 5 * x1 * x2


def f21(x1, x2):
    return 0.8 * x2 + x1 * x2 - 0.4 * x1 + 1


''' Target constants '''


x1_lbound = 0.0
x1_rbound = 4.0
x2_lbound = 0.0
x2_rbound = 3.0
step_2 = 0.01
x1_interval = np.linspace(x1_lbound, x1_rbound, (x1_rbound - x1_lbound) / step_2)
x2_interval = np.linspace(x2_lbound, x2_rbound, (x2_rbound - x2_lbound) / step_2)


def task_2():
    print()

''' LAUNCHER '''

task_1()