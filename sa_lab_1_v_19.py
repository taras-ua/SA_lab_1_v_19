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
    print("MAXMIN = %.5f" % f0, "in x = %.3f" % x0)
    return f0, x0


def minmax(space):
    f0 = max_f(space[0])
    x0 = space[0]
    for i in range(len(space)):
        print("x = %.3f" % space[i], "f1'/f1 = %.5f" % (f1_star / f1(space[i])), sep='\t', end='\t')
        print("f2/f2' = %.5f" % (f2(space[i]) / f2_star), end='\t')
        print("max = %.5f" % max_f(space[i]), end='\t')
        print("min = %.5f" % min_f(space[i]))
        if max_f(space[i]) <= f0:
            f0 = max_f(space[i])
            x0 = space[i]
    print("\n\nMINMAX = %.5f" % f0, "in x = %.3f" % x0)
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
    for i in range(len(ix)):
        iy[i] = max_of_f1_f2_star
    verts = [(a, min_of_f1_f2_star)] + list(zip(ix, iy)) + [(b, min_of_f1_f2_star)]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.8')
    ax.add_patch(poly)

    # Make the shaded MM region
    ix2, a2, b2 = minmax_maxmin_region()
    iy2 = f1(ix2)
    for i2 in range(len(ix2)):
        iy2[i2] = max_of_f1_f2_star
    verts2 = [(a2, min_of_f1_f2_star)] + list(zip(ix2, iy2)) + [(b2, min_of_f1_f2_star)]
    poly2 = Polygon(verts2, facecolor='0.5', edgecolor='0.5')
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

''' Maxmin functions '''


def maxmin12(out=False):
    if out:
        print("x1 \\ x2", end='\t')
    for i in range(len(x2_interval)):
        if out:
            print("x2=%.2f" % x2_interval[i], end='\t')
    if out:
        print()
    max_x1 = f12(x1_lbound, x2_lbound)
    for j in range(len(x1_interval)):
        if out:
            print("x1=%.2f" % x1_interval[j], end='\t')
        min_x2 = f12(x1_interval[j], x2_lbound)
        for i in range(len(x2_interval)):
            f_val = f12(x1_interval[j], x2_interval[i])
            min_x2 = min(min_x2, f_val)
            if out:
                print("%.3f" % f_val, end='\t')
        max_x1 = max(max_x1, min_x2)
        if out:
            print()
    return max_x1


def maxmin21(out=False):
    if out:
        print("x2 \\ x1", end='\t')
    for i in range(len(x1_interval)):
        if out:
            print("x1=%.2f" % x1_interval[i], end='\t')
    if out:
        print()
    max_x2 = f21(x1_lbound, x2_lbound)
    for j in range(len(x2_interval)):
        if out:
            print("x2=%.2f" % x2_interval[j], end='\t')
        min_x1 = f21(x1_lbound, x2_interval[j])
        for i in range(len(x1_interval)):
            f_val = f21(x1_interval[i], x2_interval[j])
            min_x1 = min(min_x1, f_val)
            if out:
                print("%.3f" % f_val, end='\t')
        max_x2 = max(max_x2, min_x1)
        if out:
            print()
    return max_x2


''' Pareto region searching '''


def is_in_pareto_region_2(x1, x2, f12_lim, f21_lim):
    return f12(x1, x2) >= f12_lim and f21(x1, x2) >= f21_lim


def pareto_region_2(f12_lim, f21_lim):
    pareto_reg = []
    ix1 = x1_lbound
    while ix1 <= x1_rbound:
        ix2 = x2_lbound
        while ix2 <= x2_rbound:
            if is_in_pareto_region_2(ix1, ix2, f12_lim, f21_lim):
                pareto_reg += [[ix1, ix2]]
            ix2 += step_2
        ix1 += step_2
    return pareto_reg


''' x1, x2 optimal '''


def min_delta_12(f12_star, region):
    delta = abs(f12(region[0][0], region[0][1]) - f12_star)
    delta_x1 = region[0][0]
    delta_x2 = region[0][1]
    i = 1
    while i < len(region):
        new_val = abs(f12(region[i][0], region[i][1]) - f12_star)
        if delta > new_val:
                delta = new_val
                delta_x1 = region[i][0]
                delta_x2 = region[i][1]
        i += 1
    return delta, delta_x1, delta_x2


def min_delta_21(f21_star, region):
    delta = abs(f21(region[0][0], region[0][1]) - f21_star)
    delta_x1 = region[0][0]
    delta_x2 = region[0][1]
    i = 1
    while i < len(region):
        new_val = abs(f21(region[i][0], region[i][1]) - f21_star)
        if delta > new_val:
                delta = new_val
                delta_x1 = region[i][0]
                delta_x2 = region[i][1]
        i += 1
    return delta, delta_x1, delta_x2


''' MAIN '''


def task_2():
    f12_star = maxmin12()
    print("\nf12' = %.3f" % f12_star, end='\n\n')
    f21_star = maxmin21()
    print("\nf21' = %.3f" % f21_star, end='\n\n')
    pareto = pareto_region_2(f12_star, f21_star)

    opt12, optx1_12, optx2_12 = min_delta_12(f12_star, pareto)
    opt21, optx1_21, optx2_21 = min_delta_21(f21_star, pareto)
    print("Delta[1>2] = %.3f" % opt12, end='\t')
    print("Opt x1 = %.2f" % optx1_12, end='\t')
    print("Opt x2 = %.2f" % optx2_12)
    print("Delta[2>1] = %.3f" % opt21, end='\t')
    print("Opt x1 = %.2f" % optx1_21, end='\t')
    print("Opt x2 = %.2f" % optx2_21)

    plot_print_increment = 5
    print("Gonna print %i dots. " % (len(pareto) / plot_print_increment), end='')
    if input("Continue? [y/n]: ") == "y":
        i = 0
        while i <= len(pareto):
            plotter.scatter(pareto[i][0], pareto[i][1], c='blue', s=50,
                            alpha=0.3, edgecolors='none')
            i += plot_print_increment
        plotter.xlabel('x1')
        plotter.ylabel('x2')
        plotter.grid(True)
        plotter.show()


''' LAUNCHER '''

task_2()