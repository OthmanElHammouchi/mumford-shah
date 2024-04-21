import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import sympy


def J(u):
    integral = sympy.integrate(((t * u.diff(t)) ** 2), (t, -1, 1))
    return integral


t, epsilon = sympy.symbols("t epsilon")
u = sympy.atan(t / epsilon) / sympy.atan(1 / epsilon)

fig, ax = plt.subplots()

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
fig.patch.set_alpha(0.0)
fig.set_dpi(327)

x_data, y_data = [], []
(line,) = ax.plot([], [])
text = ax.text(-0.75, 1.25, " ", fontsize="x-large")
integral = J(u)


def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    return line


def update(frame):
    u_eps = sympy.lambdify(t, u.subs(epsilon, frame))
    x_data = np.linspace(-1, 1, 1000)
    y_data = u_eps(x_data)
    line.set_data(x_data, y_data)
    val = integral.subs(epsilon, frame).simplify()
    text.set_text(r"$J(u_\epsilon) = {:.3f}$".format(val))
    return line


eps_vals = np.flip((10 * np.ones(100)) ** np.linspace(-4, 0, 100))
animation = FuncAnimation(
    fig, update, frames=eps_vals, init_func=init, repeat=True
)

writervideo = FFMpegWriter(fps=10, codec="vp9")
writervideo.setup(fig, os.path.join("results", "dirichlet.webm"))
animation.save(os.path.join("results", "dirichlet.webm"), writer=writervideo, dpi=327)

writervideo = FFMpegWriter(fps=10, codec="libx265")
writervideo.setup(fig, os.path.join("results", "dirichlet.mp4"))
animation.save(os.path.join("results", "dirichlet.mp4"), writer=writervideo, dpi=327)
