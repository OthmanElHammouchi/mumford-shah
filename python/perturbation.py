import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

rot_mat = np.array(
    [[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]]
)


def path(t):
    path = np.stack([t, np.sin(t)])
    path = rot_mat @ path
    return path[0, :], path[1, :]


def perturbation(t, q, amp):
    v = amp * t[:: (len(t) // 50)] * np.cos(t[:: (len(t) // 50)])
    u = np.zeros_like(v)
    q = q[:: (len(t) // 50)]
    t = t[:: (len(t) // 50)]
    return t, q, u, v


t = np.linspace(0, 2 * np.pi, int(1e3))
t, q = path(t)
T, Q, u, v = perturbation(t, q, 0.1)

fig, ax = plt.subplots()

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
fig.patch.set_alpha(0.0)
fig.set_dpi(327)

t, q = path(np.linspace(0, 2 * np.pi, int(1e3)))
ax.plot(t, q, color="blue")

T, Q, U, V = perturbation(t, q, 0)
(line,) = ax.plot(T + U, Q + V, color="orange")
quiver = ax.quiver(
    T,
    Q,
    U,
    V,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="blue",
    width=1e-3,
    headaxislength=0,
    headlength=0,
    headwidth=0,
)
text1 = ax.text(
    t[int(np.floor(0.6 * len(t)))],
    q[int(np.floor(0.6 * len(t)))] - 0.3,
    r"$\bm{q} + \bm{\delta q}$",
    fontsize="x-large",
    color="orange",
)
text2 = ax.text(
    t[int(np.floor(0.3 * len(t)))],
    q[int(np.floor(0.3 * len(t)))] - 0.3,
    r"$\bm{q}$",
    fontsize="x-large",
    color="blue",
)


def init():
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 5)


def update(frame):
    T, Q, U, V = perturbation(t, q, frame)
    quiver.set_UVC(U, V)
    line.set_data(T + U, Q + V)

    text2.set_position(
        (
            T[int(np.floor(0.3 * len(T)))],
            Q[int(np.floor(0.3 * len(T)))] - 0.3,
        )
    )
    text1.set_position(
        (
            (T + U)[int(np.floor(0.6 * len(T)))],
            (Q + V)[int(np.floor(0.6 * len(T)))] - 0.3,
        )
    )


amps = [np.linspace(0, 0.3, 50)]
flip = True
for i in range(4):
    amp = np.linspace(-0.3, 0.3, 100)
    if flip:
        amps.append(np.flip(amp))
        flip = False
    else:
        amps.append(amp)
        flip = True
amps = np.concatenate(amps)

animation = FuncAnimation(fig, update, frames=amps, init_func=init, repeat=True)

writervideo = FFMpegWriter(fps=30, codec="vp9")
writervideo.setup(fig, os.path.join("results", "perturbation.webm"))
animation.save(
    os.path.join("results", "perturbation.webm"), writer=writervideo, dpi=327
)

writervideo = FFMpegWriter(fps=30, codec="libx265")
writervideo.setup(fig, os.path.join("results", "perturbation.mp4"))
animation.save(os.path.join("results", "perturbation.mp4"), writer=writervideo, dpi=327)
