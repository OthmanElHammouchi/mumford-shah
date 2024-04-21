import os
import math
from dolfinx import mesh, io, fem, plot, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, dot, TrialFunction, TestFunction, inner, TestFunctions, split
from mpi4py import MPI
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image and extract dimension information
image = cv2.imread(os.path.join("data", "image", "erdos-tao.jpg"))
Lx = float(image.shape[1]) / float(image.shape[0])
Ly = 1.0
hx = Lx / float(image.shape[1] - 1)
hy = Ly / float(image.shape[0] - 1)

# Create mesh
ELEMS = 500  # per unit of length
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0, 0], [Lx, Ly]],
    [int(math.floor(Lx * ELEMS)), int(math.floor(Ly * ELEMS))],
)


# Interpolate input image on mesh
def image_fun(x):
    global hx, hy
    res = np.zeros((3, x.shape[1]))
    for k in range(x.shape[1]):
        j = int(math.floor(x[0, k] / hx))
        i = int(math.floor((Ly - x[1, k]) / hy))
        res[:, k] = image[i, j, :]
    return res


U = fem.FunctionSpace(domain, ("Lagrange", 1, (3,)))
V = fem.FunctionSpace(domain, ("Lagrange", 1))
g = fem.Function(U)
g.interpolate(image_fun)


# Create helper functions to solve the subproblems
def colour_edges(u, eps, V):
    global domain
    v = TrialFunction(V)
    w = TestFunction(V)

    a = (
        v * inner(grad(u), grad(u)) * w * dx
        + v * 1 / (4 * eps) * w * dx
        - eps * dot(grad(v), grad(w)) * dx
    )

    L = fem.Constant(domain, 1 / (4 * eps)) * w * dx

    v = fem.Function(V)
    problem = LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    vh = problem.solve()
    return vh


def colour_cartoon(v, g, U):
    g1, g2, g3 = split(g)

    u = TrialFunction(U)
    u1, u2, u3 = split(u)

    w1, w2, w3 = TestFunctions(U)

    a = (
        u1 * w1
        + (v**2) * dot(grad(u1), grad(w1))
        + u2 * w2
        + (v**2) * dot(grad(u2), grad(w2))
        + u3 * w3
        + (v**2) * dot(grad(u3), grad(w3))
    ) * dx

    L = (g1 * w1 + g2 * w2 + g3 * w3) * dx

    u = fem.Function(U)
    problem = LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    return uh


# Segment the image by alternatingly iterating over the subproblems
def segment_image(g, U, V, eps, maxiter=5):
    cartoon = g
    iter = 1
    while iter <= maxiter:
        edges = colour_edges(cartoon, eps, V)
        cartoon = colour_cartoon(edges, g, U)
        iter += 1
    return cartoon, edges


u, v = segment_image(g, U, V, 1e-6)


# Build array of mesh vertex coordinates, see https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
points = domain.geometry.x
points = points[
    np.lexsort(
        (points[:, 0], -points[:, 1])
    )  # Ensures correct orientation for plt.imshow
]

bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, points)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

cells = []
points_on_proc = []
for i, point in enumerate(points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)

# Evaluate cartoon colour components on mesh and normalise them between 0 and 255
comps = list(u.split())
comps = [comp.eval(points_on_proc, cells).reshape((ELEMS + 1, -1)) for comp in comps]
for i in range(len(comps)):
    comps[i] = np.abs(comps[i].min()) + comps[i]
    comps[i] = 255 * comps[i] / comps[i].max()
    comps[i] = comps[i].astype(np.uint8)

cartoon = np.stack(
    comps,
    axis=-1,
)

# Evaluate edges on mesh and normalise them between 0 and 255
edges = v.eval(points_on_proc, cells).reshape((ELEMS + 1, -1))
edges = np.abs(edges.min()) + edges
edges = 255 * edges / edges.max()
edges = edges.astype(np.uint8)


# Save cartoon and edges
plt.imsave(
    os.path.join("results", "colour_cartoon.png"),
    cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB),
)
plt.imsave(os.path.join("results", "colour_edges.png"), edges, cmap="grey")
