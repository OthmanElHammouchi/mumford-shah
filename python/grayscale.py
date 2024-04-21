import os
import math
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, dot, TrialFunction, TestFunction
from mpi4py import MPI
import cv2
import numpy as np
import matplotlib.pyplot as plt

ELEMS = 500  # per unit of length

# Read image and extract dimension information
image = cv2.imread(os.path.join("data", "image", "per-enflo-goose.png"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Lx = float(image.shape[1]) / float(image.shape[0])
Ly = 1.0
hx = Lx / float(image.shape[1] - 1)
hy = Ly / float(image.shape[0] - 1)

# Create mesh
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [[0, 0], [Lx, Ly]], [math.floor(Lx * ELEMS), math.floor(Ly * ELEMS)]
)


# Interpolate input image on mesh
def image_fun(x):
    global hx, hy
    res = np.zeros(x.shape[1])
    for k in range(x.shape[1]):
        j = int(math.floor(x[0, k] / hx))
        i = int(math.floor((1 - x[1, k]) / hy))
        res[k] = image[i, j]
    return res


V = fem.FunctionSpace(domain, ("Lagrange", 1))
g = fem.Function(V)
g.interpolate(image_fun)


# Create helper functions to solve the subproblems
def grayscale_edges(u, eps, V):
    global domain
    v = TrialFunction(V)
    w = TestFunction(V)

    a = (
        dot(grad(u), grad(u)) * v * w * dx
        - eps * dot(grad(v), grad(w)) * dx
        + 1 / (4 * eps) * v * w * dx
    )
    L = fem.Constant(domain, 1 / (4 * eps)) * w * dx

    v = fem.Function(V)
    problem = LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    vh = problem.solve()
    return vh


def grayscale_cartoon(v, g, V):
    u = TrialFunction(V)
    w = TestFunction(V)

    L = g * w * dx
    a = u * w * dx + (v**2) * dot(grad(u), grad(w)) * dx

    u = fem.Function(V)
    problem = LinearProblem(
        a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()
    return uh


# Segment the image by alternatingly iterating over the subproblems
def segment_image(g, V, eps, maxiter=5):
    u = g
    iter = 1
    while iter <= maxiter:
        v = grayscale_edges(u, eps, V)
        u = grayscale_cartoon(v, g, V)
        iter += 1
    return u, v


u, v = segment_image(g, V, 1e-6)

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
cartoon = u.eval(points, cells).reshape((ELEMS + 1, -1))
cartoon = np.abs(cartoon.min()) + cartoon
cartoon = 255 * cartoon / cartoon.max()
cartoon = cartoon.astype(np.uint8)

# Evaluate edges on mesh and normalise them between 0 and 255
edges = v.eval(points_on_proc, cells).reshape((ELEMS + 1, -1))
edges = np.abs(edges.min()) + edges
edges = 255 * edges / edges.max()
edges = edges.astype(np.uint8)

# Save cartoon and edges
plt.imsave(os.path.join("results", "grayscale_cartoon.png"), cartoon, cmap="grey")
plt.imsave(os.path.join("results", "grayscale_edges.png"), edges, cmap="grey")
