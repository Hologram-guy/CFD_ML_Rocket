import pyvista as pv
import os

# 1. Setup paths
case_dir = "/home/ericy/Rocket_CFD_ML/openfoam/.archive/template_v2_3"
foam_file = os.path.join(case_dir, "case.foam")

# 2. Create the dummy .foam file if it doesn't exist
if not os.path.exists(foam_file):
    with open(foam_file, 'w') as f:
        pass

# 3. Read the case
reader = pv.POpenFOAMReader(foam_file)

# 4. Set to the latest time step (e.g., your '1' folder)
reader.set_active_time_value(reader.time_values[-1])

# 5. Load the mesh
mesh = reader.read()

# 6. Access the internal mesh or specific patches
# mesh is a MultiBlock dataset. Usually, [0] is internalMesh
# internal_mesh = mesh["internalMesh"]

# print(f"Mesh loaded successfully!")
# print(f"Cells: {internal_mesh.n_cells}")
# print(f"Available Data: {internal_mesh.array_names}")

# 7. Access your Pressure data (p)
# pressure = internal_mesh.point_data["p"]


#

rocket_surface = mesh["boundary"]["rocket"]


# Now 'rocket_surface' only contains the points and faces of the rocket itself
print(f"Surface Points: {rocket_surface.n_points}")
# This is the modern, supported way to count the triangles/quads on the surface
print(f"Surface Faces: {rocket_surface.n_cells}")
print(f"Available Data: {rocket_surface.array_names}")
