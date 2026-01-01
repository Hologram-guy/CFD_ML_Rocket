
# import numpy as np
import gmsh
import sys
import math
gmsh.initialize()
gmsh.model.add("Rocket")

# --- PARAMETERS ---
L = 10.0          # Body Length
H = 3.0           # Nose Cone Height
R = 1.0           # Body Radius
FinRoot = 2.0     # Length of fin attached to body (Z-axis)
FinTip = 1.0      # Length of fin tip edge
FinHeight = 2.0   # How far fin sticks out (X-axis)
Thick = 0.1       # Fin thickness (Y-axis)

# --- FIX PARAMETERS ---
# Define how deep the fin penetrates the body to ensure a good fuse.
# It just needs to be slightly larger than 0.
EmbedDepth = 0.05  # <-- NEW PARAMETER

# --- MESH RESOLUTION ---
resolution = 0.1
gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)


# --- 1. CREATE CORE BODY ---
body = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
nose = gmsh.model.occ.addCone(0, 0, L, 0, 0, H, R, 0)
core_fusion = gmsh.model.occ.fuse([(3, body)], [(3, nose)])
core_tag = core_fusion[0][0][1]


# --- 2. CREATE MASTER FIN ---
# We draw the fin profile in the X-Z plane at Y=0.
# IMPORTANT: The inner points start inside the body radius (R - EmbedDepth)

# Bottom-Inner (Embedded) <-- CHANGED
p1 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, 0)
# Bottom-Outer
p2 = gmsh.model.occ.addPoint(R + FinHeight, 0, 0)
# Top-Outer
p3 = gmsh.model.occ.addPoint(R + FinHeight, 0, FinTip)
# Top-Inner (Embedded) <-- CHANGED
p4 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, FinRoot)

l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
fin_surf = gmsh.model.occ.addPlaneSurface([loop])

# --- CENTER AND EXTRUDE FIN ---
# 1. Move the 2D surface back by half the thickness so it's centered on Y=0
gmsh.model.occ.translate([(2, fin_surf)], 0, -Thick/2, 0) # <-- NEW STEP

# 2. Extrude by the full thickness. It now goes from -Thick/2 to +Thick/2
vol_list = gmsh.model.occ.extrude([(2, fin_surf)], 0, Thick, 0)
master_fin = vol_list[1][1]


# --- 3. ROTATE AND COPY FINS ---
all_fins = [(3, master_fin)]
for i in range(1, 4):
    angle = i * (math.pi / 2)
    new_fin_list = gmsh.model.occ.copy([(3, master_fin)])
    gmsh.model.occ.rotate(new_fin_list, 0, 0, 0, 0, 0, 1, angle)
    all_fins.append(new_fin_list[0])

# --- 4. FUSE FINS TO BODY ---
# We fuse the 'core_tag' with the list 'all_fins'
gmsh.model.occ.fuse([(3, core_tag)], all_fins)


# Synchronize opens the CAD kernel to Gmsh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)




# --- 5. EXPORT AND VIEW ---
gmsh.write("rocket_design.stl")

# Optional: Export as Mesh (uncomment if you generate a mesh)
# gmsh.model.mesh.generate(3)
# gmsh.write("rocket_design.msh")

gmsh.fltk.run()
gmsh.finalize()