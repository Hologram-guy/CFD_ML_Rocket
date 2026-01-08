import gmsh
import math

gmsh.initialize()
gmsh.model.add("Rocket_Quadrant_Final")

# --- 1. PARAMETERS (Preserved from your original) ---
L, H, R = 10.0, 3.0, 1.0
FinRoot, FinTip, FinHeight, Thick = 2.0, 1.0, 2.0, 0.1
EmbedDepth = 0.05
resolution = 0.05 

# Set Resolution Options
gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

# --- 2. CREATE FULL ROCKET ---
# Core Body
body = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
nose = gmsh.model.occ.addCone(0, 0, L, 0, 0, H, R, 0)
core_tag = gmsh.model.occ.fuse([(3, body)], [(3, nose)])[0][0][1]

# Master Fin (Centered on Y=0 via Translate)
p1 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, 0)
p2 = gmsh.model.occ.addPoint(R + FinHeight, 0, 0)
p3 = gmsh.model.occ.addPoint(R + FinHeight, 0, FinTip)
p4 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, FinRoot)
lines = [gmsh.model.occ.addLine(p1, p2), gmsh.model.occ.addLine(p2, p3),
         gmsh.model.occ.addLine(p3, p4), gmsh.model.occ.addLine(p4, p1)]
loop = gmsh.model.occ.addCurveLoop(lines)
fin_surf = gmsh.model.occ.addPlaneSurface([loop])

# THE CRITICAL STEP: Center the fin on the Y-axis
gmsh.model.occ.translate([(2, fin_surf)], 0, -Thick/2, 0)
master_fin = gmsh.model.occ.extrude([(2, fin_surf)], 0, Thick, 0)[1][1]

# Copy only the 90-degree fin (we only need the +X and +Y fins for a quadrant)
fin2_list = gmsh.model.occ.copy([(3, master_fin)])
gmsh.model.occ.rotate(fin2_list, 0, 0, 0, 0, 0, 1, math.pi/2)
fin2 = fin2_list[0][1]

# Final Fusion of the relevant parts
rocket_full = gmsh.model.occ.fuse([(3, core_tag)], [(3, master_fin), (3, fin2)])
rocket_tag = rocket_full[0][0][1]

# --- 3. THE QUADRANT SLICE ---
S = 50.0 
# Box covering +X, +Y
slicing_box = gmsh.model.occ.addBox(0, 0, -10, S, S, L + H + 20)
quadrant_result = gmsh.model.occ.intersect([(3, rocket_tag)], [(3, slicing_box)])
quadrant_tag = quadrant_result[0][0][1]

gmsh.model.occ.synchronize()

# --- 4. SURFACE FILTERING FOR SNAPPY ---
# Get all surfaces of the quadrant rocket
surfaces = gmsh.model.getBoundary([(3, quadrant_tag)], combined=False)
rocket_faces = []

for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(2, tag)
    # Filter out internal "cut" faces sitting on X=0 or Y=0
    on_x_sym = math.isclose(com[0], 0, abs_tol=1e-5)
    on_y_sym = math.isclose(com[1], 0, abs_tol=1e-5)
    
    if not (on_x_sym or on_y_sym):
        rocket_faces.append(tag)

# Create the Physical Group for export
gmsh.model.addPhysicalGroup(2, rocket_faces, name="rocket")

# --- 5. EXPORT ---
gmsh.model.mesh.generate(2)
gmsh.write("rocket_quadrant.stl")

gmsh.fltk.run()
gmsh.finalize()