import numpy as np
import logging
import sys
import os
import subprocess
from pathlib import Path
from typing import List
from scipy.stats import qmc
from dataclasses import dataclass, asdict, field
import json
import random
from copy import deepcopy
from typing import Literal

from jinja2 import Environment, FileSystemLoader
import shutil

from datetime import datetime

import gmsh
import math


@dataclass
class GlobalConfig:
    run_name: str = "default"

    # ------ Path inputs -----------
    base_dir: Path = Path("/home/ericy/Rocket_CFD_ML")
    openfoam_dir: Path = base_dir / "openfoam"
    automation_dir: Path = openfoam_dir / "Automation"

    template_dir : Path = automation_dir / "template"
    data_dir : Path = base_dir / "data"
    

    # ------- Geomtry Limit inputs ------------
    sample_space: dict = field(default_factory=lambda: {
        "L": [8.0, 15.0], 
        "H": [2.0, 5.0], 
        "R": [0.5, 2.0],
        "fin_root": [1.0, 4.0], 
        "fin_tip": [0.5, 2.5],
        "fin_height": [1.0, 3.0], 
        "thickness": [0.05, 0.2]
    })


    # ---------- Mesh Inputs --------
    # -- blockmesh --
    far_field_multiplier: float = 8.33  # (8.33 * 3 = 24.99 ~ 25)
    inlet_z_multiplier: float =  4  # (-7.69 * 13 = -99.97 ~ -100)
    outlet_z_multiplier: float = -7.69   # (2.31 * 13 = 30.03 ~ 30)

    # Background mesh density to match (10x10x40)
    # cells = (multiplier * dimension) * cells_per_m
    # 25 * 0.4 = 10 cells
    # 130 * 0.308 = 40 cells
    cells_per_m_xy: float = 0.4        
    cells_per_m_z: float = 0.308


    # -- snappy hex mesh ---
    # Refinement Levels
    level_5: int = 5
    level_4: int = 4
    level_3: int = 3
    level_2: int = 2
    
    # Distance Multipliers (based on your 0.2, 0.5, 1.0, 2.0 pattern)
    # These are multiples of R (e.g., if R=1.0, dist_5 is 0.2m)
    dist_5: float = 0.2  
    dist_4: float = 0.5  
    dist_3: float = 1.0  
    dist_2: float = 2.0
    
    
    #--- Gmsh cad creation ---
    fin_embed_depth: float = 0.05
    mesh_resolution: float = 0.05

    #execution related files
    completed_filename = "completed"
    edit_file_list = [Path("system/controlDict"), Path("system/controlDict.compressible"), Path("system/controlDict.incompressible"), Path("system/blockMeshDict"), Path("system/snappyHexMeshDict")]


    def __post_init__(self):
        self.automation_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_dir / "data" / self.run_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class RocketDesign:
# --- Inputs with Defaults ---
    id: str
    L: float = 0.0
    H: float = 0.0
    R: float = 0.0
    fin_root: float = 0.0
    fin_tip: float = 0.0
    fin_height: float = 0.0
    thickness: float = 0.0
    is_component: bool = False
    solver: str = "potential"
    component_type: Literal["auto", "body", "nose", "fin", "composite", "unknown"] = "auto"

    # --- Derived Attributes (Keep these as init=False) ---
    total_length: float = field(init=False)
    total_R: float = field(init=False)
    ref_area: float = field(init=False)
    quadrant_area: float = field(init=False)
    
    # --- Domain & Mesh ---
    # -- blockMesh --
    far_field: float = field(init=False)
    inlet_z: float = field(init=False)
    outlet_z: float = field(init=False)
    cells_x: int = field(init=False)
    cells_y: int = field(init=False)
    cells_z: int = field(init=False)
    
    # -- snappy hex mesh --
    # SnappyHexMesh Refinement Shell Distances
    r_dist_5: float = field(init=False)
    r_dist_4: float = field(init=False)
    r_dist_3: float = field(init=False)
    r_dist_2: float = field(init=False)

    # SnappyHexMesh Mesh Location Point
    loc_x: float = field(init=False)
    loc_y: float = field(init=False)
    loc_z: float = field(init=False)






    label: str = field(init=False)

    def __post_init__(self):
        # 1. Component Type Auto-Detection
        if self.component_type == "auto":
            if not self.is_component:
                self.component_type = "composite"
            elif self.fin_height > 0:
                self.component_type = "fin"
            elif self.H > 0 and self.L == 0:
                self.component_type = "nose"
            elif self.L > 0 and self.H == 0:
                self.component_type = "body"
            else:
                self.component_type = "unknown"

        # 1. Component Guarantee
        if self.is_component:
            self.solver = "potential_simple_rho"

        # 2. Rounding Primary Inputs (excludes meta fields)
        for field_name, field_def in self.__dataclass_fields__.items():
            # Only process fields that were part of the __init__ (geometric inputs)
            if field_def.init:
                val = getattr(self, field_name)
                if isinstance(val, (float, int)):
                    setattr(self, field_name, round(float(val), 3))

        # 3. Create a Succinct Directory Label
        # We multiply by 1000 and convert to int to remove decimals for cleaner folder names
        r_str = f"R{int(self.R*1000)}"
        l_str = f"L{int(self.L*1000)}"
        h_str = f"H{int(self.H*1000)}"
        
        # Fin dimensions: Root, Tip, Height, Thickness
        f_str = (f"FR{int(self.fin_root*1000)}_"
                 f"FT{int(self.fin_tip*1000)}_"
                 f"FH{int(self.fin_height*1000)}_"
                 f"Th{int(self.thickness*1000)}")
        
        self.label = f"{r_str}_{l_str}_{h_str}_{f_str}"

    def finalize_design(self, config: GlobalConfig):
        """
        Calculates all physics-based domain values using the GlobalConfig factors.
        """
        # 3. Derived Properties
        self.total_length = round(self.L + self.H, 3)
        self.ref_area = round(3.14159 * (self.R**2), 4)
        self.quadrant_area = round(self.ref_area / 4, 4)

        # ------------ Mesh Properties -----------------

        # --- blockMesh ---
        # 4. Physics-Based Domain Calculations
        # Total span of the rocket (Centerline to fin tip)
        self.total_R = round(self.R + self.fin_height, 3)
        
        # 2. Domain Bounds (Using Config Multipliers)
        self.far_field = round(self.total_R * config.far_field_multiplier, 2)
        self.inlet_z = round(self.total_length * config.inlet_z_multiplier, 2)
        self.outlet_z = round(self.total_length * config.outlet_z_multiplier, 2)
        
        # 3. Mesh Resolution (Using Config Multipliers)
        self.cells_x = int( np.abs(self.far_field * config.cells_per_m_xy))
        self.cells_y = int( np.abs(self.far_field * config.cells_per_m_xy))
        self.cells_z = int(np.abs((self.outlet_z - self.inlet_z) * config.cells_per_m_z))

        # -- snappy hex mesh --
        # Calculate the 4 physical shells
        self.r_dist_5 = round(self.total_R * config.dist_5, 3)
        self.r_dist_4 = round(self.total_R * config.dist_4, 3)
        self.r_dist_3 = round(self.total_R * config.dist_3, 3)
        self.r_dist_2 = round(self.total_R * config.dist_2, 3)

        # Safety: Location in Mesh (offset from the origin)
        self.loc_x = round(self.far_field * 0.9, 2)
        self.loc_y = round(self.far_field * 0.9, 2)
        self.loc_z = 0.0 # Staying on the XY plane for safety

class ModularGenerator:
    def __init__(self, config):
        self.config = config

    def generate_library(self, n_radii: int, n_bodies_per_r: int, n_noses_per_r: int, n_fins: int):
            r_sampler = qmc.LatinHypercube(d=1)
            radii = qmc.scale(r_sampler.random(n=n_radii), 
                            self.config.sample_space["R"][0], 
                            self.config.sample_space["R"][1]).flatten()

            body_library, nose_library = [], []

            for i, r in enumerate(radii):
                # --- Bodies ---
                l_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_bodies_per_r), 
                                    self.config.sample_space["L"][0], self.config.sample_space["L"][1]).flatten()
                for j, l in enumerate(l_samples):
                    design = RocketDesign(id=f"body_r{i}_l{j}", R=r, L=l, is_component=True)
                    design.finalize_design(self.config) # Calculates mesh/domain
                    body_library.append(design)

                # --- Noses ---
                h_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_noses_per_r), 
                                    self.config.sample_space["H"][0], self.config.sample_space["H"][1]).flatten()
                for j, h in enumerate(h_samples):
                    design = RocketDesign(id=f"nose_r{i}_h{j}", R=r, H=h, is_component=True)
                    design.finalize_design(self.config)
                    nose_library.append(design)

            # --- Fins ---
            fin_library = []
            fin_vars = ["fin_root", "fin_tip", "fin_height", "thickness"]
            fin_sampler = qmc.LatinHypercube(d=4)
            while len(fin_library) < n_fins:
                scaled = qmc.scale(fin_sampler.random(n=1), 
                                [self.config.sample_space[p][0] for p in fin_vars],
                                [self.config.sample_space[p][1] for p in fin_vars])[0]
                f_params = dict(zip(fin_vars, scaled))
                
                # Geometric validation check
                if f_params["fin_root"] >= f_params["fin_tip"]:
                    design = RocketDesign(id=f"fin_{len(fin_library):02d}", is_component=True, **f_params)
                    design.finalize_design(self.config)
                    fin_library.append(design)

            return body_library, nose_library, fin_library

    def assemble_composites(self, body_lib, nose_lib, fin_lib, samples_per_body=3):
        composites = []
        fin_deck = deepcopy(fin_lib)
        random.shuffle(fin_deck)
        
        count = 0
        for body in body_lib:
            # Match by radius within tolerance
            matching_noses = [n for n in nose_lib if abs(n.R - body.R) < 1e-4]
            
            for _ in range(samples_per_body):
                if not fin_deck:
                    fin_deck = deepcopy(fin_lib)
                    random.shuffle(fin_deck)
                
                fin = fin_deck.pop()
                nose = random.choice(matching_noses)
                
                # Assemble the full rocket
                comp = RocketDesign(
                    id=f"comp_{count:04d}", 
                    R=body.R, 
                    L=body.L, 
                    H=nose.H,
                    fin_root=fin.fin_root, 
                    fin_tip=fin.fin_tip, 
                    fin_height=fin.fin_height, 
                    thickness=fin.thickness,
                    is_component=False 
                )
                
                # IMPORTANT: Calculate the unique domain for this composite
                comp.finalize_design(self.config)
                composites.append(comp)
                count += 1
                
        return composites

    def generate_reference_case(self):
        """Generates the specific reference case based on cad_designer_v2.py"""
        design = RocketDesign(
            id="reference",
            L=10.0,
            H=3.0,
            R=1.0,
            fin_root=2.0,
            fin_tip=1.0,
            fin_height=2.0,
            thickness=0.1,
            is_component=False,
            solver="potential_simple_rho"
        )
        design.finalize_design(self.config)
        return design


class CaseGenerator:
    def __init__(self, config, rocket_design):
        self.config = config
        self.rocket_design = rocket_design

        self.data_dir = Path(self.config.data_dir)
        self.template_dir = Path(self.config.template_dir)
        self.new_case_dir = self.data_dir / self.rocket_design.id
        self.geometry_dir = self.new_case_dir / "constant" / "triSurface"
        
        # We leave this empty because the folder doesn't exist yet!
        self.env = None

    def create_case(self): 
        # 1. Clean up old failed attempts so shutil doesn't crash
        if self.new_case_dir.exists():
            completed_flag = self.new_case_dir / self.config.completed_filename
            if completed_flag.exists():
                return False 
            shutil.rmtree(self.new_case_dir)

        # 2. Copy the template into the data folder
        shutil.copytree(self.template_dir, self.new_case_dir)
        
        # 3. NOW that the folder exists, tell Jinja2 to look inside it
        self.loader = FileSystemLoader(str(self.new_case_dir))
        self.env = Environment(loader=self.loader)
        
        return True

    def update_case_files(self):
        # Safety check to ensure create_case ran first
        if not self.env:
            return

        for path in self.config.edit_file_list:
            # path = "system/controlDict"
            self.render_file(path)

    def render_file(self, rel_path):
        # rel_path = "system/controlDict"
        template_name = f"{rel_path}"
        
        # 1. Pull the template from the local case folder
        template = self.env.get_template(template_name)
        
        # 2. Render
        output_text = template.render(
            rocket=self.rocket_design, 
            config=self.config
        )
        
        # 3. Write it back to the local case folder
        target_path = self.new_case_dir / rel_path
        with open(target_path, "w") as f:
            f.write(output_text)
            
    def _run_command_logged(self, cmd_list, check=True):
        """Runs a command and logs its output in real-time."""
        with subprocess.Popen(
            cmd_list,
            cwd=self.new_case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        ) as proc:
            for line in proc.stdout:
                logging.info(line.rstrip())
        
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd_list)

    def run(self):
        """Executes the OpenFOAM automation scripts using subprocess."""
        
        # 1. Run Meshing (Allmesh.sh)
        try:
            logging.info(f"[{self.rocket_design.id}] Running Allmesh.sh...")
            self._run_command_logged(["bash", "Allmesh.sh"], check=True)
        except subprocess.CalledProcessError:
            logging.error(f"Meshing failed for {self.rocket_design.id}. Aborting.")
            return

        # 2. Determine Solver Flags based on the solver string
        # e.g., "potential_simple_rho" -> ["-p", "-s", "-r"]
        flags = []
        if "potential" in self.rocket_design.solver: flags.append("-p")
        if "simple" in self.rocket_design.solver:    flags.append("-s")
        if "rho" in self.rocket_design.solver:       flags.append("-r")

        # 3. Run Solver (Allrun.sh)
        logging.info(f"[{self.rocket_design.id}] Running Allrun.sh with flags: {flags}")
        self._run_command_logged(["bash", "Allrun.sh"] + flags, check=False)

    def generate_geometry(self):
        gmsh.initialize()
        gmsh.model.add("Rocket_Quadrant_Final")

        # --- 1. PARAMETERS ---
        L = self.rocket_design.L
        H = self.rocket_design.H
        R = self.rocket_design.R
        FinRoot = self.rocket_design.fin_root
        FinTip = self.rocket_design.fin_tip
        FinHeight = self.rocket_design.fin_height
        Thick = self.rocket_design.thickness
        EmbedDepth = self.config.fin_embed_depth
        resolution = self.config.mesh_resolution

        gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
        gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)

        # --- 2. CREATE CORE STRUCTURE ---
        core_parts = []
        if L > 0:
            body = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, R)
            core_parts.append((3, body))
        if H > 0:
            nose = gmsh.model.occ.addCone(0, 0, L, 0, 0, H, R, 0)
            core_parts.append((3, nose))

        # Resolve Core Tag Safely
        core_tag = None
        if len(core_parts) > 1:
            fuse_core = gmsh.model.occ.fuse([core_parts[0]], [core_parts[1]])
            core_tag = fuse_core[0][0][1]
        elif len(core_parts) == 1:
            core_tag = core_parts[0][1]
        # If len is 0, core_tag remains None (Correct for Fin-only cases)

        # --- 3. CREATE FINS ---
        tool_fins = []
        if FinHeight > 0:
            # We use R here even if R=0 because the fin needs an anchor point
            p1 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, 0)
            p2 = gmsh.model.occ.addPoint(R + FinHeight, 0, 0)
            p3 = gmsh.model.occ.addPoint(R + FinHeight, 0, FinTip)
            p4 = gmsh.model.occ.addPoint(R - EmbedDepth, 0, FinRoot)
            
            lines = [gmsh.model.occ.addLine(p1, p2), gmsh.model.occ.addLine(p2, p3),
                    gmsh.model.occ.addLine(p3, p4), gmsh.model.occ.addLine(p4, p1)]
            loop = gmsh.model.occ.addCurveLoop(lines)
            fin_surf = gmsh.model.occ.addPlaneSurface([loop])

            gmsh.model.occ.translate([(2, fin_surf)], 0, -Thick/2, 0)
            master_fin_res = gmsh.model.occ.extrude([(2, fin_surf)], 0, Thick, 0)
            master_fin = master_fin_res[1][1]
            tool_fins.append((3, master_fin))

            fin2_list = gmsh.model.occ.copy([(3, master_fin)])
            gmsh.model.occ.rotate(fin2_list, 0, 0, 0, 0, 0, 1, math.pi/2)
            tool_fins.append(fin2_list[0])

        # --- 4. ASSEMBLE MASTER TAG ---
        if core_tag is not None and tool_fins:
            # Full Rocket
            rocket_full = gmsh.model.occ.fuse([(3, core_tag)], tool_fins)
            rocket_tag = rocket_full[0][0][1]
        elif core_tag is not None:
            # Body/Nose Only
            rocket_tag = core_tag
        elif tool_fins:
            # Fin Only: Fuse the two quadrant fins together
            if len(tool_fins) > 1:
                fin_fuse = gmsh.model.occ.fuse([tool_fins[0]], [tool_fins[1]])
                rocket_tag = fin_fuse[0][0][1]
            else:
                rocket_tag = tool_fins[0][1]
        else:
            raise ValueError(f"No valid geometry for case {self.rocket_design.id}")

        # --- 4. THE QUADRANT SLICE ---
        S = 50.0 
        slicing_box = gmsh.model.occ.addBox(0, 0, -10, S, S, L + H + 20)
        
        # INTERSECT returns exactly like FUSE
        quadrant_result = gmsh.model.occ.intersect([(3, rocket_tag)], [(3, slicing_box)])
        quadrant_tag = quadrant_result[0][0][1] # This tag is the ONLY one that matters now

        gmsh.model.occ.synchronize()

        # --- 5. SURFACE FILTERING ---
        surfaces = gmsh.model.getBoundary([(3, quadrant_tag)], combined=False)
        rocket_faces = []

        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            on_x_sym = math.isclose(com[0], 0, abs_tol=1e-5)
            on_y_sym = math.isclose(com[1], 0, abs_tol=1e-5)
            
            # This logic ensures we ONLY export the curved "rocket" skin
            if not (on_x_sym or on_y_sym):
                rocket_faces.append(tag)

        gmsh.model.addPhysicalGroup(2, rocket_faces, name="rocket")

        # --- 6. EXPORT ---
        # Force Gmsh to ONLY save the "rocket" physical group, not the symmetry planes
        gmsh.option.setNumber("Mesh.SaveAll", 0) 
        
        gmsh.model.mesh.generate(2)
        gmsh.write(str(self.geometry_dir / "rocket_quadrant.stl"))
        # gmsh.fltk.run()

        # gmsh.fltk.run() # Kept for your debugging
        gmsh.finalize()




def assign_nested_solvers(composites, n_simple=20, n_rho=10):
    """
    Assigns solvers to the composites list. 
    Note: I adjusted the defaults to smaller numbers for your preview.
    """
    # 1. Start all as potential
    for c in composites:
        c.solver = "potential"

    # 2. Shuffle to spread out high-fidelity cases
    random.shuffle(composites)

    # 3. Assign hierarchical stack
    for i, c in enumerate(composites):
        if i < n_rho:
            c.solver = "potential_simple_rho"
        elif i < n_simple:
            c.solver = "potential_simple"
            
    return composites

def summarize_library(bodies, noses, fins, composites, base_dir):
    """
    Encapsulates the state of the generated library, prints it to console,
    and writes a permanent record to a text file using full solver names.
    """
    all_cases = bodies + noses + fins + composites
    total_count = len(all_cases)
    report_path = base_dir / "generation_report.txt"
    
    # Helper to count solver occurrences accurately
    def get_solver_stats(group):
        stats = {
            "potential_simple_rho": 0, 
            "potential_simple": 0, 
            "potential": 0
        }
        for d in group:
            stats[d.solver] = stats.get(d.solver, 0) + 1
        return stats

    report = []
    report.append("="*95)
    report.append(f"{'OPENFOAM CFD GENERATION REPORT':^95}")
    report.append(f"{str(datetime.now()):^95}")
    report.append("="*95)
    
    # Updated Table Header with Full Names
    report.append(f"{'CATEGORY':<18} | {'COUNT':<6} | {'POTENTIAL_SIMPLE_RHO':<22} | {'POTENTIAL_SIMPLE':<18} | {'POTENTIAL'}")
    report.append("-" * 95)

    groups = [
        ("Body Tubes", bodies), 
        ("Nose Cones", noses), 
        ("Fin Sets", fins), 
        ("Composites", composites)
    ]

    for name, group in groups:
        s = get_solver_stats(group)
        report.append(
            f"{name:<18} | {len(group):<6} | "
            f"{s['potential_simple_rho']:<22} | "
            f"{s['potential_simple']:<18} | "
            f"{s['potential']}"
        )

    report.append("-" * 95)
    total_s = get_solver_stats(all_cases)
    report.append(
        f"{'TOTALS':<18} | {total_count:<6} | "
        f"{total_s['potential_simple_rho']:<22} | "
        f"{total_s['potential_simple']:<18} | "
        f"{total_s['potential']}"
    )
    report.append("="*95)

    # 2. Technical Spec Preview (Random Sample)
    if composites:
        sample = random.choice(composites)
        report.append(f"\n[TECHNICAL PREVIEW: {sample.id}]")
        report.append(f"  > Geometry:       R={sample.R}m, L={sample.L}m, H={sample.H}m")
        report.append(f"  > Domain Bounds:  R_far={sample.far_field}m, Z_span=[{sample.inlet_z}, {sample.outlet_z}]")
        report.append(f"  > Shells (m):     L5:{sample.r_dist_5} | L4:{sample.r_dist_4} | L3:{sample.r_dist_3} | L2:{sample.r_dist_2}")
        report.append(f"  > Seed Point:     ({sample.loc_x}, {sample.loc_y}, {sample.loc_z})")
        report.append(f"  > Target Solver:  {sample.solver}")
    
    report.append("\n" + "="*95 + "\n")

    full_report_text = "\n".join(report)

    # Action: Print and Save
    logging.info(full_report_text)
    with open(report_path, "w") as f:
        f.write(full_report_text)
    
    logging.info(f"Report logged to: {report_path}")

def main():
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        run_name = input("Enter run name (subdirectory of data/): ").strip()
        if not run_name: run_name = "default"

    config = GlobalConfig(run_name=run_name)
    
    # Setup Logging
    log_file = config.data_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    generator = ModularGenerator(config)
    
    # 1. Generate Components
    bodies, noses, fins = generator.generate_library(5, 2, 2, 20)
    
    # 2. Assemble Composites
    composites = generator.assemble_composites(bodies, noses, fins, samples_per_body=3)
    
    # 3. Generate Reference Case
    reference_case = generator.generate_reference_case()

    # 3. Assign Solvers (Note: 10 Rho, 20 Simple, rest Potential)
    assign_nested_solvers(composites, n_simple=20, n_rho=10)

    #4 Provide stats to user and request confirmation to continue
    summarize_library(bodies, noses, fins, composites, config.data_dir)

    # 5. Save JSON
    output_plan = config.data_dir / "simulation_plan.json"
    plan_data = {
        "reference": [asdict(reference_case)],
        "components": { # Added colon here
            "bodies": [asdict(b) for b in bodies],
            "noses": [asdict(n) for n in noses],
            "fins": [asdict(f) for f in fins],
        }, # Added comma here
        "composites": [asdict(d) for d in composites]
    }
    
    with open(output_plan, 'w') as f:
        json.dump(plan_data, f, indent=4)
        
    logging.info(f"Success! Plan saved to {output_plan}")

    user_input = input("\nProceed with case generation? (y/n): ").lower()

    if user_input != 'y':
        logging.info("Aborting run.")
        return

    # 6. Creating cases
    logging.info("Creating & running cases...")
    case_data = {
        "reference": [reference_case],
        "bodies": [b for b in bodies],
        "noses": [n for n in noses],
        "fins": [f for f in fins],
        "composites": [d for d in composites]
    }

    for component_type in case_data:
        for design in case_data[component_type]:

            logging.info(design)
            case = CaseGenerator(config, design)

            # case.generate_geometry()
            case.create_case()
            case.update_case_files()
            case.generate_geometry()
            case.run()

            # sys.exit()
            # break
        


if __name__ == "__main__":
    main()
