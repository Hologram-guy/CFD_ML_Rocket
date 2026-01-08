import numpy as np
from pathlib import Path
from typing import List
from scipy.stats import qmc
from dataclasses import dataclass, asdict, field
import json
import random
from copy import deepcopy

@dataclass
class GlobalConfig:
    base_dir: Path = Path("/home/ericy/Rocket_CFD_ML")
    openfoam_dir: Path = base_dir / "openfoam"
    automation_dir: Path = openfoam_dir / "Automation"
    
    sample_space: dict = field(default_factory=lambda: {
        "L": [8.0, 15.0], 
        "H": [2.0, 5.0], 
        "R": [0.5, 2.0],
        "fin_root": [1.0, 4.0], 
        "fin_tip": [0.5, 2.5],
        "fin_height": [1.0, 3.0], 
        "thickness": [0.05, 0.2]
    })
    
    def __post_init__(self):
        self.automation_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class RocketDesign:
    id: str = ""
    L: float = 0.0
    H: float = 0.0 
    R: float = 0.0
    fin_root: float = 0.0
    fin_tip: float = 0.0
    fin_height: float = 0.0
    thickness: float = 0.0
    
    solver: str = "potential"
    is_component: bool = False 
    label: str = "" # The human-readable directory name

    def __post_init__(self):
        # 1. Component Guarantee
        if self.is_component:
            self.solver = "potential_simple_rho"

        # 2. Rounding Primary Inputs (excludes meta fields)
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, (float, int)) and field_name not in ['ref_area', 'quadrant_area']:
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

        # 3. Derived Properties
        self.total_length = round(self.L + self.H, 3)
        self.ref_area = round(3.14159 * (self.R**2), 4)
        self.quadrant_area = round(self.ref_area / 4, 4)

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
            l_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_bodies_per_r), 
                                 self.config.sample_space["L"][0], self.config.sample_space["L"][1]).flatten()
            for j, l in enumerate(l_samples):
                body_library.append(RocketDesign(id=f"body_r{i}_l{j}", R=r, L=l, is_component=True))

            h_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_noses_per_r), 
                                 self.config.sample_space["H"][0], self.config.sample_space["H"][1]).flatten()
            for j, h in enumerate(h_samples):
                nose_library.append(RocketDesign(id=f"nose_r{i}_h{j}", R=r, H=h, is_component=True))

        fin_library = []
        fin_vars = ["fin_root", "fin_tip", "fin_height", "thickness"]
        fin_sampler = qmc.LatinHypercube(d=4)
        while len(fin_library) < n_fins:
            scaled = qmc.scale(fin_sampler.random(n=1), 
                               [self.config.sample_space[p][0] for p in fin_vars],
                               [self.config.sample_space[p][1] for p in fin_vars])[0]
            f_params = dict(zip(fin_vars, scaled))
            if f_params["fin_root"] >= f_params["fin_tip"]:
                fin_library.append(RocketDesign(id=f"fin_{len(fin_library):02d}", is_component=True, **f_params))

        return body_library, nose_library, fin_library

    def assemble_composites(self, body_lib, nose_lib, fin_lib, samples_per_body=3):
        composites = []
        fin_deck = deepcopy(fin_lib)
        random.shuffle(fin_deck)
        
        count = 0
        for body in body_lib:
            matching_noses = [n for n in nose_lib if abs(n.R - body.R) < 1e-4]
            for _ in range(samples_per_body):
                if not fin_deck:
                    fin_deck = deepcopy(fin_lib); random.shuffle(fin_deck)
                
                fin = fin_deck.pop()
                nose = random.choice(matching_noses)
                
                # FIX: We only want the geometric attributes from the fin, 
                # NOT the is_component=True flag or the solver name.
                composites.append(RocketDesign(
                    id=f"comp_{count:04d}", 
                    R=body.R, 
                    L=body.L, 
                    H=nose.H,
                    fin_root=fin.fin_root, 
                    fin_tip=fin.fin_tip, 
                    fin_height=fin.fin_height, 
                    thickness=fin.thickness,
                    is_component=False # Explicitly False for composites
                ))
                count += 1
        return composites

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

def main():
    config = GlobalConfig()
    generator = ModularGenerator(config)
    
    # 1. Generate Components
    bodies, noses, fins = generator.generate_library(5, 2, 2, 20)
    
    # 2. Assemble Composites
    dataset = generator.assemble_composites(bodies, noses, fins, samples_per_body=3)
    
    # 3. Assign Solvers (Note: 10 Rho, 20 Simple, rest Potential)
    assign_nested_solvers(dataset, n_simple=20, n_rho=10)

    #4 Provide stats to user and request confirmation to continue
    print(f"Total simulations: {len(bodies)+len(noses)+len(fins)+len(dataset)}")
    print(f"example dataset {dataset[0]}")

    # 5. Save JSON
    output_plan = config.automation_dir / "simulation_plan.json"
    plan_data = {
        "components": {
            "bodies": [asdict(b) for b in bodies],
            "noses": [asdict(n) for n in noses],
            "fins": [asdict(f) for f in fins]
        },
        "composites": [asdict(d) for d in dataset]
    }
    
    with open(output_plan, 'w') as f:
        json.dump(plan_data, f, indent=4)
        
    print(f"Success! Plan saved to {output_plan}")

    # 6. Creating cases


if __name__ == "__main__":
    main()

# import numpy as np
# from pathlib import Path
# from typing import List
# from scipy.stats import qmc
# from dataclasses import dataclass, asdict, field
# import json
# import random
# from copy import deepcopy

# @dataclass
# class GlobalConfig:
#     # 1. Project Directory Structure
#     base_dir: Path = Path("/home/ericy/Rocket_CFD_ML")
#     openfoam_dir: Path = base_dir / "openfoam"
#     template_dir: Path = openfoam_dir / "template_v2_3"
#     archive_dir: Path = openfoam_dir / ".archive"
#     automation_dir: Path = openfoam_dir / "Automation"
    
#     # 2. Design Space (Ranges for Sampling)
#     # Format: [min, max]
#     sample_space: dict = field(default_factory=lambda: {
#         "L": [8.0, 15.0],
#         "H": [2.0, 5.0],
#         "R": [0.5, 2.0],
#         "fin_root": [1.0, 4.0],
#         "fin_tip": [0.5, 2.5],
#         "fin_height": [1.0, 3.0],
#         "thickness": [0.05, 0.2]
#     })
    
#     # 3. Execution Settings
#     n_samples: int = 100
#     solver: str = "rhoCentralFoam"
#     n_procs: int = 1  # For parallel runs later
    
#     def __post_init__(self):
#         # Ensure directories exist
#         self.archive_dir.mkdir(parents=True, exist_ok=True)


# @dataclass
# class RocketDesign:
#     # Body Dimensions
#     L: float           # Body Length
#     H: float           # Nose Cone Height
#     R: float           # Body Radius
    
#     # Fin Dimensions
#     fin_root: float    # Length of fin attached to body
#     fin_tip: float     # Length of fin tip edge
#     fin_height: float   # How far fin sticks out
#     thickness: float   # Fin thickness
    
#     # Solver Assignment (Defaulting to the lowest fidelity)
#     solver: str = "potentialFoam"
#     is_component: bool = False # Flag to distinguish components 
#     id: str = ""


#     # Derived Properties (Calculated automatically)
#     def __post_init__(self):
#         # 1. Component Guarantee: Force full stack if it's a library part
#         if self.is_component:
#             self.solver = "potential_simple_rho"

#         # 2. Round the primary inputs
#         for field_name in self.__dataclass_fields__:
#             val = getattr(self, field_name)
#             if isinstance(val, float):
#                 setattr(self, field_name, round(val, 3))

#         # 3. Calculate derived properties
#         self.total_length = self.L + self.H
#         self.ref_area = round(3.14159 * (self.R**2), 4)
#         self.quadrant_area = round(self.ref_area / 4, 4)

#     def save(self, path):
#         """Saves the design to a JSON file for record keeping."""
#         with open(path, 'w') as f:
#             json.dump(asdict(self), f, indent=4)
    

# # 3. The Generator Class
# class ModularGenerator:
#     def __init__(self, config):
#         self.config = config

#     def generate_library(self, n_radii: int, n_bodies_per_r: int, n_noses_per_r: int, n_fins: int):
#         # --- 1. SAMPLE RADII (The Interface Anchor) ---
#         r_sampler = qmc.LatinHypercube(d=1)
#         radii = qmc.scale(r_sampler.random(n=n_radii), 
#                           self.config.sample_space["R"][0], 
#                           self.config.sample_space["R"][1]).flatten()

#         body_library = []
#         nose_library = []

#         # --- 2. SAMPLE BODIES & NOSES (Grouped by Radius) ---
#         for r in radii:
#             # Sample Body Lengths
#             l_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_bodies_per_r), 
#                                  self.config.sample_space["L"][0], 
#                                  self.config.sample_space["L"][1]).flatten()
#             for l in l_samples:
#                 # Inside generate_library loops:
#                 body_library.append(RocketDesign(
#                     R=r, L=l, H=0, fin_root=0, fin_tip=0, fin_height=0, thickness=0, 
#                     is_component=True
#                 ))

#             # Sample Nose Heights
#             h_samples = qmc.scale(qmc.LatinHypercube(d=1).random(n=n_noses_per_r), 
#                                  self.config.sample_space["H"][0], 
#                                  self.config.sample_space["H"][1]).flatten()
#             for h in h_samples:
#                 nose_library.append({"R": r, "H": h})

#         # --- 3. SAMPLE VALID FINS ---
#         fin_library = []
#         fin_vars = ["fin_root", "fin_tip", "fin_height", "thickness"]
#         fin_sampler = qmc.LatinHypercube(d=4)
        
#         # We use a while loop to ensure we get EXACTLY n_fins valid designs
#         while len(fin_library) < n_fins:
#             raw_sample = fin_sampler.random(n=1)
#             lb = np.array([self.config.sample_space[p][0] for p in fin_vars])
#             ub = np.array([self.config.sample_space[p][1] for p in fin_vars])
#             scaled = qmc.scale(raw_sample, lb, ub)[0]
            
#             f_params = dict(zip(fin_vars, scaled))
            
#             # --- THE GEOMETRIC CHECK ---
#             if f_params["fin_root"] >= f_params["fin_tip"]:
#                 fin_library.append(f_params)
#             else:
#                 # Discard invalid geometry and continue sampling
#                 continue

#         return body_library, nose_library, fin_library

#     def assemble_composites(self, body_lib, nose_lib, fin_lib, samples_per_body=3):
#         """
#         Pairs components together while ensuring every fin in the library 
#         gets simulated across different body types.
#         """
#         composites = []
        
#         # 1. Create a "deck" of fins and shuffle it
#         # We use a list copy so we don't mess up the original library
#         fin_deck = deepcopy(fin_lib)
#         random.shuffle(fin_deck)
        
#         # 2. Iterate through each body in your library
#         for body in body_lib:
#             # Find all noses that physically fit this body (matching Radius)
#             matching_noses = [n for n in nose_lib if n["R"] == body["R"]]
            
#             if not matching_noses:
#                 print(f"Warning: No matching nose found for Radius {body['R']}. Skipping.")
#                 continue

#             for _ in range(samples_per_body):
#                 # 3. If the deck is empty, reshuffle and restart the deck
#                 # This ensures even distribution across the entire dataset
#                 if len(fin_deck) == 0:
#                     fin_deck = deepcopy(fin_lib)
#                     random.shuffle(fin_deck)
                
#                 # 4. "Deal" a fin and a nose
#                 selected_fin = fin_deck.pop()
#                 selected_nose = random.choice(matching_noses)
                
#                 # 5. Build the RocketDesign object
#                 design = RocketDesign(
#                     R=body["R"], 
#                     L=body["L"], 
#                     H=selected_nose["H"], 
#                     **selected_fin
#                 )
#                 composites.append(design)
                
#         return composites
# # --- Example Usage ---

# def print_final_budget(body_lib, nose_lib, fin_lib, composites):
#     n_b = len(body_lib)
#     n_n = len(nose_lib)
#     n_f = len(fin_lib)
#     n_c = len(composites)
    
#     total = n_b + n_n + n_f + n_c
    
#     print("--- TOTAL SIMULATION PLAN ---")
#     print(f"1. Isolated Bodies: {n_b}")
#     print(f"2. Isolated Noses:  {n_n}")
#     print(f"3. Isolated Fins:   {n_f}")
#     print(f"4. Composites:      {n_c}")
#     print(f"GRAND TOTAL:        {total} CFD Runs")
#     print("-----------------------------")

# def assign_nested_solvers(composites, n_simple=200, n_rho=50):
#     """
#     Ensures RhoCentral is a subset of SimpleFoam, 
#     and SimpleFoam is a subset of PotentialFoam.
#     """
#     # 1. Start everyone as PotentialFoam
#     for c in composites:
#         c.solver = "potential"

#     # 2. Shuffle to maintain LHS design space coverage
#     random.shuffle(composites)

#     # 3. The first N_simple cases are promoted to SimpleFoam
#     for i in range(n_simple):
#         composites[i].solver = "potential_simple"

#     # 4. The first N_rho cases (within the SimpleFoam group) 
#     # are further promoted to RhoCentralFoam
#     for i in range(n_rho):
#         composites[i].solver = "potential_simple_rho"

#     # Note: In your automation, you will run ALL solvers for the 
#     # first 50 cases, two for the next 150, and one for the rest.
#     return composites

# def main():
#     # 1. Initialize Global Configuration
#     # This sets up your directories and your design min/max ranges
#     config = GlobalConfig()
    
#     # 2. Initialize the Generator
#     generator = ModularGenerator(config)
    
#     # 3. Create the "Component Library"
#     # We want 5 distinct Radii, 2 Body lengths per radius, 
#     # 2 Nose heights per radius, and 20 unique valid fin designs.
#     print("--- Generating Modular Component Library ---")
#     body_lib, nose_lib, fin_lib = generator.generate_library(
#         n_radii=5, 
#         n_bodies_per_r=2, 
#         n_noses_per_r=2, 
#         n_fins=20
#     )
    
#     print(f"Library Created: {len(body_lib)} Bodies, {len(nose_lib)} Noses, {len(fin_lib)} Fins.")

#     # 4. Assemble the Composite Cases
#     # This pairs the components together into full rocket designs
#     print("\n--- Assembling Composite Cases ---")
#     dataset = generator.assemble_composites(
#         body_lib, 
#         nose_lib, 
#         fin_lib, 
#         samples_per_body=3  # Each body gets paired with 3 different fin/nose combos
#     )
    
#     assign_hierarchical_solvers(dataset)


#     # 5. Save the Master Plan
#     # We save this to a JSON so the automation script can read it and run OpenFOAM
#     output_plan = config.automation_dir / "simulation_plan.json"
    
#     plan_data = {
#         "components": {
#             "bodies": body_lib,
#             "noses": nose_lib,
#             "fins": fin_lib
#         },
#         "composites": [asdict(d) for d in dataset]
#     }
    

#     with open(output_plan, 'w') as f:
#         json.dump(plan_data, f, indent=4)
        
#     print(f"\nSuccess! Master plan saved to {output_plan}")
#     print(f"Total simulations required: {len(body_lib) + len(nose_lib) + len(dataset)}")
    
#     # Optional: Preview the first composite case
#     # print(f"\nPreview Case 0: {dataset[0]}")
#     print_final_budget(plan_data['components']['bodies'], plan_data['components']['noses'], plan_data['components']['fins'], plan_data['composites'])

# if __name__ == "__main__":
#     main()