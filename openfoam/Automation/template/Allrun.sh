#!/bin/bash

# --- CONFIGURATION ---
CORES=4
ARCHIVE_DIR="saved_stages"
FORCES_DIR="saved_forces"
SURFACES_DIR="saved_surfaces"

# Active Files
CONTROL="system/controlDict"
SCHEMES="system/fvSchemes"
OPTIONS="system/fvOptions"

# Template/Backup Files
DICT_INCOMP="system/controlDict.incompressible"
DICT_COMP="system/controlDict.compressible"

SCHEMES_INCOMP="system/fvSchemes.incompressible"
SCHEMES_COMP="system/fvSchemes.compressible"

OPTIONS_INCOMP="system/fvOptions.incompressible"
OPTIONS_COMP="system/fvOptions.compressible"

# --- ARGUMENT PARSING ---
RUN_POTENTIAL=false; RUN_SIMPLE=false; RUN_RHO=false; KEEP_VOLUME=false; DELETE_STL=false
while getopts "psrkd" opt; do
    case $opt in
        p) RUN_POTENTIAL=true ;;
        s) RUN_SIMPLE=true ;;
        r) RUN_RHO=true ;;
        k) KEEP_VOLUME=true ;;
        d) DELETE_STL=true ;;
        *) echo "Usage: $0 [-p] [-s] [-r] [-k (keep volume)] [-d (delete stl)]"; exit 1 ;;
    esac
done

# --- FUNCTIONS ---

get_last_log_time() {
    local logfile=$1
    if [ -f "$logfile" ]; then
        grep "^Time = " "$logfile" | tail -n 1 | awk '{print $3}' | tr -d '\r'
    fi
}

cleanup() {
    rm -rf processor*
}
trap cleanup EXIT

init_setup() {
    echo "Starting Fresh: Cleaning directories and resetting 0/ from 0.orig..."
    # rm -rf "$ARCHIVE_DIR" "$FORCES_DIR"
    # mkdir -p "$ARCHIVE_DIR" "$FORCES_DIR"
    rm -rf "$ARCHIVE_DIR" "$FORCES_DIR" "$SURFACES_DIR"
    mkdir -p "$ARCHIVE_DIR" "$FORCES_DIR" "$SURFACES_DIR"
    
    clean_time_folders
    # RESET 0 FOLDER ONLY ONCE AT THE START
    rm -rf 0 && cp -r 0.orig 0
    cp 0.orig/U.slip 0/U
    cp 0.orig/p.incompressible 0/p
}

archive_forces() {
    local stage_name=$1
    if [ -d "postProcessing" ]; then
        mkdir -p "$FORCES_DIR/$stage_name"
        cp -r postProcessing/* "$FORCES_DIR/$stage_name/"
        rm -rf postProcessing
    fi
}

extract_surfaces_and_clean() {
    local stage_name=$1
    local time_dir=$2

    echo "Extracting surfaces for time: $time_dir"
    
    # Run the surfaces function object defined in system/surfaces
    postProcess -func surfaces -time "$time_dir" > "log.surfaces_$stage_name" 2>&1

    # Move the generated VTK files to safe storage
    mkdir -p "$SURFACES_DIR/$stage_name"
    # surfaces outputs to postProcessing/surfaces/TIME/
    if [ -d "postProcessing/surfaces/$time_dir" ]; then
        cp -r "postProcessing/surfaces/$time_dir"/* "$SURFACES_DIR/$stage_name/"
        rm -rf postProcessing
    fi

    # CLEANUP: Delete the heavy volume data (the time folder)
    echo "Deleting volume data for $time_dir to save space..."
    rm -rf "$time_dir"
}

run_potential() {
    echo "--- STAGE 1: POTENTIAL FLOW ---"
    # Ensure incompressible dictionary is in place
    cp "$DICT_INCOMP" "$CONTROL"

    cp "$SCHEMES_INCOMP" "$SCHEMES"

    cp "$OPTIONS_INCOMP" "$OPTIONS"

    decomposePar > log.decomposePar
    # -writep is critical to pass pressure to simpleFoam
    mpirun -np $CORES potentialFoam -parallel -writep > log.potentialFoam
    
    # Archiving
    # 1. Reconstruct the latest time found in processors
    local actual_time=0

    # 3. Create 1234 folder in the processors (NEED THIS, BECAUSE RUNNING THE RECONSTRUCT ON TIME 0 CAUSES ERRORS)
    # COPY instead of MOVE to preserve 0 for simpleFoam
    for proc in processor*; do
        if [ -d "$proc/0" ]; then
            cp -r "$proc/0" "$proc/1234"
        fi
    done

    reconstructPar -time 1234 > log.reconstructPot


    # Force post processing
    simpleFoam -postProcess -time 1234 > log.potForces
    archive_forces "01_potentialFoam"

    if [ "$KEEP_VOLUME" = true ]; then
        mkdir -p "$ARCHIVE_DIR/01_potential"
        cp -r 1234/* "$ARCHIVE_DIR/01_potential/"
    else
        extract_surfaces_and_clean "01_potential" "1234"
    fi
    
    for proc in processor*; do rm -rf "$proc/1234"; done


}

run_simple() {
    echo "--- STAGE 2: SIMPLEFOAM (Inheriting Potential Flow) ---"
    # !!! REMOVED RESET HERE !!! 
    # simpleFoam now sees the U and p files potentialFoam wrote in processor*/0/

    # Swap to noSlip AFTER potentialFoam finishes
    sed -i '/rocket/,/}/ s/type.*slip;/type noSlip;/g' processor*/0/U
    
    mpirun -np $CORES simpleFoam -parallel > log.simpleFoam
    
    local latest=$(get_last_log_time "log.simpleFoam")
    if [ -z "$latest" ]; then
        echo "ERROR: simpleFoam failed to produce a time step. Check log.simpleFoam."
        return 1
    fi

    reconstructPar -time "$latest" > log.reconstructSimple
    
    archive_forces "02_simpleFoam"
    if [ "$KEEP_VOLUME" = true ]; then
        mkdir -p "$ARCHIVE_DIR/02_simple_incompressible"
        cp -r "$latest"/* "$ARCHIVE_DIR/02_simple_incompressible/"
    else
        extract_surfaces_and_clean "02_simple_incompressible" "$latest"
    fi
}

run_rhocentral() {
    echo "--- STAGE 3: RHOCENTRALFOAM (Compressible) ---"
    # 1. Swap ALL configurations
    cp "$DICT_COMP" "$CONTROL"
    cp "$SCHEMES_COMP" "$SCHEMES"
    cp "$OPTIONS_COMP" "$OPTIONS"

    # 2. Reset Fields to Compressible "Source of Truth"
    # We must wipe simpleFoam's kinematic p and replace with absolute p
    rm -f 0/p 0/T 0/U 0/phi 0/rho
    cp 0.orig/p.compressible 0/p
    [ -f "0.orig/T.compressible" ] && cp 0.orig/T.compressible 0/T || cp 0.orig/T 0/T
    cp 0.orig/U.coldStart 0/U

    # 3. Execution
    rm -rf processor*
    decomposePar > log.decomposePar2
    mpirun -np $CORES rhoCentralFoam -parallel > log.rhoCentralFoam

    local latest=$(get_last_log_time "log.rhoCentralFoam")
    if [ -z "$latest" ]; then
        echo "ERROR: rhoCentralFoam failed to produce a time step. Check log.rhoCentralFoam."
        return 1
    fi

    reconstructPar -time "$latest" > log.reconstructParFinal

    archive_forces "03_rhoCentralFoam"
    if [ "$KEEP_VOLUME" = true ]; then
        mkdir -p "$ARCHIVE_DIR/03_rho_final"
        cp -r "$latest"/* "$ARCHIVE_DIR/03_rho_final/"
    else
        extract_surfaces_and_clean "03_rho_final" "$latest"
    fi
}

move_logs() {
    echo "Organizing logs..."
    mkdir -p logs
    mv log.* logs/ 2>/dev/null
}

clean_time_folders() {
    echo "Cleaning numerical output directories (Preserving 0 and 0.orig)..."
    # This deletes folders like '1500' or '0.01' but keeps '0' and anything with 'orig'
    ls -d [0-9]* 2>/dev/null | grep -v "orig" | grep -v "^0$" | xargs rm -rf 2>/dev/null
}
# --- MAIN EXECUTION ---
init_setup

# # Use the && logic to ensure stages only run if requested
$RUN_POTENTIAL && run_potential
$RUN_SIMPLE && run_simple
$RUN_RHO && run_rhocentral

# --- FINAL CLEANUP ---
echo "--- FINALIZING CASE ---"
clean_time_folders
move_logs
touch completed

if [ "$KEEP_VOLUME" = false ]; then
    # AGGRESSIVE CLEANUP: Delete mesh to finalize storage savings
    echo "Deleting mesh (constant/polyMesh) to save final space..."
    rm -rf constant/polyMesh

fi

if [ "$DELETE_STL" = true ]; then
    echo "Deleting geometry (constant/triSurface) to save space..."
    rm -rf constant/triSurface
fi

echo "------------------------------------------------"
echo " Simulation Complete!"
echo " Results:   $ARCHIVE_DIR"
echo " Surfaces:  $SURFACES_DIR"
echo " Forces:    $FORCES_DIR"
echo " Logs:      logs/"
echo "------------------------------------------------"