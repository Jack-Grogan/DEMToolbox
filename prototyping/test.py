import sys
sys.path.append('../')
import DEMToolbox
from DEMToolbox import ProcessSimulationTimestep

mesh = "mesh_1300000.vtk"
particles = "particles_1300000.vtk"

sim  = ProcessSimulationTimestep(mesh, mesh)