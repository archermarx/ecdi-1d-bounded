import argparse
import sys

####################################################################
#                     COMMAND LINE ARGUMENTS                       #                  
####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('n0', type = float, nargs = "?", default = 1e17)
parser.add_argument('ppc', type = int, nargs = "?", default = 200)
parser.add_argument('N', type = int, nargs = "?", default = 200)
parser.add_argument('prefix', type = str, nargs = "?", default = "")

args = parser.parse_args()
n0 = args.n0
particles_per_cell = args.ppc
num_cells = args.N
prefix = args.prefix

quiet_start = False  # Whether to use low-discrepancy sampling to load particles

if (quiet_start):
    diag_name = "quiet"
else:
    diag_name = "noisy"

diag_name += f"_{prefix}_{n0}_{particles_per_cell}_{num_cells}"

subcycling_interval = 11

print(diag_name)
####################################################################
#                            IMPORTS                               #                  
####################################################################

import numpy as np
import cupy as cp
from math import sqrt, ceil, floor

from pywarpx import callbacks, fields, libwarpx, particle_containers, picmi
from periodictable import elements

from scipy import stats
from scipy.stats import qmc


import time

m_p = picmi.constants.m_p
m_e = picmi.constants.m_e
q_e = picmi.constants.q_e
ep0 = picmi.constants.ep0

####################################################################
#                     CONFIGURABLE OPTIONS                         #                  
####################################################################

verbose = False            # Whether to use verbose output
num_grids = 1              # Number of subgrids to decompose domain into
debye_factor = 1.5            # Grid cells per debye length
dt_factor = 3.0             # Timestep factor. dt = dt_factor * dx / v_ExB
particles_per_cell = 200    # Number of particles per cell
#dt = 2e-12
dt = 5e-12
seed = 11235813             # Random seed

#L = 26.7e-3         # Simulation domain length (m)
L = 5e-3
L_axial = 1e-2     # Virtual axial length
max_time = 2e-6    # Max time (s)
num_diags = 100     # Number of diagnostic outputs
#n0 = 1e17           # Plasma density
B0 = 2e-2           # Magnetic field strength (T)
E0 = 20e3           # Electric field strength (V/m)
species = "Xe"      # Ion species
T_e = 2.0          # Electron temperature (eV)
T_i = 0.1           # Ion temperature (eV)
u_i = 0.0         # Axial ion velocity (m/s)

####################################################################
#                        DERIVED VALUES                            #                  
####################################################################

m_i = elements.symbol(species).mass * m_p       # Ion mass
v_ExB = E0 / B0                                 # E x B drift speed
ve_rms = sqrt(q_e * T_e / m_e)                  # Electron rms speed
vi_rms = sqrt(q_e * T_i / m_i)                  # Ion rms speed
lambda_d = sqrt(ep0 * T_e / n0 / q_e)           # Electron debye length

#num_cells = ceil(L / lambda_d) * debye_factor   # Number of computational cells
dx = L / num_cells                              # Cell width
#dt = dx / v_ExB / dt_factor                     # Timestep

max_grid_size = ceil(num_cells/num_grids/2) * 2 # Maximum size of decomposed grids

max_steps = ceil(max_time / dt)                 # Maximum simulation steps 
diag_inter_time = max_time / num_diags          # Interval between diagnostic outputs (s)
diag_inter_iter = round(diag_inter_time / dt)   # Interval between diagnostic outputs (iters)

####################################################################
#                       SIMULATION SETUP                           #                  
####################################################################

# Grid
grid = picmi.Cartesian1DGrid(
    number_of_cells = [num_cells],
    warpx_max_grid_size = max_grid_size,
    warpx_blocking_factor = 1,
    lower_bound = [0],
    upper_bound = [L],
    lower_boundary_conditions = ['periodic'],
    upper_boundary_conditions = ['periodic'],
    lower_boundary_conditions_particles = ['periodic'],
    upper_boundary_conditions_particles = ['periodic']
)

# Field solver
solver = picmi.ElectrostaticSolver(grid=grid)

# Initialize simulation
sim = picmi.Simulation(
    solver = solver,
    time_step_size = dt,
    max_time = max_time,
    verbose = verbose,
    warpx_use_filter = True,
    warpx_field_gathering_algo = 'energy-conserving',
    warpx_serialize_initial_conditions = True,
    warpx_random_seed = seed,
    warpx_sort_intervals = 500
)
solver.sim = sim

# Applied fields
external_field = picmi.ConstantAppliedField(
    Ex = E0,
    By = B0
)
sim.add_applied_field(external_field)

# Particles
#particle_layout = picmi.GriddedLayout(n_macroparticle_per_cell = [particles_per_cell], grid = grid)
particle_layout = picmi.GriddedLayout(n_macroparticle_per_cell = [0], grid = grid)

num_particles = round(num_cells * particles_per_cell)

# Particle positions
x = np.zeros(num_particles)
y = np.zeros(num_particles)
z = np.linspace(0, L, num_particles+2)[1:-1]

# Bulk velocities
bulk_u_i = cp.array([u_i, 0.0, 0.0])
bulk_u_e = cp.array([0.0, 0.0, v_ExB])

# Distribution parameters
cp.random.seed(seed)

# mean and cov as numpy arrays
mean = np.zeros(3)
cov_i = vi_rms**2 * np.identity(3)
cov_e = ve_rms**2 * np.identity(3)

if quiet_start:
    qmc_engine = qmc.Halton(d = 3)
    dist_i = qmc.MultivariateNormalQMC(mean = mean, cov = cov_i, engine = qmc_engine)
    dist_e = qmc.MultivariateNormalQMC(mean = mean, cov = cov_e, engine = qmc_engine)
    ui = dist_i.random(num_particles)
    ue = dist_e.random(num_particles)
else:
    dist_i = stats.multivariate_normal(mean = mean, cov = cov_i)
    dist_e = stats.multivariate_normal(mean = mean, cov = cov_e)
    ui = dist_i.rvs(size = num_particles)
    ue = dist_e.rvs(size = num_particles)

weight = L * n0 / num_particles * np.ones(num_particles)

electrons = picmi.Species(
    particle_type = 'electron', name = 'electrons',
)
sim.add_species(electrons, layout = particle_layout)

ions = picmi.Species(
    particle_type = species, name = 'ions', mass = m_i, charge = 'q_e',
    warpx_do_subcycling = true, warpx_subcycling_interval = subcycling_iterval
)
sim.add_species(ions, layout = particle_layout)

# Diagnostics
field_diag = picmi.FieldDiagnostic(
    name = diag_name,
    grid = grid,
    period = diag_inter_iter,
    data_list = ['rho_ions', 'rho_electrons', 'Ez', 'j'],
    write_dir = 'diags',
    warpx_format = "openpmd"
)
sim.add_diagnostic(field_diag)

particle_diag = picmi.ParticleDiagnostic(
    name = diag_name,
    period = diag_inter_iter,
    write_dir = 'diags',
    warpx_format = "openpmd"
)
sim.add_diagnostic(particle_diag)

# Initialize simulation
sim.initialize_inputs()

####################################################################
#                           CALLBACKS                              #                  
####################################################################

def initialize_particles():

    elec_wrapper = particle_containers.ParticleContainerWrapper('electrons')
    elec_wrapper.add_real_comp('x_pos')

    ion_wrapper = particle_containers.ParticleContainerWrapper('ions')
    ion_wrapper.add_real_comp('x_pos')

    # Particles start randomly-distributed in x
    initial_pos = np.random.uniform(0.0, L_axial, size = num_particles)

    elec_wrapper.add_particles(
        x = x, y = y, z = z,
        ux = ue[:, 0] + bulk_u_e[0].get(),
        uy = ue[:, 1] + bulk_u_e[1].get(),
        uz = ue[:, 2] + bulk_u_e[2].get(),
        w = weight,
        x_pos = initial_pos,
        unique_particles = True
    )

    ui_x = np.sqrt(ui[:, 0]**2 + 2 * q_e / m_i * initial_pos * E0)
    
    ion_wrapper.add_particles(
        x = x, y = y, z = z,
        ux = ui_x,
        uy = ui[:, 1] + bulk_u_i[1].get(),
        uz = ui[:, 2] + bulk_u_i[2].get(),
        w = weight,
        x_pos = initial_pos,
        unique_particles = True
    )
    
callbacks.installafterinit(initialize_particles)

sim.initialize_warpx()

elec_wrapper = particle_containers.ParticleContainerWrapper('electrons')
ion_wrapper = particle_containers.ParticleContainerWrapper('ions')

# load kernels from file
name_exp = ["_wrap_particles<float>", "_wrap_particles<double>"]
kernel_file = "kernel.cu"
code = open(kernel_file).read()
mod = cp.RawModule(code = code, options = ('-std=c++20', ), name_expressions = name_exp)
k_wrap_particles_float = mod.get_function(name_exp[0])
k_wrap_particles_double = mod.get_function(name_exp[1])

def _adjust_velocity(wrapper, bulk_u, v_rms, L_axial, dt):
    xs  = wrapper.get_particle_arrays('x_pos', 0)
    uxs = wrapper.get_particle_arrays('ux', 0)
    uys = wrapper.get_particle_arrays('uy', 0)
    uzs = wrapper.get_particle_arrays('uz', 0)
    
    # CUDA device parameters
    num_blocks = 512
    threads_per_block = 512

    # Iterate over grids
    for (i, (x, ux, uy, uz)) in enumerate(zip(xs, uxs, uys, uzs)):        
        # Get number of particles
        N = x.size

        # Generate needed random numbers (can this be done in-place or inside of the kernel?)
        rands = cp.random.randn(N, 3)

        # Launch kernel to wrap particle positions and randomize velocities as necessary
        k_wrap_particles_double((num_blocks, ), (threads_per_block, ), (
            N, x, ux, uy, uz, bulk_u, v_rms, rands[:, 0], rands[:, 1], rands[:, 2], L_axial, dt)
        )

def adjust_velocity():
    #t = time.time()
    _adjust_velocity(elec_wrapper, bulk_u_e, ve_rms, L_axial, dt)
    _adjust_velocity(ion_wrapper,  bulk_u_i, vi_rms, L_axial, dt)
    #elapsed = time.time() - t
    #print("Elapsed time: ", elapsed, "s.")

# Install callback
callbacks.installbeforestep(adjust_velocity)

####################################################################
#                        RUN SIMULATION                            #                  
####################################################################
sim.step(max_steps)
