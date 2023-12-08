using AverageShiftedHistograms
using GLMakie
using CairoMakie
using CSV
using DataFrames
using FileIO
using HDF5
using PartialFunctions
using Printf
import PhysicalConstants.CODATA2018 as constants
using Serialization
using Statistics
using TimerOutputs

const q_e = constants.e.val
const k_B = constants.k_B.val
const m_e = constants.m_e.val

const DATA_DIR = "diags"
const ANALYSIS_DIR = "analysis"

function mean_and_var(x; kwargs...)
    μ = mean(x; kwargs...)
    σ2 = varm(x, μ; kwargs...)
    return μ, σ2
end

function get_particle_properties(particle_dset, L, timer)

    @timeit timer "Attributes" begin
        charge = HDF5.attributes(particle_dset["charge"])["value"][]
        mass = HDF5.attributes(particle_dset["mass"])["value"][]
        weight = particle_dset["weighting"][][1]
        N = Int(HDF5.attributes(particle_dset["charge"])["shape"][][])
        density = N *  weight / L
    end

    @timeit timer "Position" begin
        # get position
        position = zeros(N, 3)
        @. position[:, 1] = particle_dset["xPos"][]
        @. position[:, 3] = particle_dset["position"]["z"][]
    end

    @timeit timer "Velocity" begin
        # get velocity
        velocity = zeros(N, 3)
        invmass = inv(mass)
        @. velocity[:, 1] = particle_dset["momentum"]["x"][] * invmass
        @. velocity[:, 2] = particle_dset["momentum"]["y"][] * invmass
        @. velocity[:, 3] = particle_dset["momentum"]["z"][] * invmass
    end

    @timeit timer "Mean and temperature" begin
        # compute mean and variance of velocity
        mean_velocity, var_velocity = mean_and_var(velocity, dims = 1)

        # compute temperature in eV
        temperature = mass * var_velocity / q_e
    end

    return (;charge, mass, weight, density, mean_velocity, position, velocity, temperature)
end

function openpmd_iter(path)
    return tryparse(Int, split(splitext(splitpath(path)[end])[1], "_")[end])
end

function get_data(path, timer)
    @timeit timer "Opening file" begin
        fid = h5open(path, "r")
        iter = openpmd_iter(path)
        data = fid["data"]["$(iter)"]
        time = HDF5.attributes(data)["time"][]
    end

    @timeit timer "Field loading" begin
        fields = data["fields"]
        rho_ions = fields["rho_ions"][]
        rho_electrons = fields["rho_electrons"][]
        Ez = fields["E"]["z"][]
        dz = HDF5.attributes(fields["rho_ions"])["gridSpacing"][][]
        Nz = length(Ez)
        z = LinRange(dz / 2, (Nz - 1) * dz + dz/2, 100)
        field_data = (;z, rho_electrons, rho_ions, Ez)

        L = z[end] + dz / 2
    end

    @timeit timer "Particle property computation" begin
        particles = data["particles"]
        electrons = get_particle_properties(particles["electrons"], L, timer)
        ions = get_particle_properties(particles["ions"], L, timer)
    end

    dir = abspath(joinpath(ANALYSIS_DIR, splitpath(path)[end-1]))

    raw_data = (;
        dir, time, iter, fields = field_data, particles = (;electrons, ions)
    )

    close(fid)

    return raw_data
end

function condense_data!(data)
    N = length(data)
    Nx = length(data[1].fields.Ez)
    Ne = size(data[1].particles.electrons.position, 1)
    Ni = size(data[1].particles.ions.position, 1)
    time = zeros(N)
    iter = zeros(Int, N)
    Ez = zeros(Nx, N)
    ni = zeros( Nx, N)
    ne = zeros(Nx, N)
    x_e = zeros(Ne, 3, N)
    x_i = zeros(Ni, 3, N)
    v_e = zeros(Ne, 3, N)
    v_i = zeros(Ni, 3, N)
    u_e = zeros(3, N)
    u_i = zeros(3, N)
    T_e = zeros(3, N)
    T_i = zeros(3, N)

    for (i, d) in enumerate(data)
        time[i] = d.time
        iter[i] = d.iter
        Ez[:, i] .= d.fields.Ez
        ni[:, i] .= d.fields.rho_ions / d.particles.ions.charge
        ne[:, i] .= d.fields.rho_electrons / -d.particles.electrons.charge
        x_e[:, :, i] .= d.particles.electrons.position
        x_i[:, :, i] .= d.particles.ions.position
        v_e[:, :, i] .= d.particles.electrons.velocity
        v_i[:, :, i] .= d.particles.ions.velocity
        u_e[:, i] .= d.particles.electrons.mean_velocity'
        u_i[:, i] .= d.particles.ions.mean_velocity'
        T_e[:, i] .= d.particles.electrons.temperature'
        T_i[:, i] .= d.particles.ions.temperature'
    end

    return (;
        Nx, Ne, Ni,
        dir = data[1].dir,
        time, iter,
        fields = (;
            z = data[1].fields.z,
            Ez, ni, ne
        ),
        particles = (;
            ions = (;
                charge = data[1].particles.ions.charge,
                mass = data[1].particles.ions.mass,
                weight = data[1].particles.ions.weight,
                density = data[1].particles.ions.density,
                position = x_i,
                velocity = v_i,
                mean_velocity = u_i,
                temperature = T_i,
            ),
            electrons = (;
                charge = data[1].particles.electrons.charge,
                mass = data[1].particles.electrons.mass,
                weight = data[1].particles.electrons.weight,
                density = data[1].particles.electrons.density,
                position = x_e,
                velocity = v_e,
                mean_velocity = u_e,
                temperature = T_e,
            )
        )
    )

end

function isvalidfile(x)
    ext = splitext(x)[2]
    iter = openpmd_iter(x)
    return ext == ".h5" && iter > 0
end

function load_all_data(subfolder; reload = false, time = false)
    timer = TimerOutput()
    data_dir = joinpath(DATA_DIR, subfolder)
    analysis_dir = mkpath(joinpath(ANALYSIS_DIR, subfolder))

    binary_path = joinpath(analysis_dir, "data")
    binary_found = ispath(binary_path)

    data = if (!binary_found || reload)
        files = filter(isvalidfile, readdir(data_dir, join = true))
        @timeit timer "Data loading" begin
            data = [get_data(f, timer) for f in files]
        end
        @timeit timer "Sorting" begin
            sort!(data, by = x -> x.time)
        end
        @timeit timer "Data condensing" begin
            condensed = condense_data!(data)
        end
        @timeit timer "Serialization" begin
            serialize(binary_path, condensed)
        end
        condensed
    else
        deserialize(binary_path)
    end

    if (time)
        show(timer)
    end

    return data
end

function phase_space_density!(density, xs, vs, data, iter, index)
    # get data at iteration, and compute histogram
    x = @views data.position[:, index, iter]
    v = @views data.velocity[:, index, iter]

    smoothing_factor = 1.25
    mx = round(Int, 5 * smoothing_factor)
    my = round(Int, 5 * smoothing_factor)
    hist = ash(x, v; mx, my)

    # partition density array into chunks
    nchunks = length(density) ÷ Threads.nthreads()
    chunks = Iterators.partition(density, nchunks)

    # 1d to 2d indices
    indices = CartesianIndices((length(xs), length(vs)))

    # spawn threads to compute density at each chunk of the array
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            (chunk_inds,) = parentindices(chunk)
            for linear_ind in chunk_inds
                i, j = Tuple(indices[linear_ind])
                density[linear_ind] = AverageShiftedHistograms.pdf(hist, xs[i] / 1000, vs[j] * 1000)
            end
        end
    end

    # collect output
    for t in tasks
        fetch(t)
    end

    return density
end

function round_bounds(a, b; to = 5)
    divisor = 10 / to
    diff_x = b - a
    increment = exp10(floor(Int, log10(diff_x))) / divisor
    c = increment * floor(a / increment)
    d = increment * ceil(b / increment)
    return c, d
end

function plot_phase_space(dir, species, index; kwargs...)
    data = load_all_data(dir)
    plot_phase_space(data, dir, species, index; kwargs...)
end

function plot_phase_space(data, dir, species, index; time = false, interval = 1, framerate = 15, colormap = :turbo)

    GLMakie.activate!()

    analysis_dir = mkpath(joinpath(ANALYSIS_DIR, dir))

    dim = ("x", "y", "z")[index]

    fname = "phase_space_" * species * "_" * dim * ".mp4"
    anim_filename = joinpath(analysis_dir, fname)

    timer = TimerOutput()

    if (species == "e") || (species == "electrons")
        species_data = data.particles.electrons
        species_str = "Electron"
    else
        species_data = data.particles.ions
        species_str = "Ion"
    end

    @timeit timer "setup" begin
        # Get phase space extents
        vmin, vmax = extrema(@views species_data.velocity[:, index, :]) ./ 1000
        xmin, xmax = extrema(@views species_data.position[:, index, :]) .* 1000

        # Set up for nice axis bounds (v)
        vmin, vmax = round_bounds(vmin, vmax, to = 2)

        # Set up for nice axis bounds (x)
        xmin, xmax = round_bounds(xmin, xmax, to = 2)

        # Set up title title string
        title_string(time, iter) = @sprintf("%s phase space (%s)\niteration %d, time = %.2f μs", species_str, dim, iter, time * 1e6)
        titlestr = Observable(title_string(data.time[1], data.iter[1]))

        # Set up plots
        f = Figure(; fontsize = 16)

        ax = Axis(f[1,1];
            xgridcolor = :white,
            ygridcolor = :white,
            xgridwidth = 1,
            ygridwidth = 1,
            xlabel = dim * " (mm)",
            ylabel = "Velocity (km/s)",
            title = titlestr,
            xticks = WilkinsonTicks(7),
        )

        # Image resolution and pixel coordinates
        resolution = (1920, 1080) .÷ 4
        xs = LinRange(xmin, xmax, resolution[1])
        vs = LinRange(vmin, vmax, resolution[2])
        xlims!(ax, xs[1], xs[end])
        ylims!(ax, vs[1], vs[end])

        # Allocate and compute initial phase space density array
        density = zeros(resolution[1], resolution[2])
        phase_space_density!(density, xs, vs, species_data, 1, index)

        im = image!(ax, (xs[1], xs[end]), (vs[1], vs[end]), density; colormap)
        translate!(im, 0, 0, -100)
        niters = size(species_data.position, 3)

    end

    record(f, anim_filename, 1:interval:niters; framerate) do i
        @timeit timer "frame" begin
            @timeit timer "update" phase_space_density!(density, xs, vs, species_data, i, index)
            @timeit timer "render" im[3] = density
            ax.title = title_string(data.time[i], data.iter[i])
        end
    end

    if time
        show(timer)
    end

    return nothing
end

function make_phase_space_plots(dir; kwargs...)
    plot_phase_space(dir, "electrons", 1; kwargs...)
    plot_phase_space(dir, "electrons", 3; kwargs...)
    plot_phase_space(dir, "ions", 1; kwargs...)
    plot_phase_space(dir, "ions", 3; kwargs...)
end
