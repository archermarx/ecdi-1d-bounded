using ImageFiltering

function heat_flux(vel, mass)
    # compute square of speed for all particles
    v² = mapslices(x -> sum(x.^2), vel, dims = 2)[:, 1]

    # get axial velocity
    vx = vel[:, 1]

    # compute heat flux in eV / m^2 s
    integrand = @. 0.5 * mass * vx * v² / q_e
    q = mean(integrand)

    return q
end

function select_species(data, species_name)
    if (species_name in ["electrons", "e", "electron"])
        return data.particles.electrons
    elseif (species_name in ["ions", "i", "ion"])
        return data.particles.ions
    else
        error("Invalid species name chosen. Please select either 'electrons' or 'ions'.")
    end
end

function temperature_gradient(data, species, iter, index; plot = true, bin_factor = 1, center_fraction = 0.5, offset_fraction = 0.15)
    species_data = select_species(data, species)

    x = species_data.position[:, 1, iter]
    v = species_data.velocity[:, index, iter]
    mass = species_data.mass

    xmin, xmax = round_bounds(extrema(x)..., to = 5)
    num_particles = length(x)
    num_bins = ceil(Int, sqrt(num_particles) * bin_factor)

    edges = LinRange(xmin, xmax, num_bins+1)

    perm = sortperm(x)
    x_sorted = x[perm]
    v_sorted = v[perm]

    bin_id = 1
    last = 1
    temperatures = zeros(num_bins)
    bin_assignments = zeros(Int, num_particles)
    for (i, _x) in enumerate(x_sorted)
        if _x ≥ edges[bin_id+1] || i == length(x_sorted)
            temperatures[bin_id] = mass * var(v_sorted[last:i-1]) / q_e
            bin_id += 1
            last = i
        end
        bin_assignments[i] = bin_id
    end

    CairoMakie.activate!()
    offset_fraction = min(offset_fraction, (1 - center_fraction)/2)
    num_offset = round(Int, offset_fraction * num_bins)
    num_center = round(Int, (1 - center_fraction) / 2 * num_bins)
    left_inds = 1:num_center+num_offset
    right_inds = num_bins-num_center+num_offset:num_bins
    center_inds = num_center+num_offset+1:num_bins-num_center+num_offset-1

    center_positions = [0.5 * (edges[i] + edges[i+1]) for i in 1:num_bins]

    xc = center_positions[center_inds]
    m, b, std, std_m, std_b = fit_line(xc, temperatures[center_inds])

    #@show [center_inds]
    vel_inds = findall(i -> i ∈ center_inds, bin_assignments)
    #@show vel_inds
    vel_center = species_data.velocity[perm[vel_inds], :, iter]

    q = heat_flux(vel_center, mass)

    if (plot)
        T_pred = @. m * xc + b
        lower = @. T_pred - 3*std
        upper = @. T_pred + 3*std

        fig, ax, b = band(xc, lower, upper; color = (:grey, 0.2))
        plot!(ax, center_positions, temperatures)
        plot!(ax, center_positions[center_inds], temperatures[center_inds])
        lines!(ax, xc, T_pred; linestyle = :dash, linewidth = 2, color = :black)
        display(fig)
    end
    return m, std_m, q
end

function all_temp_gradients(data, species = "electron"; kwargs...)
    N = length(data.iter)

    tmin, tmax = round_bounds(data.time[1], data.time[end-1]) .* 1e6

    results_x = [temperature_gradient(data, species, i, 1; plot = false, kwargs...) for i in 1:N]
    ∇T_x = [res[1] for res in results_x] ./ 1000
    σ_x = [res[2] for res in results_x] ./ 1000
    q = [res[3] for res in results_x] ./ 1e6
    lower_x = @. ∇T_x - 3 * σ_x
    upper_x = @. ∇T_x + 2 * σ_x

    results_y = [temperature_gradient(data, species, i, 3; plot = false, kwargs...) for i in 1:N]
    ∇T_y = [res[1] for res in results_y] ./ 1000
    σ_y = [res[2] for res in results_y] ./ 1000
    lower_y = @. ∇T_y - 3 * σ_y
    upper_y = @. ∇T_y + 2 * σ_y


    ker = ImageFiltering.Kernel.gaussian((5,))
    q_filtered = imfilter(q, ker)
    colors = Makie.wong_colors()

    t_μs = data.time * 1e6

    sp_name = titlecase(species)[1:end-1]
    sp_letter = species[1]

    f = Figure(;

    )

    fontsize = 17

    vmin, vmax = round_bounds(extrema(q)...)
    ax_q = Axis(
        f[1,1],
        yticklabelcolor = colors[2],
        ylabelcolor = colors[2],
        ylabel = L"%$(sp_name) heat flux (eV mm$^{-2}$ s$^{-1}$)",
        yaxisposition = :right,
        ylabelrotation = -pi/2,
        ylabelsize = fontsize
    )
    ax_∇ = Axis(
        f[1,1],
        yticklabelcolor = colors[1],
        ylabelcolor = colors[1],
        ylabel = L"%$(sp_name) temperature gradient (eV mm$^{-1}$)",
        ylabelsize = fontsize
    )
    l_q = lines!(ax_q, t_μs, q; color = (colors[2], 0.5))
    l_f = lines!(ax_q, t_μs, q_filtered; color = colors[2], linewidth = 2)

    b_x = band!(ax_∇, t_μs, lower_x, upper_x; color = (colors[1], 0.2))
    l_grad_x = lines!(ax_∇, t_μs, ∇T_x; color = colors[1])

    b_y = band!(ax_∇, t_μs, lower_y, upper_y; color = (colors[3], 0.2))
    l_grad_y = lines!(ax_∇, t_μs, ∇T_y; color = colors[3])

    ax_∇.xlabel = L"Time ($μ$s)"
    ax_∇.xlabelsize = fontsize
    xlims!(ax_q, tmin, tmax)
    xlims!(ax_∇, tmin, tmax)
    ylims!(ax_q, vmin, vmax)
    ylims!(ax_∇, vmin, vmax)
    hidespines!(ax_q)
    hidexdecorations!(ax_q)

    Legend(
        f[0, 1],
        [
            [l_grad_x, b_x],
            [l_grad_y, b_y],
            l_q, l_f
        ],
        [
            L"$\nabla T_{%$(sp_letter), x}$ ($\pm 3\sigma$)",
            L"$\nabla T_{%$(sp_letter), y}$ ($\pm 3\sigma$)",
            L"$-q_{%$(sp_letter),x}$ (raw)",
            L"$-q_{%$(sp_letter),x}$ (filtered)",
        ];
        orientation = :horizontal, patchsize = (20, 10),
        labelsize = 17, tellwidth = false
    )

    save(joinpath(data.dir, "heat_flux.png"), f, px_per_unit = 5)
    firstind = findfirst(>=(1), t_μs)

    ts = t_μs[firstind:end]
    N = length(ts)
    κs_x = q ./ ∇T_x
    κs_y = q ./ ∇T_y
    σx = std(κs_x) / sqrt(N)
    σy = std(κs_x) / sqrt(N)
    κx = mean(κs_x[firstind:end])
    κy = mean(κs_y[firstind:end])

    return κx, κy, σx, σy
end

function fit_line(x, y)
    N = length(x)
    M = hcat(ones(N), x)
    b, m = M \ y

    pred = @. m * x + b
    var_pred = sum((pred[i] - y[i])^2 for i in eachindex(y)) / (N - 2)

    var_x = var(x)
    std_m = sqrt(var_pred / ((N - 1) * var_x))
    sumsq = sum(x.^2) / N
    std_b = std_m * sqrt(sumsq)

    std_pred = sqrt(var_pred)
    return m, b, std_pred, std_m, std_b
end

n, κx, κy, σx, σy =  let
dirs = readdir("diags")
N = length(dirs)
κx = zeros(N)
κy = zeros(N)
σx = zeros(N)
σy = zeros(N)
n = zeros(N)
for (i, dir) in enumerate(dirs)
    @show dir
    data = load_all_data(dir)

    n[i] = data.particles.electrons.density
    #make_phase_space_plots(dir, interval = 2)
    κx[i], κy[i], σx[i], σy[i] = all_temp_gradients(data, "electrons")
end
n[2:end], κx[2:end], κy[2:end], σx[2:end], σy[2:end]
end

let
    perm = sortperm(n)
    n_sorted = n[perm]
    kx_sorted = κx[perm]
    ky_sorted = κy[perm]
    sx_sorted = σx[perm]
    sy_sorted = σy[perm]
    lower_x = @. kx_sorted - 3 * sx_sorted
    upper_x = @. kx_sorted + 3 * sx_sorted
    lower_y = @. ky_sorted - 3 * sy_sorted
    upper_y = @. ky_sorted + 3 * sy_sorted

    ns = exp10.(LinRange(16, 18, 100))
    linear = @. ns / ns[1] /10
    sqrtln = @. sqrt(ns / ns[1]) /10
    logln =  @. log10(ns) / log10(ns)[1] / 10
    square = @. (ns/ns[1])^2 / 10

    colors = Makie.wong_colors()

    xticks = [1e16, 1e17, 1e18], [L"$10^{16}$", L"$10^{17}$", L"$10^{18}"]
    xlims = (8e15, 1.3e18)
    ylims = (0.08, 10)
    f = Figure()
    ax = Axis(
        f[1,1];
        xlabel = L"Number density (m$^{-3}$)", xticks,
        xlabelsize = 17,
        xticksize = 17,
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(9),
        ylabel = L"Electron thermal conductivity ($k_B$ W/mm K)", xscale =  log10,
        yscale = log10
    )
    xlims!(ax, xlims)
    ylims!(ax, ylims)
    #b_x = band!(ax, n_sorted, kx_sorted, upper_x, color = (colors[1], 0.2))
    #l_x = scatterlines!(ax, n_sorted, kx_sorted, color = colors[1])
    #b_x = band!(ax, n_sorted, ky_sorted, upper_y, color = (colors[2], 0.2))
    l_y = scatterlines!(ax, n_sorted, ky_sorted, color = colors[2])

    l_square = lines!(ax, ns, square, color = :black, linestyle = :dash)
    l_linear = lines!(ax, ns, linear, color = :black, linestyle = :dash)
    l_log = lines!(ax, ns, logln, color = :black, linestyle = :dash)
    l_sqrt = lines!(ax, ns, sqrtln, color = :black, linestyle = :dash)
    #Legend(f[0, 1], [l_x, l_y], ["x-conductivity", "y-conductivity"], orientation = :horizontal)
    save(joinpath(ANALYSIS_DIR, "conductivity.png"), f)
    f
end
