using StatsBase

function select_species(data, species_name)
    if (species_name in ["electrons", "e", "electron"])
        return data.particles.electrons
    elseif (species_name in ["ions", "i", "ion"])
        return data.particles.ions
    else
        error("Invalid species name chosen. Please select either 'electrons' or 'ions'.")
    end
end

function temperature_gradient(data, species, iter, index; bin_factor = 1, center_fraction = 0.5, offset_fraction = 0.15)
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
    for (i, _x) in enumerate(x_sorted)
        if _x ≥ edges[bin_id+1] || i == length(x_sorted)
            temperatures[bin_id] = mass * var(v_sorted[last:i]) / q_e
            bin_id += 1
            last = i + 1
        end
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
    #=
    T_pred = @. m * xc + b
    lower = @. T_pred - 3*std
    upper = @. T_pred + 3*std


    fig, ax, b = band(xc, lower, upper; color = (:grey, 0.2))
    plot!(ax, center_positions, temperatures)
    plot!(ax, center_positions[center_inds], temperatures[center_inds])
    lines!(ax, xc, T_pred; linestyle = :dash, linewidth = 2, color = :black)
    display(fig)
    =#
    return m, std_m
end

function all_temp_gradients(data, species, index; kwargs...)
    N = length(data.iter)

    results = [temperature_gradient(data, species, i, index; kwargs...) for i in 1:N]
    ∇T = [res[1] for res in results] ./ 1000
    σ = [res[2] for res in results] ./ 1000
    lower = @. ∇T - 3 * σ
    upper = @. ∇T + 2 * σ

    t_μs = data.time * 1e6
    fig, ax, b = band(t_μs, lower, upper; color = (:grey, 0.2))
    lines!(ax, t_μs, ∇T)
    ylims!(ax, -20, 1)
    ax.xlabel = "Time (μs)"
    ax.ylabel = "Electron temperature gradient (eV / mm)"

    fig
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
