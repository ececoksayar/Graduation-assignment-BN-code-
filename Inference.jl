module Inference

import BayesNets: BayesNet, rand, logpdf, infer, LikelihoodWeightedSampler
import Statistics: mean

using BayesNets
using Printf
using StatsBase: countmap
using DataFrames
using Plots         
using StatsPlots: @df, groupedbar  
using DataFramesMeta


function sample_network(bn::BayesNet, N::Int; evidence=Dict())
    # unconditional sampling
    all_samples = [rand(bn) for _ in 1:max(N*2, 1000)]  # oversample a bit

    if isempty(evidence)
        return all_samples[1:N]
    else
        filtered = filter(s -> all(get(s,k,nothing) == v for (k,v) in evidence), all_samples)
        if length(filtered) < N
            # if too few matches, keep drawing until we have enough
            while length(filtered) < N
                s = rand(bn)
                all(get(s,k,nothing) == v for (k,v) in evidence) && push!(filtered, s)
            end
        end
        return filtered[1:N]
    end
end


# Monte Carlo likelihood of evidence: sample full states and count matches.
function compute_likelihood(bn::BayesNet, evidence::Dict{Symbol,Int}; num_samples::Int=10000)
    matches = 0
    for _ in 1:num_samples
        s = rand(bn)
        if all(s[k] == v for (k,v) in evidence)
            matches += 1
        end
    end
    prob = max(matches/num_samples, 1e-10)   # avoid log(0)
    return log(prob)
end

function query_probability(bn::BayesNet, query::Symbol, evidence::Dict{Symbol,Int}=Dict(); N::Int=10000)
    samples = sample_network(bn, N)
    matching = filter(s -> all(haskey(s, k) && s[k] == v for (k,v) in evidence), samples)

    # count query values
    counts = Dict{Int,Int}()
    for s in matching
        val = s[query]
        counts[val] = get(counts, val, 0) + 1
    end
    total = sum(values(counts))
    total == 0 && return Dict{Int,Float64}(), 0
    return Dict(k => v/total for (k,v) in counts), length(matching)
end 

function query_probability(bn::BayesNet, query::Symbol; N::Int=10000)
    return query_probability(bn, query, Dict{Symbol,Int}(); N=N)
end 


function trace_delays_given(bn, evidence::Dict; samples=1000_000)
    # Initialize counts
    count = Dict(
        :Crane_Delay => zeros(3),
        :HT_Delay_Loaded => zeros(3),
        :Yard_Crane_Delay => zeros(3),
        :Total_Delay => zeros(3)
    )

    filtered = 0

    for _ in 1:samples
        s = rand(bn)
        if all(s[k] == v for (k, v) in evidence)
            for node in keys(count)
                count[node][s[node]] += 1
            end
            filtered += 1
        end
    end
    
    println("Number of matching samples: $filtered out of $samples") 

    if filtered == 0
        println("No matching samples found.")
        return
    end

    println("📊 Conditional Delay Distributions given $(evidence):\n")
    for (node, counts) in count
        probs = counts ./ filtered
        println("→ $node:")
        println("   Low:    $(round(probs[1], digits=3))")
        println("   Medium: $(round(probs[2], digits=3))")
        println("   High:   $(round(probs[3], digits=3))\n")
    end
end


#weather 
function run_sensitivity_analysis(bn::BayesNet)
    println("\n=== Structured Sensitivity Analysis ===")
    println("\n1. Weather Impact Analysis")

    # scenarios
    weather_scenarios = [
        ("Baseline",       Dict{Symbol,Int64}(:Wind => 1, :Rain => 1, :Visibility => 1)),
        ("Light Rain",     Dict{Symbol,Int64}(:Rain => 2)),
        ("Moderate Rain",  Dict{Symbol,Int64}(:Rain => 3)),
        ("Heavy Rain",     Dict{Symbol,Int64}(:Rain => 4)),
        ("Moderate Wind",  Dict{Symbol,Int64}(:Wind => 2)),
        ("Strong Wind",    Dict{Symbol,Int64}(:Wind => 3)),
        ("Moderate Visibility", Dict{Symbol,Int64}(:Visibility => 2)),
        ("Low Visibility", Dict{Symbol,Int64}(:Visibility => 3)),
        ("Severe Weather", Dict{Symbol,Int64}(:Wind => 3, :Rain => 4))
    ]

    # helper: expected value of a distribution (robust to empty)
    ev = probs -> isempty(probs) ? NaN : sum(k * v for (k,v) in probs)

    # compute expected values for each node under each scenario
    results_weather = Dict{String, Dict{Symbol, Float64}}()
    for (name, evidence) in weather_scenarios
        results_weather[name] = Dict{Symbol, Float64}()
        for (i, node) in enumerate((:Efficiency, :Crane_Delay, :Total_Delay))
            probs, nmatch = query_probability(bn, node, evidence; N=100_000)
            results_weather[name][node] = ev(probs)
            if i == 1  # Only print once per scenario
            println("Scenario '$name': $nmatch out of 100000 samples matched the evidence $evidence")
            end
        end
    end

    # output: bar chart + console + latex
    print_sensitivity_results(results_weather;
                              save_bar=true,
                              include_baseline_in_bars=false,
                              bars_sort_by=:Total_Delay,
                              bars_descending=true,
                              table_sort_by=:TotalDelay,
                              table_descending=true,
                              latex_filename="weather_impact_table.tex",
                              latex_caption="Weather impact vs. baseline (expected values).",
                              latex_label="tab:weather-impact")

    return results_weather
end

#compute expected values
function compute_ev_results(bn::BayesNet,
                            scenarios::Vector{Tuple{String,Dict{Symbol,Int}}};
                            nodes::Vector{Symbol},
                            N::Int=100_000)
    ev = probs -> isempty(probs) ? NaN : sum(k * v for (k,v) in probs)

    out = Dict{String, Dict{Symbol,Float64}}()
    for (name, evidence) in scenarios
        out[name] = Dict{Symbol, Float64}()
        #for node in nodes
            #probs = query_probability(bn, node, evidence; N=N)
            #out[name][node] = ev(probs)
        #end
        for node in nodes
            probs, nmatch = query_probability(bn, node, evidence; N=N)
            out[name][node] = ev(probs)

            if name == "Strike" && node == :Total_Delay
                println("DEBUG: Strike scenario Total_Delay distribution: $probs")
                println("DEBUG: Strike scenario Total_Delay expected value: $(out[name][node])")
            end

            if node == nodes[1]  # Only print once per scenario
                println("Scenario '$name': $nmatch out of $N samples matched the evidence $evidence")
            end
        end
    end
    return out
end

#difference than the baseline
function prepare_delta_data(results::Dict{String,Dict{Symbol,Float64}};
                            metrics::Vector{Symbol},
                            baseline_name::AbstractString="Baseline",
                            sort_by::Union{Symbol,Nothing}=nothing,
                            descending::Bool=true,
                            include_baseline::Bool=true)

    haskey(results, baseline_name) || error("Missing baseline \"$baseline_name\" in results.")

    # Baseline values per metric (NaN if missing)
    base = results[baseline_name]
    base_vals = Dict(m => get(base, m, NaN) for m in metrics)

    # Safe percentage change
    @inline safe_change(p::Float64, b::Float64) =
        b ≈ 0.0 ? (iszero(p) ? 0.0 : NaN) : ((p / b) - 1.0) * 100.0

    # Scenario order: baseline first (optional), then alphabetical
    names_sorted = sort(collect(keys(results)))
    scenario_names = include_baseline ?
        vcat([baseline_name], filter(!=(baseline_name), names_sorted)) :
        filter(!=(baseline_name), names_sorted)

    # Build the columns
    scenarios = String[]
    metric_cols = Dict{Symbol, Vector{Float64}}(m => Float64[] for m in metrics)

    for s in scenario_names
        push!(scenarios, s)
        res = results[s]
        for m in metrics
            val = get(res, m, NaN)
            push!(metric_cols[m], safe_change(val, base_vals[m]))
        end
    end

    # Assemble DataFrame
    df = DataFrame(Scenario = scenarios)
    for m in metrics
        df[!, m] = metric_cols[m]
    end

    # Sorting
    sort_key::Symbol = isnothing(sort_by) ? (isempty(metrics) ? :Scenario : metrics[1]) : sort_by
    if sort_key === :Scenario
        sort!(df, :Scenario)  # alphabetical
    elseif sort_key in metrics
        sort!(df, sort_key, rev=descending)
    else
        @warn "Unknown sort_by = $sort_key; sorting by first metric $(metrics[1])."
        sort!(df, metrics[1], rev=descending)
    end

    return df
end


function print_delta_table(df::DataFrame; metric_labels=Dict{Symbol,String}(), width::Int=64)
    cols = names(df)
    metrics = Symbol.(cols[2:end])
    labels = [get(metric_labels, m, String(m)) for m in metrics]

    println("\n=== Sensitivity Analysis Results ===")
    # Print header
    print(rpad("Scenario", 15))
    for L in labels
        print(" | ", rpad(L, 15))
    end
    println()
    println(repeat('-', width))

    # Print rows
    for r in eachrow(df)
        print(rpad(r.Scenario, 15))
        for m in metrics
            val = r[m]
            s = isfinite(val) ? @sprintf("%+8.1f%%", val) : "      —"
            print(" | ", rpad(s, 15))
        end
        println()
    end
    nothing
end


#helper function for bar plot function
function _resolve_col(df::DataFrame, col)
    cols = names(df)
    # exact match
    if col in cols
        return col
    end
    # try as String
    if String(col) in String.(cols)
        return Symbol(String(col))
    end
    # normalize (strip underscores, lowercase)
    norm(s) = replace(lowercase(String(s)), "_" => "")
    target = norm(col)
    for c in cols
        if norm(c) == target
            return c
        end
    end
    error("DataFrame must have a $col column (tried to resolve, available: $(cols))")
end

#bar plot
function create_bar_plot_generic(df::DataFrame;
                                 metrics=[:Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded, :Total_Delay],
                                 title::AbstractString="Impact Analysis",
                                 outfile::AbstractString="impact_bar.png",
                                 bar_position=:dodge)

    # Resolve the Scenario column robustly (handles :Scenario, "Scenario", :scenario, etc.)
    scenario_col = _resolve_col(df, :Scenario)

    # Resolve metric columns robustly
    metric_syms = metrics isa Symbol ? [metrics] : Vector{Symbol}(metrics)
    metric_syms = [_resolve_col(df, m) for m in metric_syms]

    # X labels and Y series
    x = string.(df[!, scenario_col])                  # ensure strings for x tick labels
    Y = [Float64.(df[!, m]) for m in metric_syms]     # numeric vectors for each metric

    # replace NaNs with 0 to avoid plotting issues
    for y in Y
        for i in eachindex(y)
            if isnan(y[i]); y[i] = 0.0; end
        end
    end

    if length(metric_syms) == 1
        p = bar(x, Y[1];
                label=false,
                title=title,
                xlabel="Scenario",
                ylabel="% Change vs Baseline",
                xrotation=45,
                size=(900,520),
                bar_position=bar_position)
    else
        p = bar(x, reduce(hcat, Y);
                #bar_position=:dodge,
                bar_position=bar_position, 
                label=String.(metric_syms),
                title=title,
                xlabel="Scenario",
                ylabel="% Change vs Baseline",
                xrotation=45,
                legend=:outertopright,
                size=(1000,560))
    end

    savefig(p, outfile)
    println("Bar plot saved as '$outfile'")
end


# helper function to turn numeric % into strings before LaTeX
function format_percent_df(df::DataFrame)
    cols = names(df)
    out = DataFrame(Scenario = df.Scenario)
    for c in cols[2:end]
        out[!, String(c)] = [isfinite(x) ? @sprintf("%+.1f%%", x) : "—" for x in df[!, c]]
    end
    return out
end

function write_latex_table(df::DataFrame, path::AbstractString;
                           caption::AbstractString="Weather impact vs. baseline (expected values).",
                           label::AbstractString="tab:weather-impact",
                           use_booktabs::Bool=true)

    esc(s) = replace(string(s), "%" => "\\%", "_" => "\\_", "&" => "\\&")

    open(path, "w") do io
        println(io, "\\begin{table}[h]")
        println(io, "  \\centering")
        println(io, "  \\small")
        cols = "l" * "r"^(ncol(df)-1)
        println(io, "  \\begin{tabular}{$cols}")
        println(io, use_booktabs ? "    \\toprule" : "    \\hline")

        header = join(esc.(names(df)), " & ")
        println(io, "    $header \\\\")
        println(io, use_booktabs ? "    \\midrule" : "    \\hline")

        for i in 1:nrow(df)
            row = join([esc(df[i, j]) for j in 1:ncol(df)], " & ")
            println(io, "    $row \\\\")
        end

        println(io, use_booktabs ? "    \\bottomrule" : "    \\hline")
        println(io, "  \\end{tabular}")
        println(io, "  \\caption{$(esc(caption))}")
        println(io, "  \\label{$(esc(label))}")
        println(io, "\\end{table}")
    end
    println("LaTeX table written to '$path'")
end

# run sensitivities, set the scenarios here
function run_terminal_sensitivity(bn::BayesNet;
                                  metrics::Vector{Symbol} = [:Crane_Delay, :HT_Delay_Empty, :HT_Delay_Loaded, :Yard_Crane_Delay, :Total_Delay],
                                  save_bar::Bool = true,
                                  bar_metric::Symbol = :Total_Delay,
                                  bar_filename::String = "terminal_impact_total_delay_bar.png",
                                  sort_by::Symbol = :Total_Delay,
                                  descending::Bool = true,
                                  include_baseline::Bool = true,
                                  latex_filename::Union{Nothing,String} = "terminal_impact_table.tex",
                                  latex_caption::AbstractString = "Terminal state sensitivity vs baseline.",
                                  latex_label::AbstractString = "tab:terminal-impact")

    println("\n2) Terminal State Sensitivity")

    #=scenarios
    terminal_scenarios = [
        ("Baseline",      Dict{Symbol,Int64}(:Terminal_Busyness=>1, :Storage_Capacity_Level=>1)),
        ("Medium Busyness", Dict{Symbol,Int64}(:Terminal_Busyness=>2)),
        ("High Busyness", Dict{Symbol,Int64}(:Terminal_Busyness=>3)),
        ("Medium Storage Availability", Dict{Symbol,Int64}(:Storage_Capacity_Level=>2)),
        ("Low Storage Availability",   Dict{Symbol,Int64}(:Storage_Capacity_Level=>3)),
        ("Worst Case",    Dict{Symbol,Int64}(:Terminal_Busyness=>3, :Storage_Capacity_Level=>3)),
    ]=#

    terminal_scenarios = [
    ("Baseline",      Dict{Symbol,Int64}(:Terminal_Busyness=>1, :Storage_Capacity_Level=>1)),
    ("Medium Busyness", Dict{Symbol,Int64}(:Terminal_Busyness=>2, :Storage_Capacity_Level=>1)),
    ("High Busyness",   Dict{Symbol,Int64}(:Terminal_Busyness=>3, :Storage_Capacity_Level=>1)),
    ("Medium Storage Availability", Dict{Symbol,Int64}(:Terminal_Busyness=>1, :Storage_Capacity_Level=>2)),
    ("Low Storage Availability",   Dict{Symbol,Int64}(:Terminal_Busyness=>1, :Storage_Capacity_Level=>3)),
    ("Worst Case",    Dict{Symbol,Int64}(:Terminal_Busyness=>3, :Storage_Capacity_Level=>3)),
    ]


    # pretty labels for console print
    metric_labels = Dict(
        :Crane_Delay      => "QC Delay",
        :HT_Delay_Empty   => "HT Delay (Empty)",
        :HT_Delay_Loaded   => "HT Delay (Loaded)",
        :Yard_Crane_Delay  => "YC Delay",
        :Total_Delay       => "Total Delay",
    )

    # 1) EVs
    results = compute_ev_results(bn, terminal_scenarios; nodes=metrics, N=100_000)

    # 2) %Δ vs baseline (DataFrame: :Scenario + metric columns)
    df = prepare_delta_data(results;
                            metrics=metrics,
                            baseline_name="Baseline",
                            sort_by=sort_by,
                            descending=descending,
                            include_baseline=include_baseline)

    # 3) Console table
    print_delta_table(df; metric_labels=metric_labels)

    # 4) Bar chart for one metric
    if save_bar
        create_bar_plot_generic(df;
            metrics=bar_metric,
            title="Terminal State Impact — $(String(bar_metric))",
            outfile=bar_filename)
    end

    # 5) Optional LaTeX
    if latex_filename !== nothing
        df_disp = format_percent_df(df)
        write_latex_table(df_disp, latex_filename;
                          caption=latex_caption,
                          label=latex_label,
                          use_booktabs=true)
    end

    return results
end


function run_availability_sensitivity(bn::BayesNet;
                                      metrics::Vector{Symbol} = [:Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded, :HT_Delay_Empty, :Total_Delay],
                                      save_bar::Bool = true,
                                      bar_metric::Symbol = :Total_Delay,
                                      bar_filename::String = "availability_impact_total_delay_bar.png",
                                      sort_by::Symbol = :Total_Delay,
                                      descending::Bool = true,
                                      include_baseline::Bool = true,
                                      latex_filename::Union{Nothing,String} = "availability_impact_table.tex",
                                      latex_caption::AbstractString = "Availability sensitivity vs baseline.",
                                      latex_label::AbstractString = "tab:availability-impact")

    println("\n4) Equipment Availability Sensitivity")

    #= Define scenarios (adjust node names and values as needed)
    availability_scenarios = [
        ("Baseline", Dict{Symbol,Int64}(:QC_Availability=>1, :YC_Availability=>1, :HT_Availability=>1)),
        ("Medium QC Availability", Dict{Symbol,Int64}(:QC_Availability=>2)),
        ("Low QC Availability", Dict{Symbol,Int64}(:QC_Availability=>3)),
        ("Medium YC Availability", Dict{Symbol,Int64}(:YC_Availability=>2)),
        ("Low YC Availability", Dict{Symbol,Int64}(:YC_Availability=>3)),
        ("Medium HT Availability", Dict{Symbol,Int64}(:HT_Availability=>2)),
        ("Low HT Availability", Dict{Symbol,Int64}(:HT_Availability=>3)),

    ] =#

    availability_scenarios = [
    ("Baseline", Dict(:QC_Availability=>1, :YC_Availability=>1, :HT_Availability=>1)),
    ("Medium QC Availability", Dict(:QC_Availability=>2, :YC_Availability=>1, :HT_Availability=>1)),
    ("Low QC Availability", Dict(:QC_Availability=>3, :YC_Availability=>1, :HT_Availability=>1)),
    ("Medium YC Availability", Dict(:QC_Availability=>1, :YC_Availability=>2, :HT_Availability=>1)),
    ("Low YC Availability", Dict(:QC_Availability=>1, :YC_Availability=>3, :HT_Availability=>1)),
    ("Medium HT Availability", Dict(:QC_Availability=>1, :YC_Availability=>1, :HT_Availability=>2)),
    ("Low HT Availability", Dict(:QC_Availability=>1, :YC_Availability=>1, :HT_Availability=>3)),
    ]


    # Compute expected values
    results = compute_ev_results(bn, availability_scenarios; nodes=metrics, N=100_000)

    # %Δ vs baseline
    df = prepare_delta_data(results;
                            metrics=metrics,
                            baseline_name="Baseline",
                            sort_by=sort_by,
                            descending=descending,
                            include_baseline=include_baseline)

    # Console table
    print_delta_table(df)

    # Bar chart
    if save_bar
        # 1. Plot Total Delay for all scenarios
        create_bar_plot_generic(df;
            metrics=:Total_Delay,
            title="Availability Impact — Total Delay",
            outfile="availability_impact_total_delay_bar.png")
    
        # 2. Grouped bar chart for all subsystem delays (excluding baseline)
        df_sub = filter(row -> row.Scenario != "Baseline", df)
        create_bar_plot_generic(df_sub;
            metrics=[:Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded],
            title="Availability Impact — Subsystem Delays",
            outfile="availability_impact_subsystem_delays_bar.png",
            bar_position=:dodge
        )
    end
   

    # LaTeX table
    if latex_filename !== nothing
        df_disp = format_percent_df(df)
        write_latex_table(df_disp, latex_filename;
                          caption=latex_caption,
                          label=latex_label,
                          use_booktabs=true)
    end

    return results
end

function run_delay_sensitivity(bn::BayesNet;
                               metrics::Vector{Symbol} = [:Crane_Delay, :HT_Delay_Empty, :HT_Delay_Loaded, :Yard_Crane_Delay, :Total_Delay],
                               save_bar::Bool = true,
                               bar_metric::Symbol = :Total_Delay,
                               bar_filename::String = "delay_interaction_total_delay_bar.png",
                               sort_by::Symbol = :Total_Delay,
                               descending::Bool = true,
                               include_baseline::Bool = true,
                               latex_filename::Union{Nothing,String} = "delay_interaction_table.tex",
                               latex_caption::AbstractString = "Delay interaction sensitivity vs baseline.",
                               latex_label::AbstractString = "tab:delay-interaction")

    println("\n3) Delay Interaction Sensitivity")

    #scenarios
    delay_scenarios = [
        ("Baseline",       Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
        ("High YC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>3)),
        ("High HT Delay (Empty)",  Dict{Symbol,Int64}(:HT_Delay_Empty=>3)),
        ("High HT Delay (Loaded)",  Dict{Symbol,Int64}(:HT_Delay_Loaded=>3)),
        ("High QC Delay",  Dict{Symbol,Int64}(:Crane_Delay=>3)),
        ("Medium YC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>2)),
        ("Medium HT Delay (Empty)",  Dict{Symbol,Int64}(:HT_Delay_Empty=>2)),
        ("Medium HT Delay (Loaded)",  Dict{Symbol,Int64}(:HT_Delay_Loaded=>2)),
        ("Medium QC Delay",  Dict{Symbol,Int64}(:Crane_Delay=>2)),
        ("All High Delay", Dict{Symbol,Int64}(:Yard_Crane_Delay=>3, :HT_Delay_Empty=>3, :Crane_Delay=>3)),
    ]

    #=delay_scenarios = [
    ("Baseline",       Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
    ("High YC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>3, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
    ("High HT Delay (Empty)",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>3, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
    ("High HT Delay (Loaded)", Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>3, :Crane_Delay=>1)),
    ("High QC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>3)),
    ("Medium YC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>2, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
    ("Medium HT Delay (Empty)",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>2, :HT_Delay_Loaded=>1, :Crane_Delay=>1)),
    ("Medium HT Delay (Loaded)", Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>2, :Crane_Delay=>1)),
    ("Medium QC Delay",  Dict{Symbol,Int64}(:Yard_Crane_Delay=>1, :HT_Delay_Empty=>1, :HT_Delay_Loaded=>1, :Crane_Delay=>2)),
    ("All High Delay", Dict{Symbol,Int64}(:Yard_Crane_Delay=>3, :HT_Delay_Empty=>3, :HT_Delay_Loaded=>3, :Crane_Delay=>3)),
    ]   =#


    # pretty labels for console print
    metric_labels = Dict(
        :Crane_Delay       => "QC Delay",
        :HT_Delay_Empty    => "HT Delay (Empty)",
        :HT_Delay_Loaded    => "HT Delay (Loaded)",
        :QC_Waiting_for_HT  => "QC Waiting for HT",
        :Total_Delay        => "Total Delay",
    )

    # 1) EVs
    results = compute_ev_results(bn, delay_scenarios; nodes=metrics, N=100_000)

    # 2) %Δ vs baseline
    df = prepare_delta_data(results;
                            metrics=metrics,
                            baseline_name="Baseline",
                            sort_by=sort_by,
                            descending=descending,
                            include_baseline=include_baseline)

    # 3) Console table
    print_delta_table(df; metric_labels=metric_labels)

    # 4) Bar chart for one metric
    if save_bar
        create_bar_plot_generic(df;
            metrics=bar_metric,
            title="Delay Interaction — $(String(bar_metric))",
            outfile=bar_filename)
    end

    # 5) Optional LaTeX
    if latex_filename !== nothing
        df_disp = format_percent_df(df)
        write_latex_table(df_disp, latex_filename;
                          caption=latex_caption,
                          label=latex_label,
                          use_booktabs=true)
    end

    return results
end


#operator

function run_strike_sensitivity(bn::BayesNet;
                                metrics::Vector{Symbol} = [:Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded, :Total_Delay],
                                save_bar::Bool = true,
                                bar_metric::Symbol = :Total_Delay,
                                bar_filename::String = "strike_impact_total_delay_bar.png",
                                sort_by::Symbol = :Total_Delay,
                                descending::Bool = true,
                                include_baseline::Bool = true,
                                latex_filename::Union{Nothing,String} = "strike_impact_table.tex",
                                latex_caption::AbstractString = "Strike impact vs baseline.",
                                latex_label::AbstractString = "tab:strike-impact")

    println("\n5) Strike Impact Sensitivity")

    # Define scenarios 
    strike_scenarios = [
        ("Baseline", Dict(:Strike=>1, :Shift=>1)),
        ("Night", Dict(:Strike=>1, :Shift=>2)),
        ("Strike", Dict(:Strike=>2, :Shift=>1)),
    ] #, :Terminal_Busyness=>1, :Wind=>1, :Rain=>1, :Visibility=>1, :Storage_Capacity_Level=>1

    println("DEBUG: Strike scenario evidence: ", strike_scenarios[2][2])

    # Compute expected values (same as other analyses)
    results = compute_ev_results(bn, strike_scenarios; nodes=metrics, N=100_000)

    # %Δ vs baseline (same format as other analyses)
    df = prepare_delta_data(results;
                            metrics=metrics,
                            baseline_name="Baseline",
                            sort_by=sort_by,
                            descending=descending,
                            include_baseline=include_baseline)

    # Console table (same format)
    print_delta_table(df)

    # Bar chart (same format)
    if save_bar
        create_bar_plot_generic(df;
            metrics=bar_metric,
            title="Strike Impact — $(String(bar_metric))",
            outfile=bar_filename)
    end

    # LaTeX table (same format)
    if latex_filename !== nothing
        df_disp = format_percent_df(df)
        write_latex_table(df_disp, latex_filename;
                          caption=latex_caption,
                          label=latex_label,
                          use_booktabs=true)
    end

    return results
end

export sample_network, compute_likelihood, query_probability, trace_delays_given, 
       run_sensitivity_analysis, compute_ev_results, prepare_delta_data, print_delta_table, 
       create_bar_plot_generic, format_percent_df, write_latex_table, 
       run_terminal_sensitivity, run_availability_sensitivity, run_delay_sensitivity, 
       run_strike_sensitivity
end