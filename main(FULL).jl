using Revise
using BayesNets
using Random
using Statistics: mean
using Printf 
using SpecialFunctions: gamma 
using DataFrames    
using PrettyTables
using Plots        
using StatsPlots
using CSV
using StatsBase: countmap


Revise.includet("src/Params.jl")
Revise.includet("src/QC_Utils.jl")
Revise.includet("src/EquipmentReliability.jl")
Revise.includet("src/FullTerminalBN.jl")
Revise.includet("src/MaintenanceReliability.jl")
Revise.includet("src/Inference.jl")

# Import all modules using module.
using .Params
using .QC_Utils
using .FullTerminalBN
using .MaintenanceReliability
using .Inference

# Set random seed
Random.seed!(123)

# Build base 
base = FullTerminalBN.base_cpds()

maint_all = MaintenanceReliability.build_equipment_availability_cpds()

println("Keys in maint_all: ", sort!(collect(keys(maint_all))))


# Merge and build BN
allcpds = merge(base, maint_all)

bn = FullTerminalBN.build_full_bn(allcpds)

s = rand(bn)

function get_cpd(bn, sym::Symbol)
    i = findfirst(bn.cpds) do cpd
        if hasfield(typeof(cpd), :target)
            return cpd.target == sym
        elseif hasfield(typeof(cpd), :variable)
            return cpd.variable == sym
        elseif hasfield(typeof(cpd), :var)
            return cpd.var == sym
        elseif hasfield(typeof(cpd), :node)
            return cpd.node == sym
        else
            fields = fieldnames(typeof(cpd))
            if !isempty(fields)
                first_field = getfield(cpd, fields[1])
                return first_field == sym
            end
            return false
        end
    end
    
    isnothing(i) && error("CPD for $sym not found in BN")
    return bn.cpds[i]
end

# Evaluate dependent CPDs manually
for key in [:Minor_Count, :Medium_Count, :Major_Count]
    cpd = get_cpd(bn, key)
    s[key] = rand(cpd, s)
end

# Print results 
@show s[:Minor_Policy] s[:Minor_Count]
@show s[:Medium_Policy] s[:Medium_Count]
@show s[:Major_Policy] s[:Major_Count]

#PM SCENARIO ANALYSIS to see impact on delays 
println("\n=== PM SCENARIO ANALYSIS ===")

# Define PM policy scenarios
pm_scenarios = Dict(
    "No PM" => Dict(
        :Minor_Policy => :None,
        :Medium_Policy => :None,
        :Major_Policy => :None
    ),
    "Minor PM Weekly" => Dict(
        :Minor_Policy => :Weekly,
        :Medium_Policy => :None,
        :Major_Policy => :None
    ),
    
    "Minor PM Monthly" => Dict(
        :Minor_Policy => :Monthly,
        :Medium_Policy => :None,
        :Major_Policy => :None
    ),
    "Medium PM Monthly" => Dict(
        :Minor_Policy => :None,
        :Medium_Policy => :Monthly,
        :Major_Policy => :None
    ),
    "Major PM Yearly" => Dict(
        :Minor_Policy => :None,
        :Medium_Policy => :None,
        :Major_Policy => :Yearly
    ),
    "Major PM Weekly" => Dict(
        :Minor_Policy => :None,
        :Medium_Policy => :None,
        :Major_Policy => :Weekly
    ),

    "Comprehensive PM" => Dict(
        :Minor_Policy => :Weekly,
        :Medium_Policy => :Monthly,
        :Major_Policy => :Yearly
    ),
    
)

# Analyze each PM scenario
pm_results = Dict{String, Dict{Symbol, Float64}}()

for (scenario_name, evidence) in pm_scenarios
    println("Analyzing: $scenario_name")
    
    # Sample the network with this PM policy
    samples = Inference.sample_network(bn, 5000; evidence=evidence)
    
    # Calculate mean delays
    delays = Dict(
        :Crane_Delay => mean([s[:Crane_Delay] for s in samples]),
        :Yard_Crane_Delay => mean([s[:Yard_Crane_Delay] for s in samples]),
        :HT_Delay_Empty => mean([s[:HT_Delay_Empty] for s in samples]),
        :HT_Delay_Loaded => mean([s[:HT_Delay_Loaded] for s in samples]),
        :Total_Delay => mean([s[:Total_Delay] for s in samples])
    )
    
    # Track availability improvements
    delays[:QC_Availability] = mean([s[:QC_Availability_Value] for s in samples if haskey(s, :QC_Availability_Value)])
    delays[:YC_Availability] = mean([s[:YC_Availability_Value] for s in samples if haskey(s, :YC_Availability_Value)])
    delays[:HT_Availability] = mean([s[:HT_Availability_Value] for s in samples if haskey(s, :HT_Availability_Value)])
    
    pm_results[scenario_name] = delays
end

# Display results
println("\n=== PM SCENARIO RESULTS ===")
baseline = pm_results["No PM"]
for (scenario, results) in pm_results
    println("\n$scenario:")
    println("  Total Delay: $(round(results[:Total_Delay], digits=2))")
    println("  QC Avail: $(round(results[:QC_Availability]*100, digits=1))%")
    println("  YC Avail: $(round(results[:YC_Availability]*100, digits=1))%") 
    println("  HT Avail: $(round(results[:HT_Availability]*100, digits=1))%")

    if scenario != "No PM"
        delay_improvement = ((baseline[:Total_Delay] - results[:Total_Delay]) / baseline[:Total_Delay]) * 100
        println("  Delay Reduction: $(round(delay_improvement, digits=1))%")
    end
end

# Create visualization - ONLY p1 with percentage improvement
using Plots

# Get scenario data and calculate percentage improvements vs baseline
baseline_delay = pm_results["No PM"][:Total_Delay]
scenario_data = []

for (name, results) in pm_results
    delay = results[:Total_Delay]
    if name == "No PM"
        improvement = 0.0  # Baseline is 0% improvement
    else
        improvement = ((baseline_delay - delay) / baseline_delay) * 100  # Positive = better
    end
    push!(scenario_data, (name, improvement))
end

# Sort by improvement (best to worst = highest % to lowest %)
sorted_scenarios = sort(scenario_data, by=x->x[2], rev=true)  

scenario_names_sorted = [x[1] for x in sorted_scenarios]
improvements_sorted = [x[2] for x in sorted_scenarios]

# Create shorter scenario names for academic paper
scenario_mapping = Dict(
    "No PM" => "No-PM",
    "Minor PM Weekly" => "Min-W",
    "Minor PM Monthly" => "Min-M", 
    "Medium PM Monthly" => "Med-M",
    "Major PM Yearly" => "Maj-Y",
    "Major PM Weekly" => "Maj-W",
    "Comprehensive PM" => "Comp"
)

# Create shortened names in the same order as your sorted scenarios
scenario_names_short_sorted = [scenario_mapping[name] for name in scenario_names_sorted]

p1 = bar(scenario_names_short_sorted, improvements_sorted,
         xlabel="PM Strategy",                
         ylabel="Delay Reduction (%)",
         xrotation=45,
         color=:viridis,
         legend=false,
         size=(900, 400),                    
         titlefontsize=16,                  
         guidefontsize=16,                  
         tickfontsize=16,                   
         left_margin=8Plots.mm,     
         bottom_margin=12Plots.mm,          
         top_margin=3Plots.mm,
         dpi=300)                          
         

savefig(p1, "pm_strategy_academic_compact.png")

# Display only p1
display(p1)

# Save only p1
savefig(p1, "pm_strategy_percentage_improvement.png")

# Generate LaTeX table
open("pm_strategy_table.tex", "w") do f
    write(f, "\\begin{table}[h]\n")
    write(f, "  \\centering\n")
    write(f, "  \\small\n")
    write(f, "  \\begin{tabular}{lrrrrr}\n")
    write(f, "    \\toprule\n")
    write(f, "    PM Strategy & Total Delay & QC Avail & YC Avail & HT Avail & Improvement \\\\\n")
    write(f, "    \\midrule\n")
    
    for scenario in scenario_names_sorted  # ← Fixed: use scenario_names_sorted
        results = pm_results[scenario]
        improvement = scenario == "No Maintenance" ? 0.0 : 
                     ((baseline[:Total_Delay] - results[:Total_Delay]) / baseline[:Total_Delay]) * 100
        
        write(f, "    $(replace(scenario, "_" => "\\_")) & ")
        write(f, "$(round(results[:Total_Delay], digits=2)) & ")
        write(f, "$(round(results[:QC_Availability]*100, digits=1))\\% & ")
        write(f, "$(round(results[:YC_Availability]*100, digits=1))\\% & ")
        write(f, "$(round(results[:HT_Availability]*100, digits=1))\\% & ")
        write(f, "$(round(improvement, digits=1))\\% \\\\\n")
    end
    
    write(f, "    \\bottomrule\n")
    write(f, "  \\end{tabular}\n")
    write(f, "  \\caption{PM strategy impact on terminal performance.}\n")
    write(f, "  \\label{tab:pm-strategy}\n")
    write(f, "\\end{table}\n")
end

println("\nPM analysis completed!")
println("Results saved to: pm_strategy_comparison.png and pm_strategy_table.tex")

# what I use for my results
portfolios = [
    ("No_PM", Dict(:Minor_Policy=>:None, :Medium_Policy=>:None, :Major_Policy=>:None)),
    ("Minor_Weekly", Dict(:Minor_Policy=>:Weekly, :Medium_Policy=>:None, :Major_Policy=>:None)),
    ("Minor_Monthly", Dict(:Minor_Policy=>:Monthly, :Medium_Policy=>:None, :Major_Policy=>:None)),
    ("Minor_Yearly", Dict(:Minor_Policy=>:Yearly, :Medium_Policy=>:None, :Major_Policy=>:None)),
    ("Medium_Weekly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:Weekly, :Major_Policy=>:None)),
    ("Medium_Monthly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:Monthly, :Major_Policy=>:None)),
    ("Medium_Yearly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:Yearly, :Major_Policy=>:None)),
    ("Major_Weekly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:None, :Major_Policy=>:Weekly)),
    ("Major_Monthly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:None, :Major_Policy=>:Monthly)),
    ("Major_Yearly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:None, :Major_Policy=>:Yearly)),
    
    # Two-type combinations
    ("Minor_Weekly+Major_Yearly", Dict(:Minor_Policy=>:Weekly, :Medium_Policy=>:None, :Major_Policy=>:Yearly)),
    ("Minor_Monthly+Major_Yearly", Dict(:Minor_Policy=>:Monthly, :Medium_Policy=>:None, :Major_Policy=>:Yearly)),
    ("Medium_Monthly+Major_Yearly", Dict(:Minor_Policy=>:None, :Medium_Policy=>:Monthly, :Major_Policy=>:Yearly)),
    ("Minor_Weekly+Medium_Monthly", Dict(:Minor_Policy=>:Weekly, :Medium_Policy=>:Monthly, :Major_Policy=>:None)),
    
    # Comprehensive combinations
    ("Minor_Weekly+Medium_Monthly+Major_Yearly", Dict(:Minor_Policy=>:Weekly, :Medium_Policy=>:Monthly, :Major_Policy=>:Yearly)),
]

function safe_mean(v)
    vals = Float64[]
    for x in v
        if x isa Number
            push!(vals, Float64(x))
        elseif x isa Distribution
            try
                push!(vals, Float64(mean(x)))
            catch
            end
        end
    end
    isempty(vals) ? NaN : mean(vals)
end


function evaluate_portfolio(bn, name::String, evidence::Dict; N=5000)
    samples = Inference.sample_network(bn, N; evidence=evidence)
    m = k -> begin
        vals = [s[k] for s in samples if haskey(s,k) && s[k] !== missing && isfinite(s[k])]
        isempty(vals) ? NaN : mean(vals)
    end

    e_eta  = m(:E_eta)
    e_beta = m(:E_beta)
    mtbf   = m(:HT_MTBF)
    mttr   = m(:HT_RepairTime)
    avail  = m(:HT_Availability_Value)

    # --- downtime breakdown ---
    if !isfinite(avail) || !isfinite(mttr)
        @warn "Skipping downtime for $(name): avail=$(avail), mttr=$(mttr)"
        pm_hours = NaN; fail_downtime = NaN; total_downtime = NaN; failures_per_year = NaN
    else
        T = 8760.0
        eq = :HT
        nmin = PM_COUNT[evidence[:Minor_Policy]]
        nmed = PM_COUNT[evidence[:Medium_Policy]]
        nmaj = PM_COUNT[evidence[:Major_Policy]]
        dur  = PM_DUR_EQ[eq]
        pm_hours = nmin*dur[:Minor] + nmed*dur[:Medium] + nmaj*dur[:Major]
        fail_downtime = T*(1 - avail) - pm_hours
        fail_downtime = max(fail_downtime, 0.0)
        failures_per_year = fail_downtime / mttr
        total_downtime = pm_hours + fail_downtime
    end

    return (
        Name = name,
        E_eta = e_eta,
        E_beta = e_beta,
        MTBF = mtbf,
        MTTR = mttr,
        Avail = avail,
        PM_hours = pm_hours,
        Fail_hours = fail_downtime,
        Total_downtime = total_downtime,
        Failures_per_year = failures_per_year
    )
end



results = [evaluate_portfolio(bn, name, ev) for (name, ev) in portfolios]

println("\n=== Maintenance Portfolio Results ===")
for r in results
    println("Portfolio: ", r.Name)
    println("  E_eta:   ", round(r.E_eta, digits=3))
    println("  E_beta:  ", round(r.E_beta, digits=3))
    println("  MTBF:    ", round(r.MTBF, digits=3))
    println("  MTTR:    ", round(r.MTTR, digits=3))
    println("  Avail:   ", round(r.Avail, digits=4))
    println("  PM_hours: ", round(r.PM_hours, digits=2))
    println("  Fail_hours: ", round(r.Fail_hours, digits=2))
    println("  Total_downtime: ", round(r.Total_downtime, digits=2))
    println("  Failures_per_year: ", round(r.Failures_per_year, digits=3))
    println()
end

using DataFrames, CSV
df = DataFrame(results)
CSV.write("downtime_breakdown_HT.csv", df)

results = [evaluate_portfolio(bn, name, ev) for (name, ev) in portfolios]

println("\n=== Maintenance Portfolio Results ===")
println("┌────────────────────────────┬────────┬────────┬────────┬────────┬────────┐")
println("│ Portfolio                  │ E_eta  │ E_beta │  MTBF  │  MTTR  │ Avail  │")
println("├────────────────────────────┼────────┼────────┼────────┼────────┼────────┤")

for r in results
    @printf("│ %-26s │ %6.3f │ %6.3f │ %6.2f │ %6.2f │ %6.3f │\n",
        r.Name, r.E_eta, r.E_beta, r.MTBF, r.MTTR, r.Avail)
end

println("└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘")



println("\n=== Operator SENSITIVITY ANALYSIS ===")
results_strike = Inference.run_strike_sensitivity(bn;
    metrics=[:Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded, :Total_Delay],
    save_bar=true,
    bar_metric=:Total_Delay,
    bar_filename="strike_impact_total_delay_bar.png",
    sort_by=:Total_Delay,
    descending=true,
    include_baseline=true,
    latex_filename="strike_impact_table.tex",
    latex_caption="Strike impact vs baseline.",
    latex_label="tab:strike-impact"
)

