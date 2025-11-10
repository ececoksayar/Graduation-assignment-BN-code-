
""" Base terminal"""

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


Revise.includet("src/Params.jl")
Revise.includet("src/QC_Utils.jl")
Revise.includet("src/EquipmentReliability.jl")
Revise.includet("src/BaseTerminalBN.jl")
Revise.includet("src/Inference.jl")

# Import all modules using module.
using .Params
using .QC_Utils
using .EquipmentReliability
using .BaseTerminalBN
using .Inference

# Set random seed
Random.seed!(1234)

# Build network components 
base_cpds = BaseTerminalBN.base_cpds()
equipment_cpds = EquipmentReliability.build_equipment_availability_cpds()

@show typeof(base_cpds)
@show typeof(equipment_cpds)

# Merge CPDs 
allcpds = Dict{Symbol,CPD}()
merge!(allcpds, base_cpds)
merge!(allcpds, equipment_cpds)

# Verify required nodes
required_equipment = [
    :QC_Age, :QC_Beta, :QC_Eta, :QC_MTBF, :QC_RepairTime, :QC_Availability,
    :YC_Age, :YC_Beta, :YC_Eta, :YC_MTBF, :YC_RepairTime, :YC_Availability,
    :HT_Age, :HT_Beta, :HT_Eta, :HT_MTBF, :HT_RepairTime, :HT_Availability
]

missing = [k for k in required_equipment if !haskey(allcpds, k)]
println("\nMissing equipment nodes: ", isempty(missing) ? "None" : missing)

# Build network and verify structure
bn = BaseTerminalBN.build_base_bn(allcpds)


#Inspect nodes via a single sample
s = rand(bn)
println("Nodes in BN: ", length(s))
println("Nodes: ", join(sort!(collect(keys(s))), ", "))



#just an example
println("\nSampling from network:")
samples = Inference.sample_network(bn, 1)  # Get 1 sample
for (i, s) in enumerate(samples)
    println("Sample $i: ", s)
end

# Check specific evidence likelihood
evidence = Dict{Symbol,Int}(:Wind => 3, :Rain => 4)
likelihood = Inference.compute_likelihood(bn, evidence)
println("Log-likelihood of severe weather: ", likelihood)


# To run trace delay inference, used for testing
# Define your evidence as a Dict, uncomment and modify as needed
forward_evidence = Dict(
    #:Wind => 3, 
    #:Rain => 1,
    #:Visibility => 1,
    #:Storage_Capacity_Level => 3,
    #:QC_Availability => 3,
    #:YC_Availability => 1,
    #:HT_Availability => 3,
    #:Terminal_Busyness => 1
)

# Call the function with your Bayesian network object `bn`
Inference.trace_delays_given(bn, forward_evidence; samples=1000000)


#checking the reliability/ availability
#Function is in the equipment reliability module
println("\nTesting Equipment Reliability:")
EquipmentReliability.test_reliability(:QC)
EquipmentReliability.test_reliability(:YC)
EquipmentReliability.test_reliability(:HT)

function collect_reliability_results(equipment_type::Symbol)
    results = NamedTuple[]
    for age in [:New, :Mid, :Old]
        beta = REL.BETA[age]
        eta = EquipmentReliability.get_eta(equipment_type)[age]
        mtbf_dist = Weibull(beta, eta)
        mtbf = mean(mtbf_dist)
        mu_mttr, sd_mttr = EquipmentReliability.get_repair_params(equipment_type)
        mttr_dist = truncated(Normal(mu_mttr, sd_mttr), 0, Inf)
        mttr = mean(mttr_dist)
        mean_availability = mtbf / (mtbf + mttr)
        thresholds  = AVAIL_THRESHOLDS[equipment_type]
        low_thresh  = thresholds[2]
        high_thresh = thresholds[3]

        margin = 0.02
        p_low = max(0, min(1, (low_thresh + margin - mean_availability) / (2 * margin)))
        p_high = max(0, min(1, (mean_availability - high_thresh + margin) / (2 * margin)))
        p_med = 1 - p_low - p_high

        push!(results, (
            Equipment = string(equipment_type),
            Age = string(age),
            Beta = beta,
            Eta = eta,
            MTBF = mtbf,
            MTTR = mttr,
            Availability = mean_availability,
            Low = p_low,
            Medium = p_med,
            High = p_high
        ))
    end
    return results
end


results = vcat(
    collect_reliability_results(:QC),
    collect_reliability_results(:YC),
    collect_reliability_results(:HT)
)
EquipmentReliability.write_reliability_latex_table(results; outfile="reliability_table.tex")

# Calls for the availability sensitivity analysis function
results_avail = Inference.run_availability_sensitivity(bn)


# Helper function to convert results Dict to DataFrame
function dict_to_df(results::Dict{String, Dict{Symbol, Float64}}, metrics::Vector{Symbol})
    scenarios = collect(keys(results))
    df = DataFrame(Scenario = scenarios)
    for m in metrics
        df[!, m] = [get(results[sc], m, NaN) for sc in scenarios]
    end
    return df
end

# Delay interaction sensitivity analysis
results_delay = Inference.run_delay_sensitivity(
    bn;
    metrics=[:Crane_Delay, :HT_Delay_Empty, :HT_Delay_Loaded, :Yard_Crane_Delay, :Total_Delay],
    save_bar=true,
    bar_metric=:Total_Delay,
    bar_filename="delay_interaction_total_delay_bar.png",
    sort_by=:Total_Delay,
    descending=true,
    include_baseline=true,
    latex_filename="delay_interaction_table.tex",
    latex_caption="Delay interaction sensitivity vs baseline.",
    latex_label="tab:delay-interaction"
)

 
# Terminal sensitivity analysis
results_terminal = Inference.run_terminal_sensitivity(
    bn;
    metrics=[:Crane_Delay, :HT_Delay_Empty, :HT_Delay_Loaded, :Yard_Crane_Delay, :Total_Delay],
    save_bar=true,
    bar_metric=:Total_Delay,
    bar_filename="terminal_impact_total_delay_bar.png",
    sort_by=:Total_Delay,
    descending=true,
    include_baseline=true,
    latex_filename="terminal_impact_table.tex",
    latex_caption="Terminal state sensitivity vs baseline.",
    latex_label="tab:terminal-impact"
)


# Weather sensitivity analysis (if you have this function)
results_weather = Inference.run_sensitivity_analysis(bn)


#df_delay = Inference.prepare_delta_data(results_delay; metrics=[:Total_Delay])
#df_delay[!, :Source] .= "Delay"


#Tornado plot
# 1. Universal baseline 
universal_baseline = Dict{Symbol,Int64}(
    :Wind => 1, :Rain => 1, :Visibility => 1,
    :Storage_Capacity_Level => 1, :Terminal_Busyness => 1,
    :QC_Availability => 1, :YC_Availability => 1, :HT_Availability => 1
)

baseline = ("Baseline", universal_baseline)


# 2. Define scenarios (only conditioned nodes, no baselines here)
weather_scenarios = [
    ("High Wind", Dict{Symbol,Int64}(:Wind => 3)),
    ("Medium Wind", Dict{Symbol,Int64}(:Wind => 2)),
    ("Heavy Rain", Dict{Symbol,Int64}(:Rain => 3)),
    ("Medium Rain", Dict{Symbol,Int64}(:Rain => 2)),
    ("Low Visibility", Dict{Symbol,Int64}(:Visibility => 3)),
    ("Medium Visibility", Dict{Symbol,Int64}(:Visibility => 2))
]

terminal_scenarios = [
    ("Medium Busyness", Dict{Symbol,Int64}(:Terminal_Busyness => 2)),
    ("High Busyness",   Dict{Symbol,Int64}(:Terminal_Busyness => 3)),
    ("Medium Storage",  Dict{Symbol,Int64}(:Storage_Capacity_Level => 2)),
    ("Low Storage",     Dict{Symbol,Int64}(:Storage_Capacity_Level => 3))
]

availability_scenarios = [
    ("Medium QC Availability", Dict{Symbol,Int64}(:QC_Availability => 2)),
    ("Low QC Availability",    Dict{Symbol,Int64}(:QC_Availability => 3)),
    ("Medium YC Availability", Dict{Symbol,Int64}(:YC_Availability => 2)),
    ("Low YC Availability",    Dict{Symbol,Int64}(:YC_Availability => 3)),
    ("Medium HT Availability", Dict{Symbol,Int64}(:HT_Availability => 2)),
    ("Low HT Availability",    Dict{Symbol,Int64}(:HT_Availability => 3))
]

# 3. Combine all scenarios with one baseline
all_scenarios = vcat([baseline], weather_scenarios, terminal_scenarios, availability_scenarios)

# 4. Compute results once
results_all = compute_ev_results(bn, all_scenarios; nodes=[:Total_Delay], N=100_000)

# 5. Prepare delta DataFrame
df_all = prepare_delta_data(results_all; metrics=[:Total_Delay], baseline_name="Baseline")

# 6. Remove baseline row (only want scenario variations)
df_all = filter(row -> row.Scenario != "Baseline", df_all)

# 7. Inspect + save
show(df_all, allcols=true)
CSV.write("tornado_data.csv", df_all)
println("Saved tornado data to tornado_data.csv")

