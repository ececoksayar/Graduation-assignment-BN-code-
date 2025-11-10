module EquipmentReliability

using BayesNets, Distributions
using SpecialFunctions: gamma 
using Printf

import ..Params: REL, AVAIL_THRESHOLDS

"""
Get ETA values for specific equipment type
"""
function get_eta(equipment_type::Symbol)
    if equipment_type == :QC
        return REL.ETA_QC
    elseif equipment_type == :YC
        return REL.ETA_YC
    else  # HT
        return REL.ETA_HT
    end
end

"""
Get MTTR parameters for specific equipment type
"""
function get_repair_params(equipment_type::Symbol)
    mu = REL.MU_MTTR[equipment_type]
    sd = REL.SD_MTTR[equipment_type]
    return mu, sd
end


"""
Build equipment availability CPDs for all equipment types
"""

function build_equipment_availability_cpds()
    d = Dict{Symbol,CPD}()

    # Build CPDs for each equipment type
    for eq in [:QC, :YC, :HT]
        # Age state
        d[Symbol(eq, "_Age")] = StaticCPD(
            Symbol(eq, "_Age"),
            NamedCategorical([:New, :Mid, :Old], [0.2, 0.6, 0.2])
        )

        # Beta parameter
        d[Symbol(eq, "_Beta")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq, "_Beta"),
            [Symbol(eq, "_Age")],
            a -> Distributions.Dirac(REL.BETA[a[Symbol(eq, "_Age")]])
        )

        # Eta parameter 
        d[Symbol(eq, "_Eta")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq, "_Eta"),
            [Symbol(eq, "_Age")],
            a -> Distributions.Dirac(get_eta(eq)[a[Symbol(eq, "_Age")]])
        )

        # Repair time - equipment specific
        mu_mttr, sd_mttr = get_repair_params(eq)
        d[Symbol(eq, "_RepairTime")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq, "_RepairTime"),
            Symbol[],
            _ -> truncated(Normal(mu_mttr, sd_mttr), 0, Inf)
        )

        # MTBF 
        d[Symbol(eq, "_MTBF")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq, "_MTBF"),
            [Symbol(eq, "_Beta"), Symbol(eq, "_Eta")],
            a -> begin
                β = a[Symbol(eq, "_Beta")]  
                η = a[Symbol(eq, "_Eta")]   
                
                # Calculate expected value of Weibull distribution
                mtbf_expected = η * gamma(1 + 1/β)
                
                Distributions.Dirac(mtbf_expected)  # Return the expected value as a fixed value
            end
        )
        
        # Availability
        d[Symbol(eq, "_Availability")] = FunctionalCPD{Categorical}(
            Symbol(eq, "_Availability"),
            [Symbol(eq, "_MTBF"), Symbol(eq, "_RepairTime")],
            a -> begin
                mtbf = a[Symbol(eq, "_MTBF")]     
                mttr = a[Symbol(eq, "_RepairTime")] 
                
                A = mtbf / (mtbf + mttr)
                
                # Get thresholds
                thresholds = AVAIL_THRESHOLDS[eq]
                low_thresh  = thresholds[2]
                high_thresh = thresholds[3]
                margin = 0.05
                
                # Calculate probabilities
                p_low = max(0, min(1, (low_thresh + margin - A) / (2 * margin)))
                p_high = max(0, min(1, (A - high_thresh + margin) / (2 * margin)))
                p_med = 1 - p_low - p_high
                
                # Safety check for NaN values
                if any(isnan.([p_low, p_med, p_high])) || (p_low + p_med + p_high) ≈ 0
                    return Categorical([0.33, 0.34, 0.33])  # Fallback
                end
                
                return Categorical([p_high, p_med, p_low])
            end
        )



    end

    return d
end

"""
Test equipment reliability calculations for a single equipment type
"""
function test_reliability(equipment_type::Symbol)
    println("\n=== Testing $equipment_type Reliability ===")
    
    # Test for each age
    for age in [:New, :Mid, :Old]
        println("\nAge: $age")
        
        # Get Beta and Eta values
        beta = REL.BETA[age]
        eta = get_eta(equipment_type)[age]
        println("Beta: $beta")
        println("Eta: $eta")
        
        # Calculate expected MTBF (mean of Weibull)
        mtbf_dist = Weibull(beta, eta)
        mtbf = mean(mtbf_dist)
        println("Expected MTBF: $(round(mtbf, digits=2)) hours")
        
        # Get MTTR parameters and distribution
        mu_mttr, sd_mttr = get_repair_params(equipment_type)
        mttr_dist = truncated(Normal(mu_mttr, sd_mttr), 0, Inf)
        println("\nMTTR Distribution:")
        println("  Mean: $(round(mu_mttr, digits=2)) hours")
        println("  StdDev: $(round(sd_mttr, digits=2)) hours")
        
        # Calculate mean availability
        mttr = mean(mttr_dist)
        mean_availability = mtbf / (mtbf + mttr)
        println("\nMean Availability: $(round(mean_availability * 100, digits=2))%")
        
        # Calculate probabilistic state distribution
        thresholds  = AVAIL_THRESHOLDS[equipment_type]
        low_thresh  = thresholds[2]
        high_thresh = thresholds[3]
        margin = 0.05

        # Calculate state probabilities
        p_low = max(0, min(1, (low_thresh + margin - mean_availability) / (2 * margin)))
        p_high = max(0, min(1, (mean_availability - high_thresh + margin) / (2 * margin)))
        p_med = 1 - p_low - p_high

        println("\nAvailability State Probabilities:")
        println("  Low: $(round(p_low * 100, digits=2))%")
        println("  Medium: $(round(p_med * 100, digits=2))%")
        println("  High: $(round(p_high * 100, digits=2))%")
    end
end


# Export the test function
export build_equipment_availability_cpds, test_reliability, write_reliability_latex_table

end # module