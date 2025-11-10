module MaintenanceReliability

using BayesNets, Distributions
using SpecialFunctions: gamma
import ..Params: REL, AVAIL_THRESHOLDS, AVAIL_THRESHOLDS_MAINTENANCE

# policy types and frequencies
const PM_TYPES   = [:Minor, :Medium, :Major]
const PM_FREQS   = [:None, :Weekly, :Monthly, :Yearly]

# events per year
const PM_COUNT = Dict(:None=>0, :Weekly=>52, :Monthly=>12, :Yearly=>1)

# Per-equipment PM durations [hours/event]
const PM_DUR_EQ = Dict{Symbol,Dict{Symbol,Float64}}(
    :QC => Dict(:Minor=>2.0, :Medium=>6.0,  :Major=>24.0),
    :HT => Dict(:Minor=>1.0, :Medium=>2.0,  :Major=>4.0),
    :YC => Dict(:Minor=>3.0, :Medium=>6.0,  :Major=>20.0),
)

# Kijima-style rejuvenation per event (virtual-age rollback fraction)
const RHO = Dict(:Minor=>0.10, :Medium=>0.30, :Major=>0.60)

# Normalization for virtual-age 
const FREQ_NORM = 12.0

# Cap total relative beta reduction to avoid unrealistic flattening
const MAX_BETA_DROP = 0.30

# Gentle sensitivity for beta (wear-out) reduction
const B_BETA = 0.50

# effectiveness scalars 
const E_ETA  = Dict(:Minor=>0.10, :Medium=>0.25, :Major=>0.60)
const E_BETA = Dict(:Minor=>0.05, :Medium=>0.15, :Major=>0.30)

# year length
const HOURS_PER_YEAR = 8760.0
const EPS = 1e-9

function get_beta(eq::Symbol)
    eq == :QC ? REL.BETA_QC :
    eq == :YC ? REL.BETA_YC :
                REL.BETA_HT
end

function get_eta(eq::Symbol)
    eq == :QC ? REL.ETA_QC :
    eq == :YC ? REL.ETA_YC :
                REL.ETA_HT
end


function build_equipment_availability_cpds()
    d = Dict{Symbol, CPD}()

    for typ in PM_TYPES
        d[Symbol(typ, "_Policy")] = StaticCPD(
            Symbol(typ, "_Policy"),
            NamedCategorical(PM_FREQS, fill(1/length(PM_FREQS), length(PM_FREQS)))
        )
    end

    for typ in PM_TYPES
        d[Symbol(typ, "_Count")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(typ, "_Count"),
            [Symbol(typ, "_Policy")],
            a -> begin
                pol = a[Symbol(typ, "_Policy")]
                Distributions.Dirac(PM_COUNT[pol])
            end
        )
    end

    d[:E_eta] = FunctionalCPD{UnivariateDistribution}(
        :E_eta, [:Minor_Count,:Medium_Count,:Major_Count],
        a -> begin
            nmin = mean(a[:Minor_Count])  / FREQ_NORM
            nmed = mean(a[:Medium_Count]) / FREQ_NORM
            nmaj = mean(a[:Major_Count])  / FREQ_NORM
            eff = 1 - (1 - E_ETA[:Minor])^nmin *
                      (1 - E_ETA[:Medium])^nmed *
                      (1 - E_ETA[:Major])^nmaj
            Distributions.Dirac(clamp(eff,0,1))
        end
    )

    d[:E_beta] = FunctionalCPD{UnivariateDistribution}(
        :E_beta, [:Minor_Count,:Medium_Count,:Major_Count],
        a -> begin
            nmin = mean(a[:Minor_Count])  / FREQ_NORM
            nmed = mean(a[:Medium_Count]) / FREQ_NORM
            nmaj = mean(a[:Major_Count])  / FREQ_NORM
            eff = 1 - (1 - E_BETA[:Minor])^nmin *
                      (1 - E_BETA[:Medium])^nmed *
                      (1 - E_BETA[:Major])^nmaj
            Distributions.Dirac(clamp(eff,0,1))
        end
    )


    for eq in (:QC,:YC,:HT)

        # age category
        d[Symbol(eq, "_Age")] = StaticCPD(Symbol(eq,"_Age"),
            NamedCategorical([:New,:Mid,:Old],[0.2,0.6,0.2])
        )

        # base Weibull params
        d[Symbol(eq, "_Beta")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_Beta"), [Symbol(eq,"_Age")],
            a -> Distributions.Dirac(get_beta(eq)[a[Symbol(eq,"_Age")]])
        )
        d[Symbol(eq, "_Eta")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_Eta"), [Symbol(eq,"_Age")],
            a -> Distributions.Dirac(get_eta(eq)[a[Symbol(eq,"_Age")]])
        )

        # adjusted η' via Kijima-style virtual-age: η' = η / φ
        d[Symbol(eq, "_Eta_Adjusted")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_Eta_Adjusted"),
            [Symbol(eq,"_Eta"), :Minor_Count,:Medium_Count,:Major_Count],
            a -> begin
                η0   = Float64(mean(a[Symbol(eq,"_Eta")]))
                nmin = Float64(mean(a[:Minor_Count]))  / FREQ_NORM
                nmed = Float64(mean(a[:Medium_Count])) / FREQ_NORM
                nmaj = Float64(mean(a[:Major_Count]))  / FREQ_NORM
                ϕ = (1 - RHO[:Minor])^nmin * (1 - RHO[:Medium])^nmed * (1 - RHO[:Major])^nmaj
                ηp = η0 / max(ϕ, 1e-6)
                Distributions.Dirac(ηp)
            end
        )

        # adjusted β' with gentle cap
        d[Symbol(eq, "_Beta_Adjusted")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_Beta_Adjusted"),
            [Symbol(eq,"_Beta"), :E_beta],
            a -> begin
                β0 = Float64(mean(a[Symbol(eq,"_Beta")]))
                e  = Float64(clamp(mean(a[:E_beta]),0,1))
                rel_drop = min(B_BETA*e, MAX_BETA_DROP)
                βp = β0 * (1 - rel_drop)
                Distributions.Dirac(βp)
            end
        )

        # corrective MTTR 
        d[Symbol(eq,"_RepairTime")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_RepairTime"), Symbol[],
            _ -> truncated(Normal(Float64(REL.MU_MTTR[eq]),
                                  Float64(REL.SD_MTTR[eq])), 0, Inf)
        )

        # numeric MTBF value (Dirac over η' * Γ(1 + 1/β'))
        d[Symbol(eq,"_MTBF")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_MTBF"),
            [Symbol(eq,"_Beta_Adjusted"), Symbol(eq,"_Eta_Adjusted")],
            a -> begin
                βp = Float64(mean(a[Symbol(eq,"_Beta_Adjusted")]))
                ηp = Float64(mean(a[Symbol(eq,"_Eta_Adjusted")]))
                mtbf_val = ηp * gamma(1 + 1/βp)
                Distributions.Dirac(Float64(mtbf_val))
            end
        )

        # Availability (NHPP minimal repair)
        # Expected failures/year f solves: f = ((T - PM - f*MTTR)/η')^β'
        # Then A = 1 - PM/T - (f*MTTR)/T

        d[Symbol(eq,"_Availability_Value")] = FunctionalCPD{UnivariateDistribution}(
            Symbol(eq,"_Availability_Value"),
            [Symbol(eq,"_Beta_Adjusted"), Symbol(eq,"_Eta_Adjusted"),
             Symbol(eq,"_RepairTime"), :Minor_Count,:Medium_Count,:Major_Count],
            a -> begin
                βp = Float64(a[Symbol(eq,"_Beta_Adjusted")])  
                ηp = Float64(a[Symbol(eq,"_Eta_Adjusted")])     
                mttr = Float64(a[Symbol(eq,"_RepairTime")])   
                nmin = Float64(a[:Minor_Count])               
                nmed = Float64(a[:Medium_Count])              
                nmaj = Float64(a[:Major_Count])               

                dur = PM_DUR_EQ[eq]
                pm_year = nmin*dur[:Minor] + nmed*dur[:Medium] + nmaj*dur[:Major]
                T = HOURS_PER_YEAR
        
                # Safety clamps
                βp = clamp(βp, 0.5, 10.0)
                ηp = clamp(ηp, 1e-3, 1e6)
                mttr = clamp(mttr, 0.01, 1e3)
                pm_year = clamp(pm_year, 0.0, 0.9*T)  # Ensure some operational time
        
                # Fixed-point iteration with better initialization
                U = max(T - pm_year, 0.1*T)  # Ensure U > 0
                f = max((U/ηp)^βp, 1e-6)     # Ensure f > 0
                
                for i in 1:10  # More iterations
                    U_new = T - pm_year - f*mttr
                    U_new = max(U_new, 0.01*T)  # Prevent U from going to zero
                    
                    f_new = (U_new/ηp)^βp
                    if !isfinite(f_new) || f_new <= 0
                        f_new = 1e-6
                    end
                    
                    # Damped update for stability
                    f = 0.3*f + 0.7*f_new
                    
                    # Check convergence
                    if abs(f_new - f) / max(f, 1e-6) < 1e-6
                        break
                    end
                end
        
                # Final availability calculation
                A_year = 1.0 - pm_year/T - (f*mttr)/T
                A_year = clamp(A_year, 0.0, 1.0)
                
                if !isfinite(A_year)
                    @warn "A_year still NaN after fixes" eq βp ηp f mttr pm_year U T
                    A_year = 0.5  # Fallback value
                end
                
                Distributions.Dirac(A_year)
            end
        )        
    end

    for eq in (:QC,:YC,:HT)
        d[Symbol(eq,"_Availability")] = FunctionalCPD{Categorical}(
            Symbol(eq,"_Availability"),
            [Symbol(eq,"_Availability_Value")],
            a -> begin
                A = Float64(a[Symbol(eq,"_Availability_Value")])  
                
                # Simple thresholds 
                if A >= 0.90
                    return Categorical([1.0, 0.0, 0.0])      
                elseif A >= 0.85  
                    return Categorical([0.0, 1.0, 0.0])      
                else
                    return Categorical([0.0, 0.0, 1.0])    
                end
            end
        )
    end

    return d
end

export build_equipment_availability_cpds, get_eta, get_beta,
       PM_TYPES, PM_FREQS, PM_COUNT, PM_DUR_EQ, RHO, FREQ_NORM

end # module
