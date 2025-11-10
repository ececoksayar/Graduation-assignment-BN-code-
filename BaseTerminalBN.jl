module BaseTerminalBN

using BayesNets, Distributions

# --- Efficiency helpers ---
const wind_efficiency = Dict(1 => 1.0, 2 => 0.70, 3 => 0.0)
const rain_multiplier = Dict(1 => 1.0, 2 => 0.95, 3 => 0.85, 4 => 0.7)
const visibility_multiplier = Dict(1 => 1.0, 2 => 0.95, 3 => 0.85, 4 => 0.7)

function efficiency_bin(eff)
    if eff >= 0.9; return 1      # Best efficiency
    elseif eff >= 0.8; return 2
    elseif eff >= 0.7; return 3
    elseif eff >= 0.6; return 4
    elseif eff >= 0.5; return 5
    elseif eff >= 0.4; return 6
    elseif eff >= 0.3; return 7
    elseif eff >= 0.2; return 8
    elseif eff >= 0.1; return 9
    else; return 10              # Worst efficiency
    end
end

function base_cpds()
    d = Dict{Symbol,CPD}()

    # Environment CPDs
    d[:Wind] = StaticCPD(:Wind, Categorical([0.917, 0.08, 0.003]))
    d[:Rain] = StaticCPD(:Rain, Categorical([0.3711, 0.3366, 0.2107, 0.0816]))
    d[:Visibility] = StaticCPD(:Visibility, Categorical([0.8637, 0.0345, 0.0493, 0.0525]))

    #Crane not operable if wind is 3
    d[:Crane_Operable] = CategoricalCPD(
        :Crane_Operable, [:Wind], [3],
        [Categorical([1.0, 0.0]), Categorical([1.0, 0.0]), Categorical([0.0, 1.0])]
    )

    # Efficiency (weather × operability) 
    efficiency_states = [(op, w, r, v) for op in 1:2, w in 1:3, r in 1:4, v in 1:4]
    efficiency_bins = [
        begin
            if op == 2  # Not operable
                10  # Lowest efficiency bin
            else
                eff = wind_efficiency[w] * rain_multiplier[r] * visibility_multiplier[v]
                if eff >= 0.9; 1
                elseif eff >= 0.8; 2
                elseif eff >= 0.7; 3
                elseif eff >= 0.6; 4
                elseif eff >= 0.5; 5
                elseif eff >= 0.4; 6
                elseif eff >= 0.3; 7
                elseif eff >= 0.2; 8
                elseif eff >= 0.1; 9
                else 10 end
            end
        end
        for (op, w, r, v) in efficiency_states
    ]

    d[:Efficiency] = CategoricalCPD(
        :Efficiency, [:Crane_Operable, :Wind, :Rain, :Visibility], [2, 3, 4, 4],
        vec([Categorical([bin == i ? 1.0 : 0.0 for i in 1:10]) for bin in efficiency_bins])
    )

    # Terminal busyness
    d[:Terminal_Busyness] = StaticCPD(
    :Terminal_Busyness, 
    Categorical([0.2, 0.6, 0.2])  # [1=Low, 2=Normal, 3=High]
)
    # Storage capacity level
    d[:Storage_Capacity_Level] = StaticCPD(
        :Storage_Capacity_Level, 
        Categorical([0.2, 0.4, 0.4])
    )

    # Delays
    
    # HT delay empty
    let d = d
        rows = Categorical[]
        for avail in 1:3, busy in 1:3
            if avail == 3  # Low availability 
                if busy == 1
                    push!(rows, Categorical([0.3, 0.3, 0.4]))  
                elseif busy == 2
                    push!(rows, Categorical([0.2, 0.3, 0.5]))  
                else
                    push!(rows, Categorical([0.1, 0.3, 0.6]))  
                end
            elseif avail == 2  # Medium availability 
                if busy == 1
                    push!(rows, Categorical([0.6, 0.3, 0.1]))  
                elseif busy == 2
                    push!(rows, Categorical([0.5, 0.3, 0.2]))  
                else
                    push!(rows, Categorical([0.3, 0.4, 0.3]))  
                end
            else  # High availability
                if busy == 1
                    push!(rows, Categorical([0.8, 0.15, 0.05]))  
                elseif busy == 2
                    push!(rows, Categorical([0.7, 0.2, 0.1]))    
                else
                    push!(rows, Categorical([0.6, 0.3, 0.1]))    
                end
            end
        end
        d[:HT_Delay_Empty] = CategoricalCPD(
            :HT_Delay_Empty,
            [:HT_Availability, :Terminal_Busyness],
            [3, 3],
            rows
        )
    end

    #Crane Delay
    let d = d
        rows = Categorical[]
        for oper in 1:2, avail in 1:3, ht in 1:3, eff in 1:10
            if oper == 2  # Not operable 
                push!(rows, Categorical([0.0, 0.0, 1.0]))  # Always high delay
            else
                if avail == 3  # Low availability
                    if ht == 3
                        if eff <= 3
                            push!(rows, Categorical([0.0, 0.1, 0.9]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.0, 0.08, 0.92]))
                        else
                            push!(rows, Categorical([0.0, 0.05, 0.95]))
                        end
                    elseif ht == 2
                        if eff <= 3
                            push!(rows, Categorical([0.0, 0.15, 0.85]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.0, 0.13, 0.87]))
                        else
                            push!(rows, Categorical([0.0, 0.1, 0.9]))
                        end
                    else # ht == 1
                        if eff <= 3
                            push!(rows, Categorical([0.0, 0.2, 0.8]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.0, 0.18, 0.82]))
                        else
                            push!(rows, Categorical([0.0, 0.15, 0.85]))
                        end
                    end
                elseif avail == 2  # Medium availability
                    if ht == 3
                        if eff <= 3
                            push!(rows, Categorical([0.0, 0.30, 0.7]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.0, 0.28, 0.72]))
                        else
                            push!(rows, Categorical([0.0, 0.25, 0.75]))
                        end
                    elseif ht == 2
                        if eff <= 3
                            push!(rows, Categorical([0.1, 0.25, 0.65]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.05, 0.28, 0.67]))
                        else
                            push!(rows, Categorical([0.05, 0.25, 0.7]))
                        end
                    else # ht == 1
                        if eff <= 3
                            push!(rows, Categorical([0.2, 0.2, 0.6]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.18, 0.2, 0.62]))
                        else
                            push!(rows, Categorical([0.1, 0.25, 0.65]))
                        end
                    end
                else  # High availability
                    if ht == 3
                        if eff <= 3
                            push!(rows, Categorical([0.6, 0.2, 0.2]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.45, 0.3, 0.25]))
                        else
                            push!(rows, Categorical([0.35, 0.4, 0.25]))
                        end
                    elseif ht == 2
                        if eff <= 3
                            push!(rows, Categorical([0.7, 0.2, 0.1]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.7, 0.25, 0.05]))
                        else
                            push!(rows, Categorical([0.5, 0.35, 0.15]))
                        end
                    else # ht == 1
                        if eff <= 3
                            push!(rows, Categorical([0.9, 0.1, 0.0]))
                        elseif eff <= 7
                            push!(rows, Categorical([0.8, 0.2, 0.0]))
                        else
                            push!(rows, Categorical([0.7, 0.3, 0.0]))
                        end
                    end
                end
            end
        end
        d[:Crane_Delay] = CategoricalCPD(
            :Crane_Delay,
            [:Crane_Operable, :QC_Availability, :HT_Delay_Empty, :Efficiency],
            [2, 3, 3, 10],
            rows
        )
    end
    
    #YC delay
    let d = d
        rows = Categorical[]
        for avail in 1:3, storage in 1:3
            if avail == 3  # Low availability 
                if storage == 1
                    push!(rows, Categorical([0.2, 0.3, 0.5]))  
                elseif storage == 2
                    push!(rows, Categorical([0.1, 0.3, 0.6]))  
                else
                    push!(rows, Categorical([0.0, 0.2, 0.8]))  
                end
            elseif avail == 2  # Medium availability 
                if storage == 1
                    push!(rows, Categorical([0.5, 0.3, 0.2]))  
                elseif storage == 2
                    push!(rows, Categorical([0.4, 0.4, 0.2]))  
                else
                    push!(rows, Categorical([0.3, 0.4, 0.3]))  
                end
            else  # High availability
                if storage == 1
                    push!(rows, Categorical([0.8, 0.15, 0.05]))  
                elseif storage == 2
                    push!(rows, Categorical([0.7, 0.2, 0.1]))    
                else
                    push!(rows, Categorical([0.6, 0.25, 0.15]))  
                end
            end
        end
        d[:Yard_Crane_Delay] = CategoricalCPD(
            :Yard_Crane_Delay,
            [:YC_Availability, :Storage_Capacity_Level],
            [3, 3],
            rows
        )
    end  
    
    # HT_Delay_Loaded
    let d = d
        rows = Categorical[]
        for avail in 1:3, yc_delay in 1:3
            if avail == 3  # Low availability
                if yc_delay == 1
                    push!(rows, Categorical([0.3, 0.3, 0.4]))
                elseif yc_delay == 2
                    push!(rows, Categorical([0.2, 0.3, 0.5]))
                else
                    push!(rows, Categorical([0.0, 0.2, 0.8]))
                end
            elseif avail == 2  # Medium availability
                if yc_delay == 1
                    push!(rows, Categorical([0.6, 0.3, 0.1]))
                elseif yc_delay == 2
                    push!(rows, Categorical([0.5, 0.3, 0.2]))
                else
                    push!(rows, Categorical([0.4, 0.4, 0.2]))
                end
            else  # High availability
                if yc_delay == 1
                    push!(rows, Categorical([0.9, 0.1, 0.0]))
                elseif yc_delay == 2
                    push!(rows, Categorical([0.8, 0.2, 0.0]))
                else
                    push!(rows, Categorical([0.7, 0.2, 0.1]))
                end
            end
        end
        d[:HT_Delay_Loaded] = CategoricalCPD(
            :HT_Delay_Loaded,
            [:HT_Availability, :Yard_Crane_Delay],  # <-- now correct
            [3, 3],
            rows
        )
    end
    
    # total delay
    let d = d
        rows = Categorical[]
        for c in 1:3, h in 1:3  # Crane_Delay, HT_Delay_Loaded
            if c == 3  # High crane delay - should dominate
                if h == 3
                    push!(rows, Categorical([0.0, 0.0, 1.0]))   # both high → full failure
                elseif h == 2
                    push!(rows, Categorical([0.0, 0.05, 0.95])) # crane high dominates
                else
                    push!(rows, Categorical([0.0, 0.1, 0.9]))   # crane high even if HT low
                end
            elseif c == 2  # Medium crane delay - stronger effect than before
                if h == 3
                    push!(rows, Categorical([0.05, 0.30, 0.65]))   # HT high + crane medium → very bad
                elseif h == 2
                    push!(rows, Categorical([0.1, 0.35, 0.55])) # both medium
                else
                    push!(rows, Categorical([0.1, 0.4, 0.5])) # crane medium alone → significant
                end
            else  # Low crane delay - HT influences more
                if h == 3
                    push!(rows, Categorical([0.35, 0.45, 0.2])) # HT high still hurts
                elseif h == 2
                    push!(rows, Categorical([0.55, 0.35, 0.1])) # HT medium
                else
                    push!(rows, Categorical([0.9, 0.1, 0.0]))   # best case
                end
            end
        end

        d[:Total_Delay] = CategoricalCPD(
            :Total_Delay,
            [:Crane_Delay, :HT_Delay_Loaded],
            [3, 3],
            rows
        )
    end


    return d 
end
       

function build_base_bn(allcpds::Dict{Symbol,CPD})
    bn = BayesNet()
    node_order = [
        # Weather and Environment
        :Wind, :Rain, :Visibility, :Crane_Operable,
        
        # Terminal State (moved earlier)
        :Terminal_Busyness, :Storage_Capacity_Level,
        
        # Equipment reliability (only include nodes defined in base_cpds)
        :QC_Age, :QC_Beta, :QC_Eta, :QC_MTBF, :QC_RepairTime, :QC_Availability,
        :YC_Age, :YC_Beta, :YC_Eta, :YC_MTBF, :YC_RepairTime, :YC_Availability,
        :HT_Age, :HT_Beta, :HT_Eta, :HT_MTBF, :HT_RepairTime, :HT_Availability,

        # Equipment Operation & Efficiency
        :Efficiency, 
        
        # Delays
        :HT_Delay_Empty,        
        #:QC_Waiting_for_HT,
        :Crane_Delay, :Yard_Crane_Delay, :HT_Delay_Loaded, :Total_Delay
    ]
    
    for name in node_order
        if haskey(allcpds, name)
            push!(bn, allcpds[name])
        else
            @warn "Missing node in CPDs: $name"
        end
    end
    return bn
end

export base_cpds, build_base_bn

end # module