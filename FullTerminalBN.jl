module FullTerminalBN

using BayesNets, Distributions

#Efficiency helpers 
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

    #QC not operable if wind is 3
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
                # now 1 is good
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

    # Operator availability

    # Shift and Strike nodes
    d[:Shift]  = StaticCPD(:Shift,  Categorical([0.60, 0.40]))   # 1=Day, 2=Night
    d[:Strike] = StaticCPD(:Strike, Categorical([0.95, 0.05]))   # 1=No, 2=Yes

    d[:Operator_Availability_QC] = CategoricalCPD(
        :Operator_Availability_QC, [:Shift, :Strike], [2, 2],
        [
            # Index 1: Shift=1, Strike=1 (Day, No Strike) 
            Categorical([0.90, 0.10, 0.00]),  
            # Index 2: Shift=2, Strike=1 (Night, No Strike)
            Categorical([0.85, 0.15, 0.00]),  
            # Index 3: Shift=1, Strike=2 (Day, Strike) 
            Categorical([0.00, 0.00, 1.00]),  
            # Index 4: Shift=2, Strike=2 (Night, Strike)
            Categorical([0.00, 0.00, 1.00]),  
        ]
    )
    
    d[:Operator_Availability_HT] = CategoricalCPD(
        :Operator_Availability_HT, [:Shift, :Strike], [2, 2],
        [
            # Index 1: Day, No Strike
            Categorical([0.95, 0.05, 0.00]),  
            # Index 2: Night, No Strike 
            Categorical([0.90, 0.10, 0.00]),  
            # Index 3: Day, Strike
            Categorical([0.00, 0.00, 1.00]),  
            # Index 4: Night, Strike
            Categorical([0.00, 0.00, 1.00]),  
        ]
    )
    
    d[:Operator_Availability_YC] = CategoricalCPD(
        :Operator_Availability_YC, [:Shift, :Strike], [2, 2],
        [
            # Index 1: Day, No Strike
            Categorical([0.95, 0.05, 0.00]),  
            # Index 2: Night, No Strike 
            Categorical([0.90, 0.10, 0.00]),  
            # Index 3: Day, Strike
            Categorical([0.00, 0.00, 1.00]),  
            # Index 4: Night, Strike
            Categorical([0.00, 0.00, 1.00]),  
        ]
    )
    
    # QC Effective Availability 
    d[:QC_Availability_Effective] = CategoricalCPD(
        :QC_Availability_Effective, [:QC_Availability, :Operator_Availability_QC], [3, 3],
        [
            # Equipment=1 (High), Operator=1 (High) → Effective=1 (High)
            Categorical([1.0, 0.0, 0.0]),
            # Equipment=1 (High), Operator=2 (Medium) → Effective=2 (Medium) 
            Categorical([0.0, 1.0, 0.0]),
            # Equipment=1 (High), Operator=3 (Low) → Effective=3 (Low)
            Categorical([0.0, 0.0, 1.0]),
            
            # Equipment=2 (Medium), Operator=1 (High) → Effective=2 (Medium)
            Categorical([0.0, 1.0, 0.0]),
            # Equipment=2 (Medium), Operator=2 (Medium) → Effective=2 (Medium)
            Categorical([0.0, 1.0, 0.0]),
            # Equipment=2 (Medium), Operator=3 (Low) → Effective=3 (Low)
            Categorical([0.0, 0.0, 1.0]),
            
            # Equipment=3 (Low), Operator=1 (High) → Effective=3 (Low)
            Categorical([0.0, 0.0, 1.0]),
            # Equipment=3 (Low), Operator=2 (Medium) → Effective=3 (Low)
            Categorical([0.0, 0.0, 1.0]),
            # Equipment=3 (Low), Operator=3 (Low) → Effective=3 (Low)
            Categorical([0.0, 0.0, 1.0]),
        ]
    )
    
    # HT Effective Availability
    d[:HT_Availability_Effective] = CategoricalCPD(
        :HT_Availability_Effective, [:HT_Availability, :Operator_Availability_HT], [3, 3],
        [
            Categorical([1.0, 0.0, 0.0]), # (1,1) → 1
            Categorical([0.0, 1.0, 0.0]), # (1,2) → 2
            Categorical([0.0, 0.0, 1.0]), # (1,3) → 3
            Categorical([0.0, 1.0, 0.0]), # (2,1) → 2
            Categorical([0.0, 1.0, 0.0]), # (2,2) → 2
            Categorical([0.0, 0.0, 1.0]), # (2,3) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,1) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,2) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,3) → 3
        ]
    )
    
    # YC Effective Availability 
    d[:YC_Availability_Effective] = CategoricalCPD(
        :YC_Availability_Effective, [:YC_Availability, :Operator_Availability_YC], [3, 3],
        [
            Categorical([1.0, 0.0, 0.0]), # (1,1) → 1
            Categorical([0.0, 1.0, 0.0]), # (1,2) → 2
            Categorical([0.0, 0.0, 1.0]), # (1,3) → 3
            Categorical([0.0, 1.0, 0.0]), # (2,1) → 2
            Categorical([0.0, 1.0, 0.0]), # (2,2) → 2
            Categorical([0.0, 0.0, 1.0]), # (2,3) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,1) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,2) → 3
            Categorical([0.0, 0.0, 1.0]), # (3,3) → 3
        ]
    )
    
    d[:Strike_Impact] = CategoricalCPD(
        :Strike_Impact, [:Strike], [2],
        [
            Categorical([1.0, 0.0, 0.0]),  # Strike=1 (No): No impact (state 1)
            Categorical([0.0, 0.0, 1.0])   # Strike=2 (Yes): High impact (state 3)
        ]
    )


    # Delays
    
    #HT delay empty
    let d = d
        rows = Categorical[]
        for avail in 1:3, busy in 1:3, strike_impact in 1:3
            if strike_impact == 3  # Strike impact forces high delay
                push!(rows, Categorical([0.0, 0.0, 1.0]))  # 100% high delay
            else
                if avail == 3  # Low availability 
                    if busy == 1
                        push!(rows, Categorical([0.3, 0.3, 0.4]))  
                    elseif busy == 2
                        push!(rows, Categorical([0.2, 0.3, 0.5])) 
                    else
                        push!(rows, Categorical([0.1, 0.3, 0.6]))  
                    end
                elseif avail == 2  # Medium availability (92-97%)
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
        end
        d[:HT_Delay_Empty] = CategoricalCPD(
            :HT_Delay_Empty,
            [:HT_Availability_Effective, :Terminal_Busyness, :Strike_Impact],  
            [3, 3, 3],  
            rows
        )
    end

    #QC delay
    let d = d
        rows = Categorical[]
        for oper in 1:2, avail in 1:3, ht in 1:3, eff in 1:10, strike_impact in 1:3
            if oper == 2 || strike_impact == 3  # Not operable OR strike impact
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
            [:Crane_Operable, :QC_Availability_Effective, :HT_Delay_Empty, :Efficiency, :Strike_Impact],
            [2, 3, 3, 10, 3],
            rows
        )
    end 

    #YC delay
    let d = d
        rows = Categorical[]
        for avail in 1:3, storage in 1:3, strike_impact in 1:3
            if strike_impact == 3  # Strike impact forces high delay
                push!(rows, Categorical([0.0, 0.0, 1.0]))  # 100% high delay
            else
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
        end
        d[:Yard_Crane_Delay] = CategoricalCPD(
            :Yard_Crane_Delay,
            [:YC_Availability_Effective, :Storage_Capacity_Level, :Strike_Impact],
            [3, 3, 3],
            rows
        )
    end




    
    #HT_Delay_Loaded
    let d = d
        rows = Categorical[]
        for avail in 1:3, yc_delay in 1:3, strike_impact in 1:3
            if strike_impact == 3  # Strike impact forces high delay
                push!(rows, Categorical([0.0, 0.0, 1.0]))  # 100% high delay
            else
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
        end
        d[:HT_Delay_Loaded] = CategoricalCPD(
            :HT_Delay_Loaded,
            [:HT_Availability_Effective, :Yard_Crane_Delay, :Strike_Impact], 
            [3, 3, 3],  
            rows
        )
    end
    
    # total delay

    let d = d
        rows = Categorical[]
        for c in 1:3, h in 1:3  # Crane_Delay, HT_Delay_Loaded
            if c == 3  # High QC delay 
                if h == 3
                    push!(rows, Categorical([0.0, 0.0, 1.0]))   
                elseif h == 2
                    push!(rows, Categorical([0.0, 0.05, 0.95])) 
                else
                    push!(rows, Categorical([0.0, 0.1, 0.9]))  
                end
            elseif c == 2  # Medium QC delay
                if h == 3
                    push!(rows, Categorical([0.05, 0.30, 0.65]))   
                elseif h == 2
                    push!(rows, Categorical([0.1, 0.35, 0.55])) 
                else
                    push!(rows, Categorical([0.1, 0.4, 0.5])) 
                end
            else  # Low QC delay 
                if h == 3
                    push!(rows, Categorical([0.35, 0.45, 0.2])) 
                elseif h == 2
                    push!(rows, Categorical([0.55, 0.35, 0.1])) 
                else
                    push!(rows, Categorical([0.9, 0.1, 0.0]))   
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
       

function build_full_bn(allcpds::Dict{Symbol,CPD})
    bn = BayesNet()
    
    node_order = [
        # 1) Weather & operability
        :Wind, :Rain, :Visibility, :Crane_Operable,

        # 2) Operator primitives & availability
        :Shift, :Strike, :Strike_Impact,
        :Operator_Availability_QC, :Operator_Availability_HT, :Operator_Availability_YC,

        # 3) Terminal state
        :Terminal_Busyness, :Storage_Capacity_Level,

        :PM_Type, :PM_Freq, :PM_Effectiveness,

         # Maintenance portfolio nodes
        :Minor_Policy, :Medium_Policy, :Major_Policy,
        :Minor_Count,  :Medium_Count,  :Major_Count,
        :E_eta, :E_beta,

    
        :QC_Age, :QC_Beta, :QC_Eta, :QC_Beta_Adjusted, :QC_Eta_Adjusted,
        :QC_MTBF, :QC_RepairTime, :QC_MTTR_Total,

        :YC_Age, :YC_Beta, :YC_Eta, :YC_Beta_Adjusted, :YC_Eta_Adjusted,
        :YC_MTBF, :YC_RepairTime, :YC_MTTR_Total,

        :HT_Age, :HT_Beta, :HT_Eta, :HT_Beta_Adjusted, :HT_Eta_Adjusted,
        :HT_MTBF, :HT_RepairTime, :HT_MTTR_Total,


        # 6) Availability (maintenance-aware, produced by MaintenanceReliability)
        :QC_Availability_Value, :YC_Availability_Value, :HT_Availability_Value,  
        :QC_Availability, :YC_Availability, :HT_Availability,  

        # 7) Effective availability 
        :QC_Availability_Effective, :HT_Availability_Effective, :YC_Availability_Effective,

        # 8) Efficiency
        :Efficiency,

        # 9) Delays
        :HT_Delay_Empty,
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

export base_cpds, build_full_bn

end # module