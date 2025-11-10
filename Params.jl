module Params
export REL, AVAIL_THRESHOLDS, AVAIL_THRESHOLDS_MAINTENANCE

#  Reliability config 
const REL = (
    # Shape parameters by age
    BETA = Dict(:New=>1.2, :Mid=>1.5, :Old=>2.0),

    BETA_QC = Dict(:New=>1.2, :Mid=>1.5, :Old=>2.0),
    BETA_YC = Dict(:New=>1.2, :Mid=>1.5, :Old=>2.0),
    BETA_HT = Dict(:New=>1.2, :Mid=>1.5, :Old=>2.0),
    
    # Scale parameters by equipment type and age
    ETA_QC = Dict(:New=>600.0, :Mid=>500.0, :Old=>400.0),
    ETA_YC = Dict(:New=>300.0, :Mid=>250.0, :Old=>220.0),
    ETA_HT = Dict(:New=>400.0, :Mid=>350.0, :Old=>300.0),


    # MTTR parameters by equipment type, normal distribution 
    MU_MTTR = Dict(
        :QC => 24,
        :YC => 14,
        :HT => 8
    ),
    SD_MTTR = Dict(
        :QC => 12,
        :YC => 7,
        :HT => 4   
    ),
    
)

# Thresholds for availability states
const AVAIL_THRESHOLDS = Dict(
    :QC => [0.0, 0.92, 0.97, 1.0],
    :YC => [0.0, 0.92, 0.97, 1.0],
    :HT => [0.0, 0.95, 0.98, 1.0]
)

const AVAIL_THRESHOLDS_MAINTENANCE = Dict(
    :QC => [0.0, 0.88, 0.95, 1.0],
    :YC => [0.0, 0.88, 0.95, 1.0],
    :HT => [0.0, 0.88, 0.95, 1.0]
)


end # module
