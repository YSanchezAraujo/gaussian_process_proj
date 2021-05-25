using MAT;
using StatsBase, LinearAlgebra;

struct SpikeInfo 
    isi::Array{Float64, 1}
    h::Histogram
    bins::Array{Float64, 1}
    f::Float64
    p::Array{Float64, 1}
end

function compute_spike_info(data_path::String, key::String, bin_size::Float64)
    data_dict = matread(data_path)
    spike_times = data_dict[key]
    if length(size(spike_times)) == 2 && size(spike_times, 2) == 1
        spike_times = dropdims(spike_times; dims=2)
    end
    total_time = ceil(maximum(spike_times))
    isi = diff(spike_times)
    bins = 1:bin_size:total_time
    h_obj = fit(Histogram, spike_times, bins)
    firing_rate = sum(h_obj.weights) / length(spike_times)
    prob_spike = h_obj.weights / length(isi)
    return SpikeInfo(isi, h_obj, bins, firing_rate, prob_spike), h_obj.weights
end

nanmean(x) = mean(filter(!isnan, x))
nanmean(x, dim) = mapslices(nanmean, x, dims=dim)
binindices(edges, data) = searchsortedlast.(Ref(edges), data)

function avg_by_index(data, index)
    avgs = zeros(Float64, maximum(index))
    for (k, s) in enumerate(unique(index))
        idx = index .== s
        avgs[k] = mean(data[idx])
    end
    return avgs
end

"""
X: (x, y) average position across cameras, and within time bins
y: spike counts within time bins
bins: bins used to create X, y
h: histogram with more information about y
gs: grid score
"""
struct GridData
    X::Array{Float64, 2}
    y::Array{Int64, 1}
    bins::StepRangeLen
    h::Histogram
    gs::Float64
end

function preprocess(spike_path::String, loc_path::String, bin_size::Float64)
    spike_dict, loc_dict = matread(spike_path), matread(loc_path)
    xPos = [loc_dict["posx"] loc_dict["posx2"]]
    yPos = [loc_dict["posy"] loc_dict["posy2"]]
    xPos_avg, yPos_avg = dropdims(nanmean(xPos, 2); dims=2), dropdims(nanmean(yPos, 2); dims=2)
    pos_time = dropdims(loc_dict["post"]; dims=2)
    min_time, max_time = floor(minimum(pos_time)), ceil(maximum(pos_time))
    bins = min_time:bin_size:max_time
    time_index_bins = binindices(bins, pos_time)
    binned_xPos_avg = avg_by_index(xPos_avg, time_index_bins)
    binned_yPos_avg = avg_by_index(yPos_avg, time_index_bins)
    h_obj = StatsBase.fit(Histogram, dropdims(spike_dict["cellTS"]; dims=2), bins)
    # finally check if there are any nan's and remove those samples
    X, y =  [binned_xPos_avg binned_yPos_avg], h_obj.weights
    x1nanidx = findall(.!isnan.(X[:, 1]))
    x2nanidx = findall(.!isnan.(X[:, 2]))
    xidx = unique(vcat(x1nanidx, x2nanidx))
    X, y = X[xidx, :], y[xidx]
    return GridData(X, y, bins, h_obj, spike_dict["grid_score"])
end

#meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
# Xg, Yg = meshgrid(1:90, 1:90)
# Xg = reshape(Xg, (90, 90))
# Yg = reshape(Yg, (90, 90))

# # look at spike counts per grid of positions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(Xg, Yg, reshape(data.y, (90, 26)), cmap="viridis")
# plt.pcolormesh(Xg, Yg, reshape([data.y;0], (90, 90)), cmap="viridis")

