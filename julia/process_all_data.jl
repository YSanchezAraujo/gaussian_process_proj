include("/Users/yoelsanchezaraujo/Desktop/preproc_data.jl");
using CSV;
using DataFrames;

# run this file to get the data into "preprocessed format" 
# that will write common data format to read into python & julia

files_path = "/Users/yoelsanchezaraujo/Desktop/grid_data_pillow"
file_pairs = []
files = readdir(files_path)

pos_strings = [x for x in files if occursin("_pos", x)]
spk_strings = [x for x in files if occursin("_T", x)]

for pstring in pos_strings
	match_str = split(pstring, "_pos")[1]
	spike_list = [j for j in spk_strings if occursin(match_str, j)]
	push!(file_pairs, (pstring, spike_list))
end

for (pos, spks) in file_pairs
	pos_path = joinpath(files_path, pos)
	spike_paths = [joinpath(files_path, x) for x in spks]
	for (j, spkp) in enumerate(spike_paths)
		data = preprocess(spkp, pos_path, 1.)
        X, y = data.X, data.y
        df = DataFrame(xpos=X[:, 1], ypos=X[:, 2], spikecount=y)
        df_spk_str = split(spks[j], ".mat")[1]
        dfname = string(df_spk_str, "_pos.csv")
        CSV.write(dfname, df)
    end
end