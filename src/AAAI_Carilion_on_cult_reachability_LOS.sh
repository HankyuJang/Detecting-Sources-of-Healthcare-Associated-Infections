function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

# Do multiprocessing
vCPU=3 #number of CPUs to use

graph_name=G_Carilion

for seeds_per_t in 1 3
do
    # Only proceed after generating time expanded graph
    echo "Generate time expanded graph"
    python gen_time_expanded_graph.py -name $graph_name -dose_response exponential -seeds_per_t $seeds_per_t
    wait

    echo "Baseline Cult"
    python d_steiner_alpha_0.py -name $graph_name -dose_response exponential -seeds_per_t $seeds_per_t &

    echo "Baseline Reachability"
    python B_reachability.py -name $graph_name -dose_response exponential -seeds_per_t $seeds_per_t &

    echo "Baseline Length of Stay"
    python B_LOS.py -name $graph_name -dose_response exponential -seeds_per_t $seeds_per_t &

    pwait  $vCPU
done
wait
