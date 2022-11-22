function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}
# Do multiprocessing
vCPU=6 #number of CPUs to use

# Baseline, P1, P2, P3
graph_name=G_Carilion

n_t_for_eval=2
dose_response=exponential

for seeds_per_t in 1 3
do
    # echo "GT -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python AAAI_prep_GT.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response
    wait

    echo "Baseline -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python final_exp_B.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response &

    echo "Expected simulations"
    echo "P1 greedy -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python final_exp_P1.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -flag_lazy T -flag_expected_simulation T -dose_response $dose_response &

    echo "Expected simulations"
    echo "P2 - ISCK greedy -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python final_exp_P2.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -flag_lazy T -flag_expected_simulation T -compute_pi greedy -dose_response $dose_response &

    echo "Expected simulations"
    echo "P2 - ISCK multiplicative update -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python final_exp_P2.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -flag_lazy T -flag_expected_simulation T -compute_pi multiplicative_update -dose_response $dose_response &

    echo "Expected simulations"
    echo "P3 - greedy ratio -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -dose_response $dose_response"
    python final_exp_P3.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -flag_expected_simulation T -dose_response $dose_response &
    python final_exp_P3.py -name $graph_name -seeds_per_t $seeds_per_t -n_t_for_eval $n_t_for_eval -flag_expected_simulation T -flag_g_constraint T -dose_response $dose_response &

    pwait $vCPU
done
wait