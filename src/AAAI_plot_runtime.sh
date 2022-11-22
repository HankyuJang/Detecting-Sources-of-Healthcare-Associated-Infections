graph_name=UIHC_HCP_patient_room_withinHCPxPx 
for seeds_per_t in 1 2 3 4 5
do
    python final_exp_result_running_time_v2.py -name $graph_name -sampled T -dose_response exponential -seeds_per_t $seeds_per_t
done