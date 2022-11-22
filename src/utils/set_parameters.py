import math
import pandas as pd

# tp: abbrev for tuned_paramters
def set_simulation_parameters(args, k_total):
    # assert k_total in [2, 4, 6]

    if args.name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
        df_row = pd.read_csv("../tables/parameter_tuning/G_UVA_k{}.csv".format(k_total))
    elif args.name in ["G_Carilion", "G_Carilion_v3"]:
        # This csv file has 1 row, 5 columns, each value corresponding to the parameter
        df_row = pd.read_csv("../tables/parameter_tuning/G_Carilion_k{}.csv".format(k_total))

    else:
        name = args.name
        if args.sampled:
            sampled = 1
        else:
            sampled = 0
        assert name in ["Karate_temporal", "UIHC_HCP_patient_room_withinHCPxPx"]

        df_tp = pd.read_csv("../tables/parameter_tuning/tuned_parameters.csv")
        df_row = df_tp[(df_tp["name"] == name) & (df_tp["k_total"] == k_total) & (df_tp["sampled"] == sampled)]
        df_row.reset_index(drop=True, inplace=True)

    rho = df_row.at[0, "trans-eff"]
    d = df_row.at[0, "die-off"]
    q = df_row.at[0, "shedding"]
    pi = df_row.at[0, "infectivity"]
    contact_area = df_row.at[0, "A(contact)"]

    return rho, d, q, pi, contact_area

# NOTE: These parameters are chosen based on the parameter tuning results
# It yields 5 - 10 % infections at time T, and a large number of recovery events
# NOTE: This sets parameters arbitrality. not used anymore
def set_simulation_parameters_prev(args, k_total):
    name = args.name
    sampled = args.sampled
    assert name in ["Karate_temporal", "UIHC_HCP_patient_room_withinHCPxPx"]
    assert k_total in [2, 4, 6]
    if name == "Karate_temporal":
        if k_total == 2:
            rho = pow(math.e, -1)
            d = pow(math.e, -2)
            q = pow(math.e, 1)
            pi = pow(math.e, -2)
            contact_area = 2000*pow(math.e, -3)
        elif k_total == 4:
            rho = pow(math.e, 0)
            d = pow(math.e, -2)
            q = pow(math.e, -1)
            pi = pow(math.e, 0)
            contact_area = 2000*pow(math.e, -4)
        elif k_total == 6:
            rho = pow(math.e, -4)
            d = pow(math.e, -3)
            q = pow(math.e, 2)
            pi = pow(math.e, -3)
            contact_area =  2000*pow(math.e, -3)
    elif name == "UIHC_HCP_patient_room_withinHCPxPx":
        if sampled: # UIHC sampled graph
            if k_total == 2:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 1)
                pi = pow(math.e, -2)
                contact_area = 2000
            elif k_total == 4:
                rho = pow(math.e, 0)
                d = pow(math.e, -3)
                q = pow(math.e, 2)
                pi = pow(math.e, -3)
                contact_area = 2000*pow(math.e, -1)
            elif k_total == 6:
                rho = pow(math.e, 0)
                d = pow(math.e, -3)
                q = pow(math.e, 0)
                pi = pow(math.e, -1)
                contact_area = 2000*pow(math.e, -1)
        else: # UIHC whole graph
            if k_total == 2:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 1)
                pi = pow(math.e, -1)
                contact_area = 2000
            elif k_total == 4:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 1)
                pi = pow(math.e, -1)
                contact_area = 2000
            elif k_total == 6:
                rho = pow(math.e, -1)
                d = pow(math.e, -3)
                q = pow(math.e, 2)
                pi = pow(math.e, -2)
                contact_area = 2000
    return rho, d, q, pi, contact_area

def set_simulation_parameters_may_not_be_an_additional_infection(args, k_total):
    name = args.name
    sampled = args.sampled
    assert name in ["Karate_temporal", "UIHC_HCP_patient_room_withinHCPxPx"]
    assert k_total in [2, 4, 6]
    if name == "Karate_temporal":
        if k_total == 2:
            rho = pow(math.e, 0)
            d = pow(math.e, -4)
            q = pow(math.e, 0)
            pi = pow(math.e, -2)
            contact_area = 2000*pow(math.e, -1)
        elif k_total == 4:
            rho = pow(math.e, 0)
            d = pow(math.e, -3)
            q = pow(math.e, -2)
            pi = pow(math.e, 0)
            contact_area = 2000*pow(math.e, -2)
        elif k_total == 6:
            rho = pow(math.e, -1)
            d = pow(math.e, -3)
            q = pow(math.e, 2)
            pi = pow(math.e, -4)
            contact_area = 2000
    elif name == "UIHC_HCP_patient_room_withinHCPxPx":
        if sampled: # UIHC sampled graph
            if k_total == 2:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 0)
                pi = pow(math.e, -1)
                contact_area = 2000
            elif k_total == 4:
                rho = pow(math.e, -2)
                d = pow(math.e, -4)
                q = pow(math.e, -2)
                pi = pow(math.e, 0)
                contact_area = 2000*pow(math.e, -2)
            elif k_total == 6:
                rho = pow(math.e, -3)
                d = pow(math.e, -4)
                q = pow(math.e, 0)
                pi = pow(math.e, -2)
                contact_area = 2000*pow(math.e, -4)
        else: # UIHC whole graph
            if k_total == 2:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 1)
                pi = pow(math.e, -1)
                contact_area = 2000
            elif k_total == 4:
                rho = pow(math.e, -1)
                d = pow(math.e, -4)
                q = pow(math.e, 1)
                pi = pow(math.e, -1)
                contact_area = 2000
            elif k_total == 6:
                rho = pow(math.e, -1)
                d = pow(math.e, -3)
                q = pow(math.e, 1)
                pi = pow(math.e, -1)
                contact_area = 2000
    return rho, d, q, pi, contact_area
