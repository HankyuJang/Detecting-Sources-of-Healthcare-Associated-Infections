# Detecting Sources of Healthcare Associated Infections

__Paper__: Hankyu Jang, Andrew Fu, Jiaming Cui, Methun Kamruzzaman, B. Aditya Prakash, Anil Vullikanti, Bijaya Adhikari and Sriram Pemmaraju, "Detecting Sources of Healthcare Associated Infections," _AAAI Conference on Artificial Intelligence. (AAAI 2023)._ Feb 2023, Washington, DC

Follow the instructions here to replicate experiments on the Carilion dataset, which is publicly available data.

## Software

If using anaconda, create the environment from `source_detection.yml`, then activate the environment.
Do the folloiwng:
```
conda env create -f source_detection.yml
conda activate source_detection
```

Otherwise, install the packages manually. We use Python3. Install the following Python libraries:

```
networkx
pandas
numpy
matplotlib
tqdm
```

## Public Data

- Carilion: public hospital data

## Private Data from two Hospitals

The private data from these hospitals cannot be shared in public due to privacy issue. We use these private hospital data in our experiments, because it has rich information (detailed contact information on patients, HCPs, and locations as well as the hospital graph structure) that is not available in public alternatives. Note that these detailed contact information is needed to better evaluate the performance of our proposed methods.
 
- UIHC
- UVA

# Replicate experiments in Carilion data

## Preprocessing

- Move the folder `G_Carilion/` to the parent directory of this repository. This folder contains a subfolder `graph_w_edge_attributes_v2/` which contains Carilion graphs (in .graphml format).

## Parameter tuning 

Simulation parameter set that yields 2-10% infection in the graph is saved as
`tables/parameter_tuning/G_Carilion_k{}.csv".format(graph_name, k_total)` where `k_total` here will be `2*args.seeds_per_t`. 

NOTE: there's no need to run `parameter_tuning_others.py`. But if you'd like to run it, then go to `src/` then run the following:
```
python parameter_tuning_others.py -name G_Carilion -seeds_per_t 1
python parameter_tuning_others.py -name G_Carilion -seeds_per_t 3
```

## Experiments

The ground truth observation preparation, running random baseline, and running all our algorithms can be done by executing this bash script:

```
./AAAI_exp_G_Carilion_GT_B_P1_P2_P3.sh
```

Then, execute the following bash script to run baseline methods

```
./AAAI_Carilion_on_cult_reachability_LOS.sh
```

## Generating result table and plots

Run the following script to generate the figures and table

```
./AAAI_Carilion_final_exp_F1best_result.sh
```
