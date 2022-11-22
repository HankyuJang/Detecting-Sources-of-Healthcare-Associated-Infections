"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last modified: Nov 2021

Description: This script generates plots for checking ISCK guarantee within iterations
"""

import argparse
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def read_files(infile):
    npzfiles = np.load(infile)
    array_P_hit = npzfiles["array_P_hit"]
    array_N_hit = npzfiles["array_N_hit"]
    array_P_hit_frac = npzfiles["array_P_hit_frac"]
    array_N_hit_frac = npzfiles["array_N_hit_frac"]
    array_MCC = npzfiles["array_MCC"]
    npzfiles.close()
    return array_P_hit, array_N_hit, array_P_hit_frac, array_N_hit_frac, array_MCC

def line_plot(array, xlabel, ylabel, title, outfile):

    fig, ax = plt.subplots()
    ax.plot(array)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig.savefig(outfile)
    plt.close()

def scatter_plot_on_loss(x, y, xlabel, ylabel, title, name, filename):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    #############################################
    # start: https://stackoverflow.com/a/25497638/4595935
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    # end: https://stackoverflow.com/a/25497638/4595935
    #############################################
    fig.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
            help= 'network to use: Karate_temporal | UIHC_HCP_patient_room_2011 | UIHC_HCP_patient_room_2011_sampled')
    args = parser.parse_args()
    name = args.name

    label_list = ["P_hit", "N_hit", "P_hit_frac", "N_hit_frac", "MCC"]
    for k in [1,2,3]:
        infile = "../npz/ISCK_approx_check/{}/arrays_k{}.npz".format(name, k)
        array_list = read_files(infile)

        for label, array in zip(label_list, array_list):

            outfile = "../plots/ISCK_approx_check/{}/{}_k{}.png".format(name, label, k)
            line_plot(array, "iterations", label, "ISCK {} per iteration".format(label), outfile)


