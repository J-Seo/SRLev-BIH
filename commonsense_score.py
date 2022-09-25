import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import argparse

from hclt2022.SRLev_BIH import srlev_bih

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main(args):
    print(args.is_mean)
    srlev, bih, srl_bih = srlev_bih(args.hypothesis_file, args.reference_file, is_mean=args.is_mean)
    print("SRL 점수:", srlev)
    print("BIH 점수:", bih)
    print("SRLev-BIH 점수:", srl_bih)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_mean", required=False, type=boolean_string, default=False)
    parser.add_argument("--reference_file", required=True, type=str)
    parser.add_argument("--hypothesis_file", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--own_task_name", required=True, type=str)
    parser.add_argument("--do_predict", required=True, type=bool)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()
    main(args)
