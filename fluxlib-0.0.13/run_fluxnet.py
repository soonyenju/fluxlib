import pickle
from fluxnet import Fluxnet2015
# from gapfill.gapfill_candi import Candidate
from pathlib import Path
import pandas as pd
# from pysy.toolbox.utils import Yaml
from scitbx import Yaml

def main():
    extract_data_frome_fluxdata()
    # gap_fill("gapfill/gapfill_candi_config.yaml")

# def gap_fill(config_file):
#     candidate = Candidate(config_file)
#     # candidate.preprocess()
#     candidate.dispatch()


def extract_data_frome_fluxdata():
    p = "fluxnet_config.yaml"
    fn = Fluxnet2015(p)
    # fn.untar()
    fn.search_file()
    fn.retrieve()
    # with open(r"output/fluxnet2015.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(data)



if __name__ == "__main__":
    main()