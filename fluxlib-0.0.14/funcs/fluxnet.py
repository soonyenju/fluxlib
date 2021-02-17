import pickle
import multiprocessing
import numpy as np
import pandas as pd
from functools import reduce
from pathlib import Path
# from pysy.toolbox.utils import Yaml, unzip, create_all_parents, pbar
from scitbx import Yaml, unzip, create_all_parents, pbar

class Fluxnet2015(object):
    def __init__(self, path_cfg):
        fyaml = Yaml(path_cfg)
        self.config = fyaml.load()
        self.directories = self.config["directories"]
        self.params = self.config["params"]
        self.root = self.directories["workspace"] # root dir for all fluxnet2015 data
        if not isinstance(self.root, Path):
            self.root = Path(self.root)

    def untar(self):
        # initialize multicores
        cores = multiprocessing.cpu_count()
        pool  = multiprocessing.Pool(processes=cores)

        # unzip use multiprocessors 
        src_dir = self.root.joinpath(self.directories["zipped_flux_folder"])
        dst_dir = self.root.joinpath(self.directories["upzipped_folder"])
        create_all_parents(dst_dir)
        paths = list(Path(src_dir).glob(r"*.zip"))
        lengs = len(paths)
        out_dirs = [dst_dir for leng in range(lengs)]
        print(f"core number is: {cores}")
        pool.starmap(unzip, zip(paths, out_dirs))
        print(f"all fluxnet data are unzipped...")
    
    def __get_meta(self):
        # generate necessary info, extract site ID, longitude, latitude as a dataframe
        meta_dir = self.root.joinpath(self.directories["meta_file"])
        meta = pd.read_excel(meta_dir, usecols = ["SITE_ID", "VARIABLE", "DATAVALUE"])
        lats = meta[meta["VARIABLE"] == "LOCATION_LAT"]
        lats = lats.rename(columns={"DATAVALUE": "LAT"})
        lons = meta[meta["VARIABLE"] == "LOCATION_LONG"]
        lons = lons.rename(columns={"DATAVALUE": "LON"})
        igbps = meta[meta["VARIABLE"] == "IGBP"]
        igbps = igbps.rename(columns={"DATAVALUE": "IGBP"})
        meta = reduce(
            lambda left, right: pd.merge(
                left, right, on = ["SITE_ID"],
                how = "outer"
            ), [igbps, lats, lons]
        )
        # meta = pd.merge(lats, lons, on='SITE_ID')
        self.meta = meta.loc[:, ["SITE_ID", "IGBP", "LAT", "LON"]]
        # print(meta)
        # site_ids = meta["SITE_ID"].values
        # print(site_ids)
        print(f"meta data was accessed...")

    def search_file(self):
        freq = self.params["freq"]
        dst_dir = self.root.joinpath(self.directories["upzipped_folder"])
        paths = Path(dst_dir).rglob("*.csv")

        if self.params["site_name"]:
            if not isinstance(self.params["site_name"], list):
                self.params["site_name"] = [self.params["site_name"]]
            paths = [p for p in paths if p.as_posix().split("_")[1] in self.params["site_name"]]

        temp_paths = []
        for p in paths:
            temp = p.stem.split("_")
            if freq == "HH":
                if "HH" in temp or "HR" in temp: # HR rather than HH in US-UMB
                    temp_paths.append(p)
            if freq == "DD":
                if "DD" in temp:
                    temp_paths.append(p)
            if freq == "MM":
                if "MM" in temp:
                    temp_paths.append(p)
        # paths comprise ERAI and FULLSET in order:
        paths = temp_paths
        site_names_in_paths = np.array([Path(p).stem.split("_")[1] for p in paths]) # duplicates should remain
        # print(site_names_in_paths) # duplicates exist, e.g., ['AR-SLu' 'AR-SLu' 'AR-Vir' 'AR-Vir']
        print(f"data files are searched...")

        data_dict = {}

        # iterate each row (each site) of meta which is a dataframe
        self.__get_meta()
        for index, row in self.meta.iterrows():
            if (self.params["site_name"]) and (row["SITE_ID"] not in self.params["site_name"]):
                continue
            if row["SITE_ID"] not in site_names_in_paths:
                continue
            print(index, row["SITE_ID"])
            si = row["SITE_ID"]
            data_dict[si] = {}
            data_dict[si]["IGBP"] = row["IGBP"]
            data_dict[si]["LON"] = row["LON"]
            data_dict[si]["LAT"] = row["LAT"]
            idxs = np.where(site_names_in_paths == si)[0]
            data_dict[si]["PATHS"] = [paths[i].as_posix() for i in idxs]
            # print(data_dict)
            # exit(0)

        site_info_dir = self.root.joinpath(self.directories["site_info"])
        create_all_parents(site_info_dir, flag = "f")
        fyaml = Yaml(site_info_dir)
        fyaml.dump(data_dict)
        print(f"fluxnet files were iterated...")
        
    def retrieve(self):
        site_names = self.params["site_name"]
        savepath = self.root.joinpath(self.directories["savepath"])
        site_info_dir = self.root.joinpath(self.directories["site_info"])
        fyaml = Yaml(site_info_dir)
        info = fyaml.load()

        create_all_parents(savepath, flag = "d")

        if not site_names:
            site_names = list(info.keys())

        vars_req = self.params["vars"]
        # read each fluxnet file
        records = {}
        for count, (site_name, cell) in enumerate(info.items()):
            if not site_name in site_names:
                continue
            print(site_name)
            igbp = cell["IGBP"]
            lat = cell["LAT"]
            lon = cell["LON"]
            paths = cell["PATHS"]
            data = pd.read_csv(paths[1]) # fullset data
            # # Read ERA met params:
            # aux = pd.read_csv(paths[0]) # ERAI auxiliary data
            # timestamps = data["TIMESTAMP_START"].values
            # aux = aux.loc[aux["TIMESTAMP_START"].isin(timestamps)].reset_index(drop = True)
            # check the variable names
            # print(aux.columns.values)
            # print(data.columns.values)
            try:
                # print the time range of each fluxnet data
                print(Path(paths[1]).stem.split("_")[-2])
            except Exception as e:
                print(e)
                print(paths)
            vars_req_temp = [v for v in vars_req if v in data.columns]
            df = data[vars_req_temp]
            pbar(count, len(info.items()))
            item = {
                "igbp": igbp,
                "lat":  lat,
                "lon": lon,
                "values": df
            }
            records[site_name] = item
            df.to_csv(savepath.joinpath(f'{site_name}_{igbp}_LAT_{lat}_LON_{lon}.csv'))
        if "." in savepath.stem:
            create_all_parents(savepath, flag = "f")
            with open(savepath, "wb") as f:
                pickle.dump(records, f, protocol = pickle.HIGHEST_PROTOCOL)
            print(f"required records are exported to {savepath}...")

        # else:
        #     create_all_parents(savepath, flag = "d")
        #     for site_id, record in records.items():
        #         record["values"].to_csv(savepath.joinpath(f'{site_id}_{record["igbp"]}_LAT_{record["lat"]}_LON_{record["lon"]}.csv'))
