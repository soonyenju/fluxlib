# read tob1 and ghg files.
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
import re

class Raw(object):
    """
    acknowledgement: USDA-ARS-USSL 
    citation: 
        T. H. Skaggs, R. G. Anderson, J. G. Alfieri, T. M. Scanlon, W. P. Kustas (2018). 
        Fluxpart: Open source software for partitioning carbon dioxide and water vapor fluxes. 
        Agricultural and Forest Meteorology 253--254:218--224, 
        doi:10.1016/j.agrformet.2018.02.019.
    url: https://github.com/usda-ars-ussl/fluxpart/blob/master/fluxpart/util.py
    """
    def __init__(self, data_folder, MUTE = True):
        super().__init__()
        # initialize params: {
        self.MUTE = MUTE
        self.TYPE_MAP = {
            "IEEE4": np.float32,
            "IEEE8": np.float64,
            "LONG": np.int32,
            "ULONG": np.uint32,
        }
        # }

        # dir of data:
        self.data_folder = Path(data_folder)
    
    def __call__(self, data_type = "tob1"):
        if data_type == "tob1":
            if not self.MUTE: print("processing tob1 files...")
            paths = self.data_folder.glob(r"*.dat")
            dfs_generator = self.read_tob1(paths)
        elif data_type == "ghg":
            if not self.MUTE: print("processing ghg files...")
            paths = self.data_folder.glob(r"*.ghg")
            dfs_generator = self.read_ghg(paths)
        else:
            raise Exception("Wrong flux data type.")
        return dfs_generator

    def read_tob1(self, tobfiles, count = -1):
        """read tob1 files to dataframes."""
        for tobfile in tobfiles:
            with open(tobfile, "rb") as f:
                f.readline()
                names = f.readline().decode().strip().replace('"', "").split(",")
                f.readline()
                f.readline()
                types = f.readline().decode().strip().replace('"', "").split(",")
                dtype = np.dtype([(n, self.TYPE_MAP[t]) for n, t in zip(names, types)])
                array = np.fromfile(f, dtype=dtype, count=count)
                df = pd.DataFrame(array)
                yield df

    def read_ghg(self, ghg_files, *args, **kwargs):
        """Buffered pd.read_csv of data split across multiple files."""
        for ghg_file in ghg_files:
            with zipfile.ZipFile(ghg_file) as z:
                meta_basename = ghg_file.with_suffix(".metadata").name
                try:
                    with z.open(meta_basename) as f:
                        lines = f.readlines()
                        # convert string to dict:{
                        lines = [line.decode().strip().split("=") for line in lines if b"=" in line]
                        # use results to make a dict
                        lines = dict(lines)

                        skip_rows = int(lines["header_rows"]) - 1
                        sep = lines["separator"]
                        if sep == "tab":
                            sep = "\t"
                        else:
                            sep = ","
                        # }
                except Exception as e:
                    if not self.MUTE: print(e)
                basename = ghg_file.with_suffix(".data").name
                try:
                    df = pd.read_csv(z.open(basename), sep = sep, skiprows = skip_rows, *args, **kwargs)
                except Exception as e:
                    if not self.MUTE: print(e.args[0])
                    continue
            if isinstance(df, pd.io.parsers.TextFileReader):
                yield from df
            else:
                yield df   


# """
# example:
# """
# def main():
#     path = "data_ghg"
#     flux_read = Raw(path, MUTE = False)
#     dfs = flux_read(data_type = "ghg")
#     for i, df in enumerate(dfs):
#         # print(df)
#         df.to_csv("ghg_" + str(i).zfill(3) + ".csv")

# if __name__ == "__main__":
#     main()