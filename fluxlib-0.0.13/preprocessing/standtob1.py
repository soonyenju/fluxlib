# coding: utf-8

# author: sz394@exeter.ac.uk
# date: 22/10/2019

# description: this code sorts tob1 filenames and stores sorted into a text file.
# usage: Make_EdiRe_dirlist.py [-f] [--t]
#     -f: directory of folder containing input tob1s, the default is inputs;
#     --t: output text file directory, the default is output.txt;
# example: python Make_EdiRe_dirlist.py G:\workspace\Converted_Roth\Roth_1_N --t=RothN.txt --c=2000000
# python Make_EdiRe_dirlist.py G:\workspace\Roth\Converted_north\2017_11_02 --t=2017_11_02.txt --action=remove
# python Make_EdiRe_dirlist.py G:\workspace\Roth\Converted_north\2017_11_02 --t=2017_11_02.txt --action=daynight

import os, re, multiprocessing, argparse, shutil
from pathlib import Path

class EdiReObj(object):
    def __init__(self, tob1_folder_path):
        super()
        self.folder = tob1_folder_path

    def sort(self, t_dir, c, pool):
        print("ready to go...")
        # iterate the folder to find tob1 files.
        paths = Path(self.folder).glob(r"*.dat")
        # paths = list(paths) # if iterator doesnot work, uncomment this line.
        print("tob1 files iterated successfully...")

        # have a list of lists derived from the function split_filename
        list_of_splits = pool.map(split_filename, paths)
        # remove the filenames of which the size is smaller than the criterion.
        list_of_splits = list(filter(lambda x: x[6] >= c, list_of_splits))
        print("tob1 files loaded successfully...")
        # sort the filenames according to timestamps and only extract sorted filenames.
        list_of_splits.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5]))
        print("sorted successfully...")
        list_of_files = list(zip(*list_of_splits))[0]

        # write into files.
        with open(t_dir, 'w') as f:
            f.writelines("%s\n" % placeholder for placeholder in list_of_files)
        print("writting successfully, bye!")

    def remv_incoms(self, flag):
        '''
        remove the incomplete files from tob1 folder, incomplete files are those
        smaller than the max large or specified number.
        '''
        # iterate the folder to find tob1 files.
        paths = Path(self.folder).glob(r"*.dat")
        paths = list(paths) # generator cannot be used more than once.

        if flag == "max":
            print("Removing all incomplete files...")
            # fs_list = [p.stat().st_size for p in paths]
            fs_list = map(lambda p: p.stat().st_size, paths)
            max_size = max(fs_list)
        else:
            print("Removing user-defined size files specified by param c...")
            max_size = flag

        # codes here can be optimized!
        print(f"the max size is {max_size} bits.")
        for p in paths:
            if p.stat().st_size < max_size:
                print(f"{p.name} is being removed...")
                p.unlink() # remove file
                # p.rmdir() # remove the whole directory

    def daynight(self, sunrise, sunset, move):
        '''
        split files into diurnal and nocturnal, put them into different folders, respectively. 
        '''
        # iterate the folder to find tob1 files.
        paths = Path(self.folder).glob(r"*.dat")
        # create diurnal folder
        day_folder = Path.cwd().joinpath("day")
        day_folder.mkdir(parents=True, exist_ok=True)
        # create nocturnal folder
        night_folder = Path.cwd().joinpath("night")
        night_folder.mkdir(parents=True, exist_ok=True)

        # codes here can be optimized!
        print("start moving/copying day&night files....")
        for p in paths:
            pattern = r"\d{4}_\d{2}_\d{2}_\d{4}"
            results = re.findall(pattern, p.stem)
            if len(results) != 1:
                raise Exception("Wrong filename format!")
            _, _, _, HHMM = results[0].split("_")
            HH = int(HHMM[0: 2])
            if HH > sunrise and HH < sunset:
                # day
                to_file = day_folder.joinpath(p.name)
            else:
                to_file = night_folder.joinpath(p.name)
            if move == False:
                shutil.copy(p.as_posix(), to_file.as_posix())
            else:
                os.rename(p.as_posix(), to_file.as_posix())
        print("mission accomplised!")


    def __del__(self):
        pass

def split_filename(p):
    '''
    Use regex to find the timestamp information in one tob1 file,
    and split into YYYY, mm, dd, HH, MM, respectively.
    Return a list including filename and splitted time info.
    '''
    filesize = p.stat().st_size
    filename = p.as_posix()
    pattern = r"\d{4}_\d{2}_\d{2}_\d{4}"
    results = re.findall(pattern, filename)
    if len(results) != 1:
        raise Exception("Wrong filename format!")
    YYYY, mm, dd, HHMM = results[0].split("_")
    YYYY = int(YYYY)
    mm = int(mm)
    dd = int(dd)
    HH = int(HHMM[0: 2])
    MM = int(HHMM[2::])
    return [filename, YYYY, mm, dd, HH, MM, filesize]

def main() :
    # define command line params.
    parser = argparse.ArgumentParser(description = "please specify directories of folder containing to be sorted tob1 files and of output txt file.")
    parser.add_argument('f', type=str, default = "inputs",
                        help="directory of folder containing input tob1s.")

    parser.add_argument('--t', type=str, default = "output.txt",
                        help="output text file directory.")

    parser.add_argument('--c', type=int, default = 10, #bits
                        help="criterion")

    parser.add_argument('--action', type=str, default = "sort",
                        help="action to do with tob1s")

    parser.add_argument('--sunrise', type=int, default = 6,
                        help="time of sunrise")

    parser.add_argument('--sunset', type=int, default = 18,
                        help="time of sunset, 24-hour system")

    parser.add_argument('--move', type=bool, default = False,
                        help="if move or copy day/night tob1 files")


    args = parser.parse_args()
    
    # transform Windows directories into posix
    if "\\" in args.f:
        folder = args.f.replace("\\", "/")
    else:
        folder = args.f

    if "\\" in args.t:
        t_dir = args.t.replace("\\", "/")
    else:
        t_dir = args.t

    # check the capability of multiprocessing and create the pool.
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cores)

    # initialize object
    eddy = EdiReObj(folder)
    if args.action == "sort":
        eddy.sort(t_dir, args.c, pool)

    elif args.action == "remove":
        if args.c == parser.get_default('c'):
            eddy.remv_incoms("max")
        else:
            eddy.remv_incoms(args.c)
    elif args.action == "daynight":
        eddy.daynight(args.sunrise, args.sunset, args.move)
        
    else:
        raise Exception("Not identified action!")



if __name__ == "__main__":
    main()