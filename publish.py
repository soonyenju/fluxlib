import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "publish a package with parameters")
    parser.add_argument('--pipreq', type = bool,
                        default = False, help='if check pip requirements using pipreq')
    parser.add_argument('--clean', type = bool,
                        default = True, help='remove directory build, dist and pysy.egg-info')
    args = parser.parse_args()
    if args.pipreq:
        print("checking requirements...")
        os.system("pipreqs ./ --encoding=utf-8")
    # publish_folders = [
    #     "build",
    #     "dist",
    #     "pysy.egg-info"
    # ]
    print("checking dirs...")
    cur_dirs = os.listdir()
    # build   
    os.system("python setup.py sdist bdist_wheel")
    print("package is built...")
    # push
    os.system("twine upload --repository-url https://upload.pypi.org/legacy/ dist/*")
    print("package is publised...")
    if args.clean:
        print("clearing up...")
        new_dirs = [p for p in os.listdir() if p not in cur_dirs]
        # new_dirs = publish_folders
        for p in new_dirs:
            # print(p)
            shutil.rmtree(p)
    print("all done.")