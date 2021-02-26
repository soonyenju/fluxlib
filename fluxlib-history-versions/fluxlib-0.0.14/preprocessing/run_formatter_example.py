from formatize import Formatter
# formatting
format_cfg = "format_cfg_example.yaml"
# Can use ERA5 parameters
formatter = Formatter(format_cfg, verbose = 2, tolerant = 0)
# save formatted df into csv and mds txt
formatter(handle = "stop", start_count = 0)