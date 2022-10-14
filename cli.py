import argparse
import sys

import main
from usps import realtime_predict, train, weights_simplify, config

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--main",
                        "-m",
                        action='store_true',
                        help="minimal test")
arg_parser.add_argument("--usps-test",
                        "-test",
                        action='store_true',
                        help="start usps test")
arg_parser.add_argument("--usps-train",
                        "-train",
                        action='store_true',
                        help="start usps train")
arg_parser.add_argument("--usps-simplify",
                        "-s",
                        action='store_true',
                        help="simplify usps weights")
arg_parser.add_argument("--usps-preview",
                        "-p",
                        action='store_true',
                        help="preview usps realtime predict result")
arg_parser.add_argument("--usps-preview-file-name",
                        "-pf",
                        type=str,
                        default="",
                        help=f"usps realtime predict image name, default: {config.default_img_name}")
arg_parser.add_argument("--pycharm",
                        "-pycharm",
                        action='store_true',
                        help="pycharm preview mode")
opt = arg_parser.parse_args()

if (len(sys.argv) == 1):
    print("Use cli.py -h to get more help")
    exit()

if opt.main:
    main.run()
elif opt.usps_preview or (opt.usps_preview_file_name != ""):
    realtime_predict.run(opt.usps_preview_file_name, opt.pycharm)
elif opt.usps_train:
    train.run(True)
elif opt.usps_test and not opt.usps_simplify:
    print("Test accuracy:")
    train.run(False)

if opt.usps_simplify:
    weights_simplify.run()
    print("Simplifying weights finished")
