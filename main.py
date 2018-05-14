# -*- coding:utf-8 -*-
import sys
import getopt
from base.model import model
from base.read_data import draw_mat

def main(argv):
    inputfile = ''
    outputfile = ''
    checkpoint = '999'
    try:
        opts, args = getopt.getopt(argv,"m:r:c",["mode=","rate=","checkpoint"])
    except getopt.GetoptError:
        print('test.py -m/--mode <mode> -r/--rate <rate>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-r", "--rate"):
            rate = arg
        elif opt in ("-c", "--checkpoint"):
            checkpoint = arg
    

    model(mode,rate,checkpoint)

if __name__ == "__main__":
   main(sys.argv[1:])