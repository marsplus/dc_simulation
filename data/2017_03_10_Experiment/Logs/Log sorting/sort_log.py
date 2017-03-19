#-------------------------------------------------------------------------------
# Name:        sort_log
# Purpose:     Sort an experiment log
#
# Author:      Zlatko
#
# Created:     07.07.2016
# Copyright:   (c) Zlatko 2016
#-------------------------------------------------------------------------------

import csv
import datetime
import re
import operator
import numpy as np

import helpers

# prefix = "C:/Users/PC4all/Dropbox/Decentralized Coordination/Experiments/Adversarial/07.07.2016 (Experiment)/Logs/Log sorting/"
prefix = "./"

exp_label = "03.10.17_log_unsorted"

def read_experiment_log(label):
    # open the experiment log .txt file and read the lines into a list
    with open(prefix+label+".txt") as f:
        content = f.readlines()

    # remove trailing '\n' characters from lines
    content = [x.strip('\n') for x in content]

    listOfEntries = []
    index = 0

    while (index < len(content)):
        index = helpers.getNextEntry(content, index, listOfEntries)

    listOfEntries = sorted(listOfEntries, key=lambda entry: entry['timestamp'])

    # deal with 'around midnight' issue
    min_time = listOfEntries[0]['timestamp']
    max_time = listOfEntries[-1]['timestamp']
    if (min_time < helpers.timeDelta("01:00:00.00") or max_time > helpers.timeDelta("23:00:00.00")):
        for i in range(len(listOfEntries)):
            listOfEntries[i]['timestamp'] = listOfEntries[i]['timestamp'] + helpers.timeDelta("12:00:00.00")

            # if adding an offset of 12 hrs takes us to a +1 day datetime object, subtract 24 hrs
            if (listOfEntries[i]['timestamp'] >= helpers.timeDelta("24:00:00.00")):
                listOfEntries[i]['timestamp'] = listOfEntries[i]['timestamp'] - helpers.timeDelta("24:00:00.00")

    listOfEntries = sorted(listOfEntries, key=lambda entry: entry['timestamp'])

    return listOfEntries

def main():
    listOfEntries = read_experiment_log(exp_label)

    f = open('03.10.17_log_sorted.txt', 'w')

    for entry in listOfEntries:
        f.write("###\n")
        f.write(helpers.stringify_timedelta(entry['timestamp']) + "\n")
        f.write(entry['type'] + "\n")

        for line_of_content in entry['content']:
            f.write(line_of_content + "\n")

        f.write("###\n")

    f.close()

if __name__ == '__main__':
    main()
