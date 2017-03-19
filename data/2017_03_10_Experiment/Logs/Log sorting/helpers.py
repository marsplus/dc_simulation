#-------------------------------------------------------------------------------
# Name:        helpers
# Purpose:
#
# Author:      Zlatko
#
# Created:     21.01.2016
# Copyright:   Zlatko 2016
#-------------------------------------------------------------------------------
import datetime
import re

# convert timedelta object to 'proper' string (hh:mm:ss.xxx)
def stringify_timedelta(t):
    time_parts = str(t).split('.')
    hms = time_parts[0]
    msc = "000"
    if len(time_parts) > 1:
        msc = time_parts[1][0:3]

    time_string = hms + "." + msc

    return time_string

# convert time string to timedelta object
def timeDelta(timeString):
    timeArray = re.split(':', timeString)

    hrs = int(timeArray[0])
    mns = int(timeArray[1])
    scs = float(timeArray[2])

    td = datetime.timedelta(hours=hrs, minutes=mns, seconds=scs)

    return td

# calculates time difference
def calculateTimeDifference(start, end):
    startTimeDelta = timeDelta(start)
    endTimeDelta = timeDelta(end)

    return endTimeDelta - startTimeDelta

# extract a single log entry
def getNextEntry(logContent, index, listOfEntries):
    newEntry = {}
    newEntry['timestamp'] = timeDelta(logContent[index+1])
    newEntry['type'] = logContent[index+2]

    entryContent = []
    index = index+3
    while (logContent[index] != '###'):
        entryContent.append(logContent[index])
        index = index + 1

    newEntry['content'] = entryContent

    listOfEntries.append(newEntry)

    return index+1

# extract adjacency matrux
def getAdjacencyMatrix(content):
    n = len(content)
    adj = [[False for x in range(n)] for x in range(n)]

    for i in range(0, n):
        row = re.split('\t', content[i])
        for j in range(0, n):
            if row[j] == "true":
                adj[i][j] = True

    return adj

# extract neighbors of nodes
def getNeighbors(adj):
    n = len(adj)
    neigh = []
    nodeNeigh = []

    for i in range(0, n):
        nodeNeigh = []

        for j in range(0, n):
            if adj[i][j]:
                nodeNeigh.append(j)

        neigh.append(nodeNeigh)

    return neigh

# calculate the sizes of node neighborhoods
def getNeighborhoodSizes(neigh):
    n = len(neigh)
    neighSizes = []

    for i in range(0, n):
        neighSizes.append(len(neigh[i]))

    return neighSizes

# extract structured message
def getStructuredMessage(node, colors, neigh):
    nodeNeigh = neigh[node]
    message = ""
    redCount = 0
    greenCount = 0

    for i in range(0, len(nodeNeigh)):
        if colors[nodeNeigh[i]] == 'red':
            redCount =  redCount + 1
        elif colors[nodeNeigh[i]] == 'green':
            greenCount = greenCount + 1

    if redCount >= greenCount:
        message = "RED " + str(redCount) + ", " + "GREEN " + str(greenCount)
    else:
        message = "GREEN " + str(greenCount) + ", " + "RED " + str(redCount)

    return message

# extract structured message type ('none' if color counts are equal, 'red'/'green' if the dominant color count is 'red'/'green'
def getStructuredMessageType(colors, neigh):
    redCount = 0
    greenCount = 0

    for i in range(len(neigh)):
        if (colors[neigh[i]] == 'red'):
            redCount = redCount + 1
        elif (colors[neigh[i]] == 'green'):
            greenCount = greenCount + 1

    mType = {'red': 0, 'green': 0}

    if (redCount >= greenCount):
        mType['red'] = 1
    if (greenCount >= redCount):
        mType['green'] = 1

    return mType

# get signed, combined, global ratio
# -1.0: all network nodes are red; +1.0 all network nodes are green
def getSignedGlobalRatio(colors):
    red_count = 0.0
    green_count = 0.0

    for i in range(0, len(colors)):
        if colors[i] == "red":
            red_count  = red_count + 1.0
        elif colors[i] == "green":
            green_count = green_count + 1.0

    signed_ratio = (green_count - red_count)/float(max(1,len(colors)))

    return signed_ratio

# get signed, global difference
# -# network nodes: all network nodes are red; +# network nodes all network nodes are green
def getSignedGlobalDifference(colors):
    red_count = 0.0
    green_count = 0.0

    for i in range(0, len(colors)):
        if colors[i] == "red":
            red_count  = red_count + 1.0
        elif colors[i] == "green":
            green_count = green_count + 1.0

    signed_difference = green_count - red_count

    return signed_difference

# get the ratio of nodes colored with 'color' and the total number of network nodes
def getGlobalRatio(color, colors):
    count = 0.0

    for i in range(0, len(colors)):
        if colors[i] == color:
            count  = count + 1

    return count/max(1, len(colors))

# get the ratio of neigbors colored with 'color' and the total number of neighbors
def getLocalRatio(nodeNeigh, color, colors):
    count = 0.0

    for i in range(0, len(nodeNeigh)):
        if colors[nodeNeigh[i]] == color:
            count = count + 1

    #if len(nodeNeigh) == 0:
     #   nb = raw_input('A problem appears here. Continue? ')

    return count/max(1, len(nodeNeigh))

# get signed, combined, local ratio
# -1.0: all neighboring nodes are red; +1.0 all neighboring nodes are green
def getSignedLocalRatio(nodeNeigh, colors):
    red_count = 0.0
    green_count = 0.0

    for i in range(0, len(nodeNeigh)):
        if colors[nodeNeigh[i]] == "red":
            red_count  = red_count + 1
        elif colors[nodeNeigh[i]] == "green":
            green_count = green_count + 1

    signed_ratio = (green_count - red_count)/max(1,len(nodeNeigh))

    return signed_ratio

# get signed, local difference
# -# neighborhood nodes: all neighboring nodes are red; +# neighborhood nodes all neighboring nodes are green
def getSignedLocalDifference(nodeNeigh, colors):
    red_count = 0.0
    green_count = 0.0

    for i in range(0, len(nodeNeigh)):
        if colors[nodeNeigh[i]] == "red":
            red_count  = red_count + 1
        elif colors[nodeNeigh[i]] == "green":
            green_count = green_count + 1

    signed_difference = green_count - red_count

    return signed_difference

# get normalized, signed, combined, local ratio
def getNormalizedSignedLocalRatio(nodeNeigh, colors):
    signed_ratio = getSignedLocalRatio(nodeNeigh, colors)

    norm_ratio = signed_ratio

    # normalize by multiplying with neigh_size/net_size
    if (len(colors) * len(nodeNeigh) > 0):
        norm_ratio = signed_ratio * len(nodeNeigh) / len(colors)

    return norm_ratio

# calculate marginal information difference (currently used only for structured messages)
def calculateMarginalDiff(marginal_info, signed_global_ratio):
    # if marginal_info points to the right 'direction' and magnitude is greater than signed_global_ratio
    # we set the difference to 0.0
    if ((marginal_info * signed_global_ratio > 0) and (abs(marginal_info) > abs(signed_global_ratio))):
        marginal_diff = 0.0
    else:
        marginal_diff = abs(marginal_info - signed_global_ratio)

    return marginal_diff


# convert string to have all upercase letters
def upper_repl(match):
     return match.group(0).upper()

# convert string to have all lowercase letters
def lower_repl(match):
     return match.group(0).lower()
