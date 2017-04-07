from collections import defaultdict

neighbors = [0, 2, 3, 5, 8, 9, 12, 13, 14, 16, 17]
with open('result/simResult.txt', 'rb') as fid:
	data = fid.readlines()
data = [item.decode('UTF-8').strip().split(',') for item in data]


ret = defaultdict(list)
visibleNodeDecision = {}
flag = False
k = None
for record in data:
    # first time encounter a decision record of visible node
    if not flag and record[-1] == 'vis':
        visibleNodeDecision[record[2]] = record[1]
        flag = True
        k = record[2]
    # encounter decision records made by non-visible nodes
    elif flag and not record[-1] == 'vis':
        if int(record[0]) in neighbors:
            ret[k].append(record)
    # second time encounter a decision record made by visible node
    elif flag and record[-1] == 'vis':
        visibleNodeDecision[record[2]] = record[1]
        k = record[2]





