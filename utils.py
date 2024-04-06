def stringify(val):
    return ','.join([str(x) for x in val])

def flatten(arr):
    return [i for j in arr for i in j]
