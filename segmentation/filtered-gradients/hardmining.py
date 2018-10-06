

def linearModifier(curEpoch, endEpoch):
    # epoch must be 0 indexed
    if curEpoch > endEpoch:
        return 1
    else:
        return curEpoch / endEpoch