import re
def print2DArray(T):
    for r in T:
        for c in r:
            print(c,end = " ")
        print()
        
def equationsPossible(equationsStr):
    reachable = dict()
    truT =[ [-1]*5 for i in range(5)]
    stArr = equationsStr.split(", ")
    for statement in stArr: 
        result = re.search("([a-z])(.*)([a-z])", statement)
        # print(truT)
        # print("")
        indexX = ord(result.group(1)) - 97
        indexY = ord(result.group(3)) - 97
    
        print(statement)
        
        if (result.group(2) == "=="):
            if (result.group(1) not in reachable):
                reachable[result.group(1)] = []
            if (result.group(3) not in reachable):
                reachable[result.group(3)] = []
            reachable[result.group(1)].append(result.group(3))
            reachable[result.group(3)].append(result.group(1))
        if (result.group(2) == "!="):
            reach = set()
            if (result.group(1) in reachable):
                for letter in reachable[result.group(1)]:
                    reach.add(letter)
                    if (letter in reachable):
                        for letter_friend in reachable[letter]:
                            reach.add(letter_friend)
            if result.group(3) in reach:
                return False
        # print(indexX)
        # print(indexY)
        # print(result.group(2)) 
        # if (result.group(2) == "=="):
        #     if (truT[indexX][indexY] == -1):
        #         truT[indexX][indexY] = 1
        #         # print(result.group(1) + " " + result.group(2) + " " + result.group(3))
        #         # print(truT[indexX][indexY])
        #         # print2DArray(truT)
        #     if (truT[indexY][indexX] == -1):
        #         truT[indexY][indexX] = 1
        #         # print(result.group(1) + " " + result.group(2) + " " + result.group(3))
        #         # print(truT[indexY][indexX])
        #         # print2DArray(truT)
        #     if (truT[indexX][indexY] == 0):
        #         return False
        #     if (truT[indexY][indexX] == 0):
        #         return False
        # if (result.group(2) == "!="):
        #     if (truT[indexX][indexY] == -1):
        #         truT[indexX][indexY] = 0
        #         # print(result.group(1) + " " + result.group(2) + " " + result.group(3))
        #         # print(truT[indexX][indexY])
        #         # print2DArray(truT)
        #     if (truT[indexY][indexX] == -1):
        #         truT[indexY][indexX] = 0
        #         # print(result.group(1) + " " + result.group(2) + " " + result.group(3))
        #         # print(truT[indexY][indexX])
        #         # print2DArray(truT)
        #     if (truT[indexX][indexY] == 1):
        #         return False
        #     if (truT[indexY][indexX] == 1):
        #         return False
    # place your code here
    return True

print(equationsPossible("a==b, b!=a"))