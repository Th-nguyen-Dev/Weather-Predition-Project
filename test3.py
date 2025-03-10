def smallestRectangle(arr):
    # place your code here
    oneX = []
    oneY = []
    for rowindex, row in enumerate(arr):
        for colindex, col in enumerate(row):
            if (arr[rowindex][colindex] == 1):
                oneY.append(rowindex)
                oneX.append(colindex)
    
    maxX = max(oneX)
    minX = min(oneX)
    maxY = max(oneY)
    minY = min(oneY)
    
    print("X: ",oneX)
    print("Y: ",oneY)
    
    print("X")
    print(minX, " ", maxX)
    
    print("Y")
    print(minY, " ", maxY)
    
    return [maxX-minX + 1, maxY - minY +1]   
def printArray(arr):
    for row in arr:
        print(" ".join(map(str, row)))

arr = [
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 1, 0, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
]
print(smallestRectangle(arr))