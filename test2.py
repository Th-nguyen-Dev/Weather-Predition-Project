class Result:
    def __init__(self, best_match, query_alignment, reference_alignment, score):
        self.best_match = best_match
        self.query_alignment = query_alignment
        self.reference_alignment = reference_alignment
        self.score = score
def find_best_match(query, references):
    # place your code 
    points = []
    results = []
    for reference in references:
        point = 0
        result=""
        for index, letter in enumerate(query):
            if (index >= len(reference)):
                point+=0
            elif (letter == reference[index]):
                point+=2
                result+="|"
            elif (reference[index] == '-'):
                point=-2
                result+="-"
            elif (letter != reference[index]):
                point-=1
                result+="."

        points.append(point)
        results.append(result)
    
    best_reference = references[points.index(max(points))] 
    best_match = results[points.index(max(points))]
    best_point = max(points)
    
    print(results)
    print(points)
    print(reference)
    
    return Result(best_match,query,best_reference,best_point)

obj = find_best_match("ARNDC", ["CCDAA", "RAAAA", "NDCR", "CAANAC"])
print(obj.best_match)
print(obj.query_alignment)
print(obj.reference_alignment)
print(obj.score)