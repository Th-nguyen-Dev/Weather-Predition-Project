import numpy as np
# do not change/remove this class
class EventNode:
    def __init__(self, event_name, attendees):
        self.event_name = event_name
        self.attendees = attendees
        self.next = None

class Solution:
    @staticmethod
    def calculate_leaderboard(head : EventNode):
        # place your code here
        result = dict()
        while(head != None):
            for attendee in head.attendees:
                if attendee not in result:
                    result[attendee] = 0
                else:
                    result[attendee]+=1
            head = head.next
            
        print(result)
        sort_alpha = sorted(result.keys())
        print(sort_alpha)
        
        result_alpha = dict()
        for name in sort_alpha:
            result_alpha[name] = result[name]
            
        print(result_alpha) 
        
        reuslt_sort = {k: v for k, v in sorted(result_alpha.items(), key=lambda item: item[1], reverse=True)}
        # sort_score = {k: v for k, v in sorted(sort_alpha.items(), key=lambda item: item[1])}
        # print(sort_score)
        print(reuslt_sort)
        return list(reuslt_sort.keys())
    

coding_workshop = EventNode("Coding Workshop", ["Alice", "Bob", "Charlie"])
tech_talk = EventNode("Tech Talk", ["Charlie", "Alice", "David"])
hackathon = EventNode("Hackathon", ["Emma", "Charlie", "Alice"])

# Linking the nodes
coding_workshop.next = tech_talk
tech_talk.next = hackathon

# Calculating and printing the leaderboard
print("Test 1 Leaderboard:")
print(Solution.calculate_leaderboard(coding_workshop))