
#task 1
import random
win=0
lost=0
print("Lets start the game")
a=input("Press Enter to start rolling the dice")
for i in range(1,501):
    d1=random.randint(1,6)
    d2=random.randint(1,6)
    sum_dice= d1+d2
    """
    print("Sum of numbers that you got on dice 1 and dice 2 are:",d1,"and",d2,"=",sum_dice)"""
    if sum_dice in [7,11]:
        win+=1
    elif sum_dice in [2,3,12]:
       lost+=1
    else:
        value=0
        while value!= sum_dice and value!=7:
            dice1=random.randint(1,6)
            dice2=random.randint(1,6)
            value= dice1+dice2
        if value==7:
            lost+=1
        elif value==sum_dice:
            win+=1

print("You won:",win)
print("you lost:",lost)
print("Probility of winning:",win/500)
     
        
#task2
import random
duplicate_count=0
for n in range(5,101,5):
    duplicate_count=0
    for x in range(1,26):
        lst1=random.choices(range(1,365),k=n)
        lst2=set(lst1)
        if len(lst1)==len(lst2):
           duplicate_count=duplicate_count
        elif len(lst1)!=len(lst2):
            duplicate_count+=1
    probility=duplicate_count/25
    print("Duplictes are:",duplicate_count)
    print("Probility is:",probility)
    

    


