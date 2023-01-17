ff = open("dataset_sunny.csv", "r")
data= ff.readlines()
def classifier():
    
    for i in data:
        dataset= i.split()
        print(dataset)
        
        if dataset==['sunny,hot,high,FALSE,no']:
            print("hello")
            

classifier()

