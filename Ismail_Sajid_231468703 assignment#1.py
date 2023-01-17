import random
class Animal:
    def __init__(self,gender=None,strength=None):
        
        if not gender:
            self._gender = random.choice(['M','F'])
        else:
            self._gender = gender
        if not strength:
            self._strength = random.random()
        else:
            self._strength = strength

        
    def get_gender(self):
        return self._gender
    def get_strength(self):
        return self._strength
class Bear(Animal):
    def __init__(self):
        super().__init__(gender=None,strength=None)
class Fish(Animal):
    def __init__(self):
        super().__init__(gender=None,strength=None)
class River:
    def __init__(self,length):
        self._length=length
        self.lst=[]
        for i in range(self._length):
            ran=random.random()
            if 0.02<ran<0.52:
                a="fish"
                self.lst.append(Fish())
                
            elif ran<0.02:
                b="bear"
                self.lst.append(Bear())
                
            else:
                self.lst.append(None)
      
    def move_index(self):
        mov=random.random()
        if mov>=0.50 and self.lst[i]!=None:
            move=random.randint(-1,1)
            if move != 0 and 0 <= i + move < self._length:
                if self.lst[i+move]==None:
                    self.lst[i+move]=self.lst[i]
                    self.lst[i]=None
                elif type(self.lst[i])==type(self.lst[i+move]):
                    if self.lst[i].get._gender()!=self.lst[i+move].get._gender():
                        if type(self.lst[i])==Fish:
                            self.lst.replace(None,Fish())
                        else:
                            self.lst.replace(None,Bear())
                    else:
                        if self.lst[i].get_strength() > self.lst[i+move1].get_strength():
                            self.lst[i+move]=self.lst[i]
                            self.lst[i]=None
                else:
                    if type(lst[i])==Bear:
                        self.lst[i+move]=self.lst[i]
                        self.lst[i]=None
                
    def cell(self):
        for i in range(len(self.lst)):
            self.move_index(i)
              

                
    def display(self):
        s = '|'
        for x in self.lst:
            if x:
                if type(x) == Bear:
                    s += 'B'
                elif type(x) == Fish:
                    s += 'F'
                s += str(x.get_strength())
                s += x.get_gender()
            else:
                s += '  '
            s += '|'
        print(s)        

eco=River(10000)
for i in range(5):
    print("Running:",i)
    eco.display()
