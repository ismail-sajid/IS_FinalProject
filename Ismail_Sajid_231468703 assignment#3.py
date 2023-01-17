#Name: Ismail Sajid
#Roll No: 231468703
print("Simulating Scheduled CPU Jobs")
from random import *
import random
class PriorityQueueBase:
    class _Item:
        __slots__ = '_key', '_value'

        def __init__(self, k, v):
            self._key = k
            self._value = v

        def __lt__(self, other):
            return self._key < other._key  
        def __repr__(self):
           return '({0},{1})'.format(self._key, self._value)
    def is_empty(self):  
        return len(self) == 0

class HeapPriorityQueue(PriorityQueueBase):
    def _parent(self, j):
        return (j - 1) // 2

    def _left(self, j):
        return 2 * j + 1

    def _right(self, j):
        return 2 * j + 2

    def _has_left(self, j):
        return self._left(j) < len(self._data)  

    def _has_right(self, j):
        return self._right(j) < len(self._data)
    def _swap(self, i, j):
        """Swap the elements at indices i and j of array."""
        self._data[i], self._data[j] = self._data[j], self._data[i]

   
    def _upheap(self, j):
        parent = self._parent(j)
        if j > 0 and self._data[j] < self._data[parent]:
            self._swap(j, parent)
            self._upheap(parent)  
            

    def _downheap(self, j):
        if self._has_left(j):
            left = self._left(j)
            small_child = left  
            if self._has_right(j):
                right = self._right(j)
                if self._data[right] < self._data[left]:
                    small_child = right
            if self._data[small_child] < self._data[j]:
                self._swap(j, small_child)
                self._downheap(small_child)
       
    def __init__(self):
        
        self._data = []

    def __len__(self):
       
        return len(self._data)

    def add(self, key, value):
       
        self._data.append(self._Item(key, value))
        self._upheap(len(self._data) - 1)  # upheap newly added position

    def min(self):
        if self.is_empty():
            raise Empty('Priority queue is empty.')
        item = self._data[0]
        return (item._key, item._value)

    def remove_min(self):
        """Remove and return (k,v) tuple with minimum key.

        Raise Empty exception if empty.
        """
        if self.is_empty():
            raise Empty('Priority queue is empty.')
        self._swap(0, len(self._data) - 1)  # put minimum item at the end
        item = self._data.pop()  # and remove it from the list;
        self._downheap(0)  # then fix new root
        return (item._key, item._value)


def jobpriority():
    c=random.randint(-20,19)
    return(c)
def jobid():
    first=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    x=random.randint(0,10)
    y=random.randint(0,10)
    z=random.randint(0,10)
    name=first[randint(0,25)]+first[randint(0,25)]+first[randint(0,25)]+str(x)+str(y)+str(z)
    return(name)
def randomlength():
    b=random.randint(1,100)
    return(b)
    
    
def main():
    CPU= HeapPriorityQueue()
    JOB= HeapPriorityQueue()
    totaljobs=0
    time_for_process_job=0
    for i in range(10080):
        a=random.random()
        if a<0.04:
            CPU.add(jobpriority(),jobid())
            totaljobs+=1
        elif a>0.04 and a<1:
           pass
    for y in range(10080):
        if CPU.__len__()!=0:
            job=CPU.remove_min()
            length=randomlength()
            JOB.add(job[1],length)
            print("The I'd of the current running job:",job[1])
            time_for_process_job+=length
    print("Average waiting time:",time_for_process_job/(totaljobs-1))
    print("Jobs left in priority queue:",CPU.__len__())
            
                      
main()       
