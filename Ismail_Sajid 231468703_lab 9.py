#task 1

class Tree:
    class Position:
        def element(self):
            raise NotImplementedError("must be implemented by subclass")
        def __eq__(self,other):
            raise NotImplementedError("must be implemented by subclass")
        def __ne__(self,other):
            return not (self == other)
    def root(self):
        raise NotImplementedError("must be implemented by subclass")
    def parent(self,p):
        raise NotImplementedError("must be implemented by subclass")
    def num_children(self,p):
        raise NotImplementedError("must be implemented by subclass")
    def children(self,p):
        raise NotImplementedError("must be implemented by subclass")
    def __len__(self):
        raise NotImplementedError("must be implemented by subclass")
    def is_root(self,p):
        return self.root()==p
    def is_leaf(self,p):
        return self.num_children(p)==0
    def is_empty(self):
        return len(self)==0
    def depth(self,p):
        if self.is_root(p):
            return 0
        else:
            return 1+self.depth(self.parent(p))
    def height1(self):
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))
    def height2(self,p):
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self.height2(c) for c in self.children(p))
    def height(self,p=None):
        if p is None:
            p=self.root()
        return self.height2(p)
    def __iter__(self):
        for p in self.position():
            yield p.element()
    def preorder(self):
        if not self.is_empty():
            for p in self.subtree_preorder(self.root()):
                yield p
    def subtree_preorder(self,p):
        yield p
        for c in self.children(p):
            for other in self.subtree_preorder(c):
                yield other
    def positions(self):
        return self.preorder()
    def postorder(self):
        if not self.is_empty():
            for p in self.subtree_postorder(self.root()):
                yield p
    def subtree_postorder(self, p):
        for c in self.children(p):
            for other in self.subtree_postorder(c):
                yield other
        yield p
    def breadthfirst(self):
        if not self.is_empty():
            fringe = LinkedQueue()
            fringe.enqueue(self.root())
            while not fringe.is_empty( ):
                p = fringe.dequeue( )
                yield p
                for c in self.children(p):
                    fringe.enqueue(c)
                    
        
class BinaryTree(Tree):
    def left(self, p):
        raise NotImplementedError("must be implemented by subclass")
    def right(self, p):
        raise NotImplementedError("must be implemented by subclass")
    def sibling(self, p):
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)
            else:
                return self.left(parent)
    def children(self,p):
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)
    def inorder(self):
        if not self.is_empty():
            for p in self.subtree_inorder(self.root()):
                yield p
    def subtree_inorder(self, p):
        if self.left(p) is not None:
            for other in self.subtree_inorder(self.left(p)):
                yield other
        yield p
        if self.right(p) is not None:
            for other in self.subtree_inorder(self.right(p)):
                yield other
    def positions(self):
        return self.inorder( )
    
class LinkedBinaryTree(BinaryTree):
    class _Node:
        __slots__ = '_element', '_parent', '_left', '_right'

        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right

    class Position(BinaryTree.Position):
        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node

    def _validate(self, p):
        if not isinstance(p, self.Position):
            raise TypeError("p must be proper Position type")
        if p._container is not self:
            raise TypeError("p does not belong to this continer")
        if p._node._parent is p._node:
            raise ValueError("p is no longer Valid")
        return p._node

    def make_position(self, node):
        return self.Position(self, node) if node is not None else None

    def __init__(self):
        self._root = None
        self._size = 0

    def __len__(self):
        return self._size

    def root(self):
        return self.make_position(self._root)

    def parent(self, p):
        node = self._validate(p)
        return self.make_position(node._parent)

    def left(self, p):
        node = self._validate(p)
        return self.make_position(node._left)

    def right(self, p):
        node = self._validate(p)
        return self.make_position(node._right)

    def num_children(self, p):
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count

    def add_root(self, e):
        if self._root is not None: raise ValueError("Root Exists")
        self._size = 1
        self._root = self._Node(e)
        return self.make_position(self._root)

    def add_left(self, p, e):
        node = self._validate(p)
        if node._left is not None: raise ValueError("left child already exists")
        self._size += 1
        node._left = self._Node(e, node)
        return self.make_position(node._left)

    def add_right(self, p, e):
        node = self._validate(p)
        if node._right is not None: raise ValueError("right child already exists")
        self._size += 1
        node._right = self._Node(e, node)
        return self.make_position(node._right)

    def replace(self, p, e):
        node = self._validate(p)
        old = node._element
        node._element = e
        return old

    def delete(self, p):
        node = self._validate(p)
        if self.num_children(p) == 2: raise ValueError("p has two children")
        child = node._left if node._left else node._right
        if child is not None:
            child._parent = node._parent
        if node is self._root:
            self._root = child
        else:
            parent = node._parent
            if node is parent._left:
                parent._left = child
            else:
                parent._right = child
        self._size -= 1
        node._parent = node
        return node._element

    def attach(self, p, t1, t2):
        node = self._validate(p)
        if not self.is_leaf(p): raise ValueError("position must be leaf")
        if not type(self) is type(t1) is type(t2):
            raise TypeError("Tree types must match")
        self._size += len(t1) + len(t2)
        if not t1.is_empty():
            t1._root._parent = node
            node._left = t1._root
            t1._root = None
            t1._size = 0
        if not t2.is_empty():
            t2._root._parent = node
            node._right = t2._root
            t2._root = None
            t2._size = 0
            
def main():
    print("Task 1")
    T=LinkedBinaryTree()       
    r=T.add_root(50)
    x=T.add_left(r,9)
    c=T.add_right(r,25)
    d=T.add_left(x,6)
    e=T.add_right(x,4)
    f=T.add_left(c,5)
    g=T.add_right(c,1)
    a=T.add_left(d,99)
  
    print("Length of tree is:",T.__len__())
main()

#task 2
class ExpressionTree(LinkedBinaryTree):
    def __init__ (self, token, left=None, right=None):
        super().__init__()
        if not isinstance(token, str):
            raise TypeError('Token must be a string')
        self.add_root(token)
        if left is not None:
            if token not in "+-*x/" :
                raise ValueError("token must be valid operator")
            self.attach(self.root(), left, right)
    def __str__(self):
        pieces = []
        self._parenthesize_recur(self.root(), pieces)
        return ''.join(pieces)
    def _parenthesize_recur(self, p, result):
        if self.is_leaf(p):
            result.append(str(p.element()))
        else:
            result.append('(')
            self._parenthesize_recur(self.left(p), result)
            result.append(p.element())
            self._parenthesize_recur(self.right(p), result)
            result.append(')')
    def evaluate(self):
        return self._evaluate_recur(self.root())
    def _evaluate_recur(self, p):
        if self.is_leaf(p):
            return float(p.element())
        else:
            op = p.element( )
            left_val = self._evaluate_recur(self.left(p))
            right_val = self._evaluate_recur(self.right(p))
            if op == '+' :
                return left_val + right_val
            elif op == '-' :
                return left_val - right_val
            elif op == '/' :
                return left_val / right_val
            else:
                return left_val * right_val
    def build_expression_tree(tokens):
        S = []
        for t in tokens:
            if t in '+-x*/':
                S.append(t)
            elif t not in '()':
                S.append(ExpressionTree(t))
            elif t == ')':
                right = S.pop()
                op = S.pop()
                left = S.pop()
                S.append(ExpressionTree(op, left, right))
        return S.pop()


print("Task 2")
et1 = ExpressionTree('*')
root = et1.root()
m = et1.add_left(root, '+')
et1.add_left(m, '3')
et1.add_right(m, '1')
et1.add_right(root, '4')
et2 = ExpressionTree('+')
root = et2.root()
m = et2.add_left(root, '-')
et2.add_left(m, '9')
et2.add_right(m, '5')
et2.add_right(root, '2')
tree = ExpressionTree('/', et1,et2)
e = str(tree)
new_tree = ExpressionTree.build_expression_tree(e)
print(new_tree)


#task 3
class ExpressionTree(LinkedBinaryTree):

    def __init__(self, token, left=None, right=None):
        super().__init__()
        if not isinstance(token, str):
            raise TypeError('Token must be a string')
        self.add_root(token)
        if left is not None:
            if token not in '+-*x/':
                raise ValueError("token must be valid operator")
            self.attach(self.root(), left, right)

    def __str__(self):
        pieces = []
        self.parenthesize_recur(self.root(), pieces)
        return ''.join(pieces)

    def parenthesize_recur(self, p, result):
        if self.is_leaf(p):
            result.append(str(p.element()))
        else:
            result.append('(')
            self.parenthesize_recur(self.left(p), result)
            result.append(p.element())
            self.parenthesize_recur(self.right(p), result)
            result.append(')')

    def evaluate(self, optdict=None):
        x = optdict.copy()
        return self.evaluate_recur(self.root(), x)

    def evaluate_recur(self, p, optdict1=None):
        if self.is_leaf(p):
            try:
                float(p.element())
                return float(p.element())
            except:
                return float(optdict1[p.element()])
        else:
            op = p.element()
            left_val = self.evaluate_recur(self.left(p), optdict1)
            right_val = self.evaluate_recur(self.right(p), optdict1)
            if op == '+':
                return left_val + right_val
            elif op == '-':
                return left_val - right_val
            elif op == '/':
                return left_val / right_val
            else:
                return left_val * right_val
    def postfix(self):
        for t in self.postorder():
            print(t.element(), end=' ')
    
    def build_expression_tree(tokens):
        S = []
        for t in tokens:
            if t in '+-x*/':
                S.append(t)
            elif t not in '()':
                S.append(ExpressionTree(t))
            elif t == ')':
                right = S.pop()
                op = S.pop()
                left = S.pop()
                S.append(ExpressionTree(op, left, right))
        return S.pop()
print ("task 3")    
exp = '(a*(b+c))'
dic = {'a': 2, 'b': 6, 'c': 4}
Tree1 = ExpressionTree.build_expression_tree(exp)
print(Tree1.evaluate(dic))

print("task 4")
Tree1.postfix()
