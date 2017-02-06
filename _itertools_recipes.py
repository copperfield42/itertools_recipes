#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
modulo auxiliar del modulo itertools_recipes con implementación 
en pyton puro de las funciones disponibles en el modulo itertools
compatible con python 2.7 y 3.5"""

from itertools   import chain as _chain, repeat
from collections import deque as _deque
from operator    import add   as _add
from sys         import maxsize as _maxsize

__all__ = ['accumulate', 'chain', 'combinations', 'combinations_with_replacement', 
           'compress', 'count', 'groupby', 'islice', 'permutations', 'product', 
           'starmap', 'tee', 'zip_longest']

try:#python 2
    from itertools import imap as map, izip as zip 
except ImportError:
    #python 3
    pass

try:#python 2
    range = xrange
except NameError:
    #python 3
    pass    
    
version="2.4"

###posible fallos de importacion en python anteriores son:
##accumulate new in 3.2 agregado func in 3.3
##combinations_with_replacement new in 3.1 and 2.7
##compress new in 3.1 and 2.7
##count añadido el step in 3.1 and 2.7
##chain.from_iterable new in 2.6
##combinations new in 2.6
#groupby new in 2.4 
##islice Changed in version 2.5: accept None values for default start and step.
#izip Changed in version 2.4 When no iterables are specified, 
#     returns a zero length iterator instead of raising a TypeError exception.
##izip_longest new in 2.6
##permutations new in 2.6
##product new in 2.6
##starmap Changed in version 2.6: Previously, starmap() required the function arguments to be tuples. 
#        Now, any iterable is allowed
##tee new in 2.4


    

def accumulate(iterable, func=_add):
    u"""Return series of accumulated sums (or other binary function results).
       accumulate([1,2,3,4,5]) --> 1 3 6 10 15
       accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120"""
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


class chain(_chain):
    
    @classmethod
    def from_iterable(cls,iterables):
        u"""chain.from_iterable(['ABC', 'DEF']) --> A B C D E F"""
        for it in iterables:
            for element in it:
                yield element


def tee(iterable, n=2):
    u"""tee(iterable, n=2) --> tuple of n independent iterators."""
    it = iter(iterable)
    deques = [_deque() for i in range(n)]
    def gen(mydeque):
        while True:
            if not mydeque:             # when the local deque is empty
                try:
                    newval = next(it)   # fetch a new value and
                except StopIteration:
                    return
                for d in deques:        # load it to all the deques
                    d.append(newval)
            yield mydeque.popleft()
    return tuple(gen(d) for d in deques)


def product(*args, **kwds):
    u"""product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111"""
    pools = [tuple(pool) for pool in args] * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
        
    
def permutations(iterable, r=None):
    u"""permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        permutations(range(3)) --> 012 021 102 120 201 210 """
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    for indices in product(range(n), repeat=r):
        if len(set(indices)) == r:
            yield tuple(pool[i] for i in indices)     

            
def is_sorted(iterable):
    a,b = tee(iterable)
    next(b,None)
    return all( x<=y for x,y in zip(a,b) )
   
    
def combinations(iterable, r):
    u"""combinations('ABCD', 2) --> AB AC AD BC BD CD
       combinations(range(4), 3) --> 012 013 023 123"""
    pool = tuple(iterable)
    n = len(pool)
    for indices in permutations(range(n), r):
        if is_sorted(indices):  #sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def combinations_with_replacement(iterable, r):
    u"""Return successive r-length combinations of elements in the iterable
        allowing individual elements to have successive repeats.
        combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"""
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=r):
        if is_sorted(indices): #sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def compress(data, selectors):
    u"""compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F"""
    return (d for d, s in zip(data, selectors) if s)
            

def count(start=0, step=1):
    u"""count(10) --> 10 11 12 13 14 ...
       count(2.5, 0.5) -> 2.5 3.0 3.5 ..."""
    n = start
    while True:
        yield n
        n += step


class groupby(object):
    u"""[k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
       [list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D"""
       
    def __init__(self, iterable, key=None):
        if key is None:
            key = lambda x: x
        self.keyfunc = key
        self.it = iter(iterable)
        self.tgtkey = self.currkey = self.currvalue = object()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        while self.currkey == self.tgtkey:
            self.currvalue = next(self.it)    # Exit on StopIteration
            self.currkey = self.keyfunc(self.currvalue)
        self.tgtkey = self.currkey
        return (self.currkey, self._grouper(self.tgtkey))
    
    next = __next__ # para versiones anteriores 
    
    def _grouper(self, tgtkey):
        while self.currkey == tgtkey:
            yield self.currvalue
            try:
                self.currvalue = next(self.it)
            except StopIteration:
                return
            self.currkey = self.keyfunc(self.currvalue)
    
        
def islice(iterable, *args):
    u"""islice('ABCDEFG', 2) --> A B
        islice('ABCDEFG', 2, 4) --> C D
        islice('ABCDEFG', 2, None) --> C D E F G
        islice('ABCDEFG', 0, None, 2) --> A C E G"""
    s = slice(*args)
    it = iter(range(s.start or 0, s.stop or _maxsize, s.step or 1))
    try:
        nexti = next(it)
    except StopIteration:
        return
    for i, element in enumerate(iterable):
        if i == nexti:
            yield element
            try: #para futura compativilidad con python 3.7
                nexti = next(it)
            except StopIteration:
                return 


def starmap(function, iterable):
    u"""starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000"""
    for args in iterable:
        yield function(*args)


class ZipExhausted(Exception):
    pass

def zip_longest(*args, **kwds):
    u"""zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-"""
    fillvalue = kwds.get('fillvalue')
    counter = [len(args) - 1]
    def sentinel():
        #nonlocal counter
        if not counter[0]:
            raise ZipExhausted
        counter[0] -= 1
        yield fillvalue
    fillers = repeat(fillvalue)
    iterators = [_chain(it, sentinel(), fillers) for it in args]
    try:
        while iterators:
            yield tuple(map(next, iterators))
    except ZipExhausted:
        pass
        
    
            


