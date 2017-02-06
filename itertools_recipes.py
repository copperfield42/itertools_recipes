#!/usr/bin/env python
# -*- coding: utf-8 -*-
u"""
Implementación de los recipes de Itertools compatible con python 2.7 y 3.5
Además para versiones anteriores de python, pone a disposición los equivalentes
de las funciones disponibles en itertools en python 3.5 eso incluye 'ifilter',
'izip', 'ifilterfalse', 'izip_longest', 'imap' con sus nombres como en python 3
'filter', 'zip', 'filterfalse', 'zip_longest', 'map'"""

import collections as _collections
import operator as _operator
import random as _random
#import numbers as _numbers
#import sys as _sys
#from functools import partial as _partial

version="2.4"
_sentinela = object()

#comunes a 2.7 y 3.5
##from itertools import chain, combinations, combinations_with_replacement, \
##                      compress, count, cycle, dropwhile, groupby, islice, \
##                      permutations, product, repeat, starmap, takewhile, tee \
##

from itertools import chain, count, cycle, dropwhile, repeat, takewhile

try:#python 2
    from itertools import ifilter as filter, imap as map, izip as zip, \
                          ifilterfalse as filterfalse #, izip_longest as zip_longest
except ImportError:
    #python 3
    from itertools import filterfalse #, zip_longest

try:#python 2
    range = xrange
    _reduce = reduce
except NameError:
    #python 3
    from functools import reduce as _reduce




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



try: #python 3
    from itertools import accumulate
    accumulate([1,2,3],_operator.add) #3.1
except (ImportError,TypeError):
    #python 2 o anterior a 3.2
    from _itertools_recipes import accumulate


try:
    chain.from_iterable
except AttributeError:
    #python 2.5-
    from _itertools_recipes import chain


try:
    from itertools import combinations
except ImportError:
    from _itertools_recipes import combinations


try:
    from itertools import combinations_with_replacement
except ImportError:
    from _itertools_recipes import combinations_with_replacement


try:
    from itertools import compress
except ImportError:
    #python 2.6- o 3.0
    from _itertools_recipes import compress


try:
    count(0,2)
except TypeError:
    #python 2.6- o 3.1-
    from _itertools_recipes import count


try:
    from itertools import groupby
except ImportError:
    from _itertools_recipes import groupby


try:
    from itertools import islice
except ImportError:
    from _itertools_recipes import islice


try:
    from itertools import permutations
except ImportError:
    from _itertools_recipes import permutations


try:
    from itertools import product
except ImportError:
    from _itertools_recipes import product


try:
    from itertools import starmap
except ImportError:
    from _itertools_recipes import starmap


try:
    from itertools import tee
except ImportError:
    from _itertools_recipes import tee


try: #python 3
    from itertools import zip_longest
except ImportError:
    try:
        from itertools import izip_longest as zip_longest
    except ImportError:
        #python 2.5-
        from _itertools_recipes import zip_longest


############################################################################################
##-------------------------------- Itertools recipes ---------------------------------------
############################################################################################


def repeatfunc(func, times=None, *args):
    u"""Repeat calls to func with specified arguments.

        Example:  repeatfunc(random.random)"""
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def tabulate(function, start=0):
    u"""Return function(0), function(1), ..."""
    return map(function, count(start))

def consume(iterator,n=None):
    u"""Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        _collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def nth(iterable, n, default=None):
    u"""Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default)

def quantify(iterable, pred=bool):
    u"""Count how many times the predicate is true"""
    return sum(map(pred, iterable))

def padnone(iterable):
    u"""Returns the sequence elements and then returns None indefinitely.

        Useful for emulating the behavior of the built-in map() function."""
    return chain(iterable, repeat(None))

def ncycles(iterable, n):
    u"""Returns the sequence elements n times

        ncycles("XYZ",3) --> X Y Z X Y Z X Y Z"""
    return chain.from_iterable(repeat(tuple(iterable), n))

def dotproduct(vec1, vec2, sum=sum, map=map, mul=_operator.mul):
    u"""sum(map(mul, vec1, vec2))"""
    return sum(map(mul, vec1, vec2))

def flatten(listOfLists):
    u"""Flatten one level of nesting"""
    return chain.from_iterable(listOfLists)

def pairwise(iterable):
    u"""s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def roundrobin(*iterables):
    u"""roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # Recipe credited to George Sakkis
    pending = len(iterables)
    try: #python 3
        iter([]).__next__
        nexts = cycle(iter(it).__next__ for it in iterables)
    except AttributeError: #Python 2
        nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def partition(pred, iterable):
    u"""Use a predicate to partition entries into false entries and true entries
        partition(is_odd, range(10)) --> 0 2 4 6 8  and  1 3 5 7 9"""
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

def unique_everseen(iterable, key=None):
    u"""List unique elements, preserving order. Remember all elements ever seen.
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D"""
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def unique_justseen(iterable, key=None):
    u"""List unique elements, preserving order. Remember only the element just seen.
        unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
        unique_justseen('ABBCcAD', str.lower) --> A B C A D"""
    return map(next, map(_operator.itemgetter(1), groupby(iterable, key)))

def iter_except(func, exception, first=None):
    u"""Call a function repeatedly until an exception is raised.

        Converts a call-until-exception interface to an iterator interface.
        Like __builtin__.iter(func, sentinel) but uses an exception instead
        of a sentinel to end the loop.

        Examples:
            iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
            iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
            iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
            iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
            iter_except(s.pop, KeyError)                             # non-blocking set iterator"""
    try:
        if first is not None:
            yield first()  # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exception:
        pass

def first_true(iterable, default=False, pred=None):
    u"""Returns the first true value in the iterable.

        If no true value is found, returns *default*

        If *pred* is not None, returns the first item
        for which pred(item) is true.


        first_true([a,b,c], x) --> a or b or c or x
        first_true([a,b], x, f) --> a if f(a) else b if f(b) else x """
    return next(filter(pred, iterable), default)

def random_product(*args, **kwds):
    u"""Random selection from itertools.product(*args, **kwds)"""
    #es asi para hacerlo compatible con python 2
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    return tuple(_random.choice(pool) for pool in pools)

def random_permutation(iterable, r=None):
    u"""Random selection from itertools.permutations(iterable, r)"""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(_random.sample(pool, r))

def random_combination(iterable, r):
    u"""Random selection from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(_random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def random_combination_with_replacement(iterable, r):
    u"""Random selection from itertools.combinations_with_replacement(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(_random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


############################################################################################
##-------------------------- Itertools recipes modificados ---------------------------------
############################################################################################

def take(n, iterable, container=tuple):
    u"""Regresa los primeros n elementos del iterable en el contenedor espesificado.

        take(3,"ABCDEFGHI") --> A B C"""
    #container = karg.get("container",tuple)
    return container(islice(iterable, n))

def tail(n, iterable):
    u"""Return an iterator over the last n items,
        if n is none return a iterator over all
        elemens in iterable save the first

        tail(3,'ABCDEFG') --> E F G
        tail(None,'ABCDEFG')   --> B C D E F G """
    if n is None:
        return islice(iterable,1,None)
    return iter(_collections.deque(iterable, maxlen=n))

def powerset(iterable,ini=0,fin=None):
    u"""Da todas las posibles combinaciones de entre ini y fin elementos del iterable
        Si fin no es otorgado default a la cantidad de elementos del iterable

        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        powerset([1,2,3],2) -->  (1,2) (1,3) (2,3) (1,2,3)
        powerset([1,2,3],0,2) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) """
    if ini<0 or (fin is not None and fin<0):
        raise ValueError("El rango de combinaciones debe ser no negativo")
    elem = tuple(iterable)
    if fin is None:
        fin = len(elem)
    return chain.from_iterable( combinations(elem, r) for r in range(ini,fin+1) )

def all_equal(iterable, key=None):
    u"""Returns True if all the elements are equal to each other"""
    #new recipe in 3.6
    g = groupby(iterable,key)
    return next(g, True) and not next(g, False)

def grouper(iterable, n, fillvalue=None, longest=True):
    u"""Collect data into fixed-length chunks or blocks
        grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        grouper('ABCDEFG', 3, longest=False) --> ABC DEF """
    args = [iter(iterable)] * n
    if longest:
        return zip_longest(*args, fillvalue=fillvalue)  
    else:
        return zip(*args)      
    
############################################################################################
##----------------------------------- Mis recipes ------------------------------------------
############################################################################################



def chunker(n, iterable, **karg):
    u"""Regresa un iterador con los datos agrupados en bloques de a lo
        sumo n elementos del iterable.
        Ejemplo:
        >>> ej=[1,2,3,4,5,6,7,8,9]
        >>> list(chunker(4,ej))
        [(1, 2, 3, 4), (5, 6, 7, 8), (9,)]

        Se puede espesificar el contenedor con el parametro key-only
        'container' en cuyo caso, los elementos son guardados en
        una instancia del mismo."""
    #http://stackoverflow.com/a/41245197/5644961
    container = karg.get("container",tuple)
    it = iter(iterable)
    elem = container( islice(it,n) )
    while elem:
        yield elem
        elem = container( islice(it,n) )


def _len_range(ran):
    #http://stackoverflow.com/questions/14754877/pre-compute-lenrangestart-stop-step
    #http://stackoverflow.com/a/19440091/5644961
    if isinstance(ran,range):
        try:
            return len(ran)
        except OverflowError:
            pass
        start,stop,step = 0, 0, 1
        if all( hasattr(ran,x) for x in ("start","stop","step") ):
            start,stop,step =  ran.start, ran.stop, ran.step
        else:
            elem = list(map(int,repr(ran).replace("xrange(","").replace(")","").split(",")))
            if len(elem) == 1:
                stop = elem[0]
            elif len(elem) == 2:
                start,stop = elem
            else:
                start,stop,step = ran
        return max(0, (stop - start) // step + bool((stop - start) % step))
    else:
        raise ValueError("Se esperaba una instancia de {}".format(range))

def ilen(iterable) :
    u"""Dice la cantidad de elementos del iterable iterando sobre el mismo si es necesario."""
    try:
        return len(iterable)
    except TypeError:
        return sum( 1 for _ in iterable )        
    except OverflowError as oe:
        #print("over")
        if isinstance(iterable,range):
            #print("range")
            return _len_range(iterable)
        else:
            raise oe




try: #python 2
    _basestring = (basestring,)
except NameError:
    #python 3
    _basestring = (str,bytes)

def flatten_total(iterable, flattype=_collections.Iterable, ignoretype=_basestring):
    u"""Flatten all level of nesting of a arbitrary iterable"""
    #http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    #unutbu version
    remanente = iter(iterable)
    while True:
        try: #para futura compativilidad con python 3.7
            elem = next(remanente)
        except StopIteration:
            return 
        if isinstance(elem,flattype) and not isinstance(elem,ignoretype):
            remanente = chain( elem, remanente )
        else:
            yield elem

def flatten_level(iterable,nivel=1,flattype=_collections.Iterable, ignoretype=_basestring):
    u"""Flatten N levels of nesting of a arbitrary iterable"""
    #http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    #Cristian version modificada
    if nivel < 0:
        yield iterable
        return
    for elem in iterable:
        if isinstance(elem,flattype) and not isinstance(elem,ignoretype):
            for sub in flatten_level(elem,nivel-1,flattype,ignoretype):
                yield sub
        else:
            yield elem

def irange(start,stop,step=1):
    u"""Simple iterador para producir valores en [start,stop)"""
    return takewhile(lambda x: x<stop, count(start,step))

def groupwise(iterable,n):
    u"""s -> (s0,s1,...,sn-1), (s1,s2,...,sn), (s2, s3,...,sn+1), ..."""
    grupo = tee(iterable,n)
    for i,e in enumerate(grupo):
        consume(e,i)
    return zip( *grupo )

def ida_y_vuelta(iterable):
    u"""s-> s0,s1,s2,...,sn-2,sn-1,sn,sn-1,sn-2,...,s2,s1,s0"""
    try:
        ida = iter(iterable)
        vue = reversed(iterable)
        next(vue,None)
        for x in chain(ida,vue):
            yield x
    except TypeError:
        vue = list()
        for x in iterable:
            yield x
            vue.append(x)
        if vue:
            vue.pop()
        for x in reversed(vue):
            yield x

def iteratefunc(func,start,times=None,iter_while=None):
    u"""Generador de relaciones de recurrencia de primer order:
        F0 = start
        Fn = func( F(n-1) )

        iteratefunc(func,start)   -> F0 F1 F2 ...
        iteratefunc(func,start,n) -> F0 F1 F2 ... Fn-1
        
        Sirve por ejemplo para mapas logisticos: Xn+1=rXn(1-Xn) que seria 
        iteratefunc(lambda x: r*x*(1-x),x0)
        
        El conjunto de Mandelbrot podria ser:
        iteratefunc(lambda z: c + z**2 , 0, iter_while=lambda x: abs(x)<=2 )

        func: función de 1 argumento cuyo resultado es usado para
              producir el siguiente elemento de secuencia en la
              siguiente llamada.
        start: elemento inicial de la secuencia.
        times: cantidad de elementos que se desea generar de la
               secuencia, si es None se crea una secuencia infinita.
        iter_while: si es otorgado se producen elementos de la secuencia
                    mientras estos cumplan con esta condición. """
    seq = padnone([start])
    if times is not None:
        seq = islice(seq,times) 
    result = accumulate(seq,lambda x,_:func(x))
    if iter_while:
        return takewhile(iter_while,result)
    return result 

def iteratefunc2ord(a,b,binop=_operator.add, func1=None, func2=None, times=None):
    '''Generador de relaciones de recurrencia de 2do order:
        F0 = a
        F1 = b
        Fn = binop( func1( F(n-1) ),  func2( F(n-2)) )

        por ejemplo los numeros de Fibonacci
        F0=0
        F1=1
        Fn = F(n-1) + F(n-2)
        es  iteratefunc2ord(0,1,add) -> 0 1 1 2 3 5 8 ....

        F0=10
        F1=1
        Fn = 8*F(n-1) - F(n-2)**2
        es iteratefunc2ord(10,1,sub,lambda x: x*8,lambda y:y**2)
        -> 10, 1, -92, -737, -14360, ...
        
        
        a,b: casos bases
        func1, func2: funciones de 1 argumento que reciben el elemento 
                      anterior, y el anterior a este en la secuencia
                      respectivamente. Por defecto son la funcion identidad.
        binop: funcion de 2 argumentos que toma el resultado de las funciones 
               anteriormente nombradas y produce el siguiente elemento de 
               secuencia.
        times: cantidad de elementos que se desea generar de la
               secuencia, si es None se crea una secuencia infinita.
        
        '''
    if func1 is None:
        func1 = lambda x: x
    if func2 is None:
        func2 = lambda y: y
    seq = padnone([a,b])
    if times is not None:
        seq = islice(seq,times) 
    yield next(seq) # a
    yield next(seq) # b
    a,b = b, binop(func1(b),func2(a))
    for _ in seq:
        yield b
        a,b = b, binop(func1(b),func2(a))

def dual_reduce(iterable, func1=min, func2=max, default=_sentinela):
    u"""Aplica las funciones dadas de 2 argumentos a todos los 
        elementos del iterable en una sola iteración.
        Equibalente a (reduce(func1,iterable[,default[0]]),reduce(func1,iterable[,default[1]]))
        Si el iterable esta vacio se regresa default si el mismo es provisto si no arroja
        TypeError"""
    def fun(previous, new):
        a,b = previous
        return func1(a,new), func2(b,new)
    iterable = iter(iterable)
    try:
        ini = next(iterable)
    except StopIteration:
        if default is _sentinela:
            raise TypeError("dual_reduce() of empty sequence with no default value")
        return default
    if default is _sentinela:
        return _reduce( fun, chain([(ini,ini)],iterable) )
    else:
        return _reduce( fun, chain([default,ini],iterable ) )

def dual_accumulate(iterable, func1=min, func2=max):
    u"""Regresa una serie de valores correspondientes a la aplicación
        de las funciones dadas a cada elemento del iterable en una
        sola iteración.
        Equivalente a zip(accumulate(iterable,func1),accumulate(iterable,func2))"""
    iterable = iter(iterable)
    try:
        ini = next(iterable)
    except StopIteration:
        return iter( () )
    return accumulate( chain([(ini,ini)],iterable),
                       lambda pre,x: (func1(pre[0],x),func2(pre[1],x))
                       )

def strip_last(iterable,n):
    """Entrega todos menos los ultimos n elementos del iterable
       strip_last("123",1) -> 1 2 """
    #if isinstance(iterable,_collections.Sized ):
    try:
        stop = len(iterable) - n
        if stop > 0:
            for x in islice(iterable,stop):
                yield x
    #else:
    except TypeError:
        n    = n+1
        it   = iter(iterable)
        cola = _collections.deque(islice(it,n), maxlen=n)
        while len(cola)==n:
            yield cola.popleft()
            try: #para futura compativilidad con python 3.7
                cola.append( next(it) )
            except StopIteration:
                return

def range_of(iterable,*argv,**karg):#ini,fin,step=1,key=None):
    """range_of(iterable[,stop][,*,key])
       range_of(iterable[,start,stop[,step]][,*,key])
       Dado un iterable ordenado de elementos entrega el equivalente a
       
       it = iter(iterable)
       x = next(it)
       while x <= stop:
           if star <= x:
               yield x
           x = next(it)
       """
    key = karg.pop('key',lambda x:x)
    if karg:
        raise TypeError('got an unexpected keyword argument(s): '+", ".join(map(repr,karg)))
    lim = slice(*argv)
    result = None
    if lim.stop is None:
        result = iter(iterable)
    else:
        result = takewhile(lambda x: key(x)<=lim.stop, iterable )
    if lim.start is not None:
        result = dropwhile( lambda x: lim.start>key(x),result)
    if lim.step is None:
        return result
    else:
        return islice(result,None,None,lim.step)

def splitAt(n,iterable):
    """Regresa 2 iterables, el primero contiene los primeros n elementos
       del iterable dado y el segundo contiene el resto"""
    a,b = tee(iterable,2)
    return islice(a,n),islice(b,n,None)

