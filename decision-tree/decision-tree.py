from math import log
from PIL import Image,ImageDraw

my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]


# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
    # Make a function that tells us if a row is in the first group (true) or the second group (false)
    split_function = None
    if isinstance(value, int) or isinstance(value, float): # check if the value is a number i.e int or float
        split_function = lambda row:row[column] >= value
    else:
        split_function = lambda row:row[column] == value

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

# divideset() divides the set rows into 2 children sets,
# based on the criteria of the column number and the value that it takes.
# The data is divided into two sets,
# one where split_function returns True (set1) and one where it returns False (set2).
# If it the data is numeric, the True criterion is that the value in this column is greater than the given value.
# If the data is not numeric, split_function simply determines whether the column’s value is the same as value.


# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row)-1]
        if r not in results: results[r] = 0
        results[r] += 1
    return results

# Tesing
# >>> print(decision.uniquecounts(decision.my_data))
# {'None': 7, 'Premium': 3, 'Basic': 6}


# Entropy is the sum of p(x)log(p(x)) across all 
def entropy(rows):
    results = uniquecounts(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p*log(p,2)
    return ent

# Tesing
# >>> set1, set2 = decision.divideset(decision.my_data, 3, 20)
# >>> decision.entropy(set1), decision.entropy(set2)
# (1.4488156357251847, 0.9182958340544896)


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb


def buildtree(rows,scoref=entropy): #rows is the set, either whole dataset or part of it in the recursive call, 
                                    #scoref is the method to measure heterogeneity. By default it's entropy.
    if len(rows)==0: return decisionnode() #len(rows) is the number of units in a set
    current_score=scoref(rows)

    # Set up some variables to track the best criteria
    best_gain=0.0
    best_criteria=None
    best_sets=None
  
    column_count=len(rows[0])-1   #count the # of attributes/columns. 
                                #It's -1 because the last one is the target attribute and it does not count.
    for col in range(0,column_count):
        # Generate the list of all possible different values in the considered column
        global column_values        #Added for debugging
        column_values={}            
        for row in rows:
            column_values[row[col]]=1   
        # Now try dividing the rows up for each value in this column
        for value in column_values.keys(): #the 'values' here are the keys of the dictionnary
            (set1,set2)=divideset(rows,col,value) #define set1 and set2 as the 2 children set of a division
      
            # Information gain
            p=float(len(set1))/len(rows) #p is the size of a child set relative to its parent
            gain=current_score-p*scoref(set1)-(1-p)*scoref(set2) #cf. formula information gain
            if gain>best_gain and len(set1)>0 and len(set2)>0: #set must not be empty
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
        
  # Create the sub branches   
    if best_gain>0:
        trueBranch=buildtree(best_sets[0])
        falseBranch=buildtree(best_sets[1])
        return decisionnode(col=best_criteria[0],value=best_criteria[1],
                            tb=trueBranch,fb=falseBranch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print(str(tree.results))
    else:
        print(str(tree.col)+':'+str(tree.value)+'? ')
        # Print the branches
        print(indent+'T->', end=" ")
        printtree(tree.tb,indent+'  ')
        print(indent+'F->', end=" ")
        printtree(tree.fb,indent+'  ')


def getwidth(tree):
    if tree.tb==None and tree.fb==None: return 1
    return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
    if tree.tb==None and tree.fb==None: return 0
    return max(getdepth(tree.tb),getdepth(tree.fb))+1

def drawtree(tree,jpeg='tree.jpg'):
    w=getwidth(tree)*100
    h=getdepth(tree)*100+120

    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    drawnode(draw,tree,w/2,20)
    img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
    if tree.results==None:
        # Get the width of each branch
        w1=getwidth(tree.fb)*100
        w2=getwidth(tree.tb)*100

        # Determine the total space required by this node
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        # Draw the condition string
        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

        # Draw links to the branches
        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

        # Draw the branch nodes
        drawnode(draw,tree.fb,left+w1/2,y+100)
        drawnode(draw,tree.tb,right-w2/2,y+100)
    else:
        txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
        draw.text((x-20,y),txt,(0,0,0))


def classify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
        return classify(observation,branch)