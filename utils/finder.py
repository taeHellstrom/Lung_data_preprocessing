import os

def find_leafes( root ):
    """ finds folders with no subfolders """

    for root, dirs, files in os.walk(root):
        
        if not dirs: # can't go deeper
            return root

