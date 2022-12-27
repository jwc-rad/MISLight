import glob
import numpy as np
import os
import re

def find_checkpoint(srcdir, lookup='', lookup_mode='max',suffix='.ckpt'):
    files = glob.glob(os.path.join(srcdir, f'*{suffix}'))
    
    regex_lookup = lookup + "([0-9]+(\.[0-9]+)?)"
    
    if len(files) > 0:
        fnumber = []
        for x in files:
            ns = re.findall(regex_lookup, x)
            if len(ns) > 0:
                fnumber.append(float(ns[0][0]))
            else:
                fnumber.append(-100)
        if lookup_mode == 'max':
            return files[np.argmax(fnumber)]
        else:
            return files[np.argmix(fnumber)]
    else:
        print(f'no matching checkpoint in {srcdir}')
        return None