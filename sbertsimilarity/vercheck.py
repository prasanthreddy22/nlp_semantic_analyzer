

import os.path
def check(model):
	if os.path.isdir(model):
   		return 1
	else:
   		return 0