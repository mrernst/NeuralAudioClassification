import os
import random
import torch
import numpy as np
import argparse

if __name__ == "__main__":
	# parse arguments
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--type',
						default='supervised',
						type=str)
	
	args = parser.parse_args()

	if args.type == 'supervised':
		# start supervised run
		os.system("python3 supervised.py 1")
	elif args.type == 'selfsupervised':
		# start self-supervised run
		os.system("python3 selfsupervised.py 1")
	else:
		raise NotImplementedError("not yet implemented")


