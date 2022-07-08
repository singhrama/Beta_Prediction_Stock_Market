#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import pandas as pd
import option_util as util
import re
import json

### read in arguments
args = util.get_args()

default_args = {'chunks': 40,
                'starts': None,
                'stops': None,
                'partition': 'general',
                'walltime': '24:00:00',
                'cpus': 2,
                'gigs': 20,
                'jobname': args['runcode'],
                'bashfilename': args['runcode'].split('.')[0]}
args = {**default_args, **args}


### assign arguments
runcode = args['runcode']
chunks = args['chunks']
starts = args['starts']
stops = args['stops']
partition = args['partition']
walltime = args['walltime']
cpus = args['cpus']
gigs = args['gigs']
jobname = args['jobname']
bashfilename = args['bashfilename']

### directories
dirs = util.get_dirs()
projdir = dirs['bashdir']
#logdir = dirs['logdir']
codedir = dirs['codedir']
jsondir = dirs['input_json']
#pythonpath = dirs['pythonpath']

### number of jobs and basic parameters
N = chunks
exefile = codedir + runcode

### get starts and stops
# ~ if starts is None or stops is None:
    # ~ runlist = []
# ~ elif type(starts) == int:
    # ~ runlist = list(range(starts, stops+1))
# ~ else:
    # ~ starts = [int(x) for x in starts.split(',')]
    # ~ stops = [int(x) for x in stops.split(',')]
    # ~ runlist = []
    # ~ for start, stop in zip(starts, stops):
        # ~ runlist += list(range(start, st
        
        
def proportional_dividing(N, n):
	arr = []
	if N == 0:
		return arr
	elif n == 0:
		arr.append(N)
		return arr
	r = N // n
	for i in range(n-1):
		arr.append(r)
	arr.append(N-r*(n-1))
	last_n = arr[-1]
	if last_n > r:
		if abs(r-last_n) > 1:
			diff = last_n - r
			for k in range(diff):
				arr[k] += 1
				arr[-1] = r
	return arr

def chunk_split(items, chunks):
	arr = proportional_dividing(len(items), chunks)
	splitted = []
	for chunk_size in arr:
		splitted.append(items[:chunk_size])
		items = items[chunk_size:]
	return splitted

def get_jobN(name):
	return [l[offset:offset + split_size] for offset in offsets]
	matches = re.findall('\d+', name)
	if len(matches) > 0:
		return matches[-1]
	return

### get status
def get_queue():
	status = subprocess.run(['squeue',  '--user=cdavis40'], stdout = subprocess.PIPE)
	status = str(status).split('\\n')
	status = status[1:-1]
	colnames = ['jobid', 'partition', 'name', 'user', 'st', 'time', 'nodes', 'reason']
	outdata = []
	if status == []:
		return pd.DataFrame(columns = colnames + ['jobN'])
	for row in status:
		row = row.split(' ')
		row = [x for x in row if x != '']
		outdata.append(dict(zip(colnames, row)))
	outdata = pd.DataFrame(outdata)
	outdata['jobN'] = [get_jobN(name) for name in outdata.name]
	return(outdata)

def cancel_job(idnumb):
	subprocess.run(['scancel', idnumb])

def submit_job(ii):
	subprocess.run(['sbatch', projdir + str(ii) + bashfilename + '.sh'])

cusip_sec = pd.read_csv("/N/slate/singrama/Input_Files/vsurfpd_cusip_data.csv")
ocu_lt = list(zip(cusip_sec.cusip,cusip_sec.secid))

runlist = chunk_split(ocu_lt, chunks)

### create submit files
for i in range(N):
	if len(runlist) > 0: # or (i+1) in runlist:
		print('Writing Job File Number %d' % (i+1))
		args['chunkid'] = i+1
		ea_js = jsondir + '/' + 'CUSIPs_' + str(i+1) + '.json'
        
		with open(ea_js, 'w') as chti:
			json.dump( runlist[i], chti)
			chti.close()
       
		with open(projdir + str(i+1) + bashfilename + '.sh', 'w') as fptr:
			fptr.write('#!/bin/bash\n')
            # ~ if school == 'chicago':
			fptr.write('#SBATCH -J Stocks_Data_Download\n')
			fptr.write('#SBATCH -o output_SDD_%j.txt\n')
			fptr.write('#SBATCH -e error_SDD_%j.err\n')
			fptr.write('#SBATCH --mail-type=FAIL\n')
			fptr.write('#SBATCH --mail-user=singrama@iu.edu\n')
			fptr.write('#SBATCH --nodes=4\n')
			fptr.write('#SBATCH --ntasks-per-node=4\n')
			fptr.write('#SBATCH --partition=' + partition + '\n')
			fptr.write('#SBATCH --mem=' + str(gigs) + 'G\n')
			fptr.write('#SBATCH --time=' + walltime + '\n')
			fptr.write('#SBATCH --job-name=' + str(i+1) + jobname + '\n')
			#fptr.write('#SBATCH --output="' + logdir + bashfilename + str(i+1) + '-%A.log"\n')
			fptr.write('module load python\n')
			fptr.write('srun python ')
			fptr.write(exefile + util.args_to_string(args) + '\n\n')

### submit jobs
for i in range(N):
	if len(runlist)>0: #len(runlist) == 0 or (i+1) in runlist:
		print('Submitting Job %d:' % (i+1))
		submit_job(i+1)
		print('\tJob %d Submitted' % (i+1))
