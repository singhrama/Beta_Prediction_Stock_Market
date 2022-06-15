#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import pandas as pd
import Data_SEC_Extract as util
import re

### read in arguments
args = util.get_args()
default_args = {'chunks': 10,
                'starts': None,
                'stops': None,
                'partition': 'standard',
                'walltime': '24:00:00',
                'cpus': 2,
                'gigs': 6,
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
logdir = dirs['logdir']
codedir = dirs['codedir']
pythonpath = dirs['pythonpath']

### number of jobs and basic parameters
N = chunks
exefile = codedir + runcode

### get starts and stops
if starts is None or stops is None:
    runlist = []
elif type(starts) == int:
    runlist = list(range(starts, stops+1))
else:
    starts = [int(x) for x in starts.split(',')]
    stops = [int(x) for x in stops.split(',')]
    runlist = []
    for start, stop in zip(starts, stops):
        runlist += list(range(start, stop+1))
        
        

def get_jobN(name):
    matches = re.findall('\d+', name)
    if len(matches) > 0:
        return matches[-1]
    return ''

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

### create submit files
for i in range(N):
    if len(runlist) == 0 or (i+1) in runlist:
        print('Writing Job File Number %d' % (i+1))
        args['chunkid'] = i+1
        with open(projdir + str(i+1) + bashfilename + '.sh', 'w') as fptr:
            fptr.write('#!/bin/bash\n')
            if school == 'chicago':
                fptr.write('#SBATCH --account=phd\n')
            fptr.write('#SBATCH --partition=' + partition + '\n')
			fptr.write('#SBATCH --mem=' + str(gigs) + 'G\n')
            fptr.write('#SBATCH --time=' + walltime + '\n')
            fptr.write('#SBATCH --job-name=' + str(i+1) + jobname + '\n')
            fptr.write('#SBATCH --output="' + logdir + bashfilename + str(i+1) + '-%A.log"\n')
			fptr.write('module load python\n')
            fptr.write('srun python3.6')
            fptr.write('/N/u/singrama/Carbonate/Documents/Beta_Conditional/ ' + exefile + util.args_to_string(args) + '\n\n')          

### submit jobs
for i in range(N):
    if len(runlist) == 0 or (i+1) in runlist:
        print('Submitting Job %d:' % (i+1))
        submit_job(i+1)
        print('\tJob %d Submitted' % (i+1))
