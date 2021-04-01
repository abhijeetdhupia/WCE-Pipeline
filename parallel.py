import os
from multiprocessing import Pool

processes = ('process1.py', 'process2.py', 'process3.py', 'process4.py')


def run_process(process):
    os.system('python {}'.format(process))
    

def mp_pool():
    pool = Pool()
    pool.map(run_process, processes)
    pool.close()
    pool.join()
