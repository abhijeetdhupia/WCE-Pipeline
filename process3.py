import os
import time
import sys
import subprocess
from process import Network

class p3():
    def __init__(self):
        self.run()

    def run(self):
        #######INPUTS########
        pid = subprocess.Popen("ls patients/ -t1 |  head -n 1",
                           shell=True,
                           stdout=subprocess.PIPE,
                           universal_newlines=True).communicate()[0]
        root_path = os.getcwd() +'/patients/' + pid.strip()
        img_path = root_path + '/Images/dir_3'
        # print('img_path', img_path)
        #####################
        #tmp = time.time()
        Network(3, img_path, root_path)
        #tend = time.time()
        #print(tend-tmp)

if __name__ == "__main__":
    p3()
