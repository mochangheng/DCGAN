import os
import glob

class Logger():
    def __init__(self,path):
        self.file = open(path,'a+')
    def log(self,text):
        print(text)
        self.file.write(str(text)+'\n')
        self.file.flush()
    def close(self):
        self.file.close()


def create_new_run(path,run_name):
    if not os.path.isdir(path):
        os.mkdir(path)
    dirs = glob.glob(os.path.join(path,'*'))
    max_num = 0
    for dir_name in dirs:
        num = dir_name.split('/')[-1].split('-')[0]
        if not num.isdigit():
            continue
        num = int(num)
        if num > max_num:
            max_num = num
    run_num = max_num+1
    run_num = '{:05d}'.format(run_num)
    run_dir = os.path.join(path,'-'.join([run_num,run_name]))
    os.mkdir(run_dir)
    return run_dir