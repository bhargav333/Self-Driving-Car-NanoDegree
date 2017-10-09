"""
Utility Fabric file which is used for automation of tasks such as download models from AWS train and fetch 
Newly trained models
"""

from fabric.api import  env, run
from fabric.contrib.project import rsync_project
from fabric.context_managers import prefix
import os
WORKDIR = '/home/carnd/CarND-Behavioral-Cloning-P3'

env.shell = "/bin/bash -l -i -c" #load the environment
env.hosts = ['52.59.71.221']
env.user = 'carnd'


def sync():
    """
    Synchronize code in the local repo with AWS - GPU instance 
    :return: 
    """

    rsync_project(remote_dir=WORKDIR, local_dir='.', delete=True,
                  exclude=['*.pyc', '*.DS_Store', '.git', '.cache/*', '*json', '*pkl','models*','*zip'])


def fetch_models():
    """
    Fetch last trained model from AWS
    :return: 
    """

    os.system('rsync -rtuv {}@{}:/home/carnd/CarND-Behavioral-Cloning-P3/models ./'.format(env.user,
                                                                                           env.hosts[0]))


def deploy():

    """
    Redeploy code, train model on AWS and download last version locally
    :return: 
    """

    sync()

    with prefix('cd {} && source activate carnd-term1'.format(WORKDIR)):

        run('python train.py')

    fetch_models()



