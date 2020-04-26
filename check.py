import os
import subprocess
import time
from datetime import datetime

counter = 0
while True:
    time.sleep(5)

    stream = os.popen('nvidia-smi')
    output = stream.read()

    lines = output.split('\n')

    gpu_used = {'0': 0, '1': 0}
    for line in reversed(lines[:-2]):
        words = line.split()

        if len(words) <= 2:
            break
        gpu_used[words[1]] += int(words[5][:-3])

    if counter % 60 == 0:
        print(datetime.now(), gpu_used)
    counter += 1

    found = False
    for key, value in gpu_used.items():
        if value < 6000:
            found = True
            with open('stdout-%s.txt' % counter, 'w') as f:
                process = subprocess.Popen(['/home/eloghmani/myjodie/evaluate_all_epochs.sh', 'lastfm', 'interaction', key],
                                           stdout=f,
                                           stderr=f)
            break
    if found:
        break
