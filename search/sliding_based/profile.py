from time import perf_counter as now

from search.sliding_based.framework import sliding_framework
from utilities.box2D import Box2D
from utilities.dataset import load_yolo_for

if __name__ == '__main__':
    file_name = '/home/lg/VDBM/spatiotemporal/resource/dataset/traj_taipei_0412.pkl'
    data, _, _, = load_yolo_for(file_name)

    label = {0: 1}
    interval = 0, 10*60*30
    duration = 3, 100
    region = Box2D((546, 727, 427, 569))

    count = 0
    start = now()
    for i in sliding_framework(data, region, label, duration[0], interval):
        count += 1
    end = now()
    print('result count:', count, 'using', end - start, 's.')


