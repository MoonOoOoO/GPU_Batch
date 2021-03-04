from multiprocessing import Process
import os
import time
import multiprocessing as mp


def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 10))


if __name__=='__main__':
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    pool = mp.Pool(32)
    pool.map(long_time_task, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    # for i in [1,2]:
    #     p1 = Process(target=long_time_task, args=(i,))
    #     # p2 = Process(target=long_time_task, args=(2,))
    #     print('等待所有子进程完成。')
    #     p1.start()
    #     # p2.start()
    #     p1.join()
    #     # p2.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))