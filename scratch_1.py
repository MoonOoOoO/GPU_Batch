import time
from multiprocessing import Pool, Process
from concurrent.futures import ProcessPoolExecutor


def process_file(i):
    print(i)
    time.sleep(3)


N_PROC = 16
l = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# p = Pool(processes=N_PROC)
# tic = time.time()
# p.map(process_file, l)
# p.close()
# print(time.time() - tic)

tic = time.time()
processes = []
for i in l:
    proc = Process(target=process_file, args=(i,))
    processes.append(proc)
    proc.start()
for proc in processes:
    proc.join()
print(time.time() - tic)

executor = ProcessPoolExecutor(max_workers=N_PROC)
executor.map(process_file, l)
