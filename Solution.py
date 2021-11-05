import json
import sys
import copy
from typing import ByteString
import numpy as np

SMALL_JOB_NUM = 5
TOTAL_CORE = 120
NODE_NUM = 6
CORE_PER_NODE = 20

# core衰减系数
CORE_FACTOR = [-0.0002 * i * i - 0.0063 * i + 1.0065 for i in range(1, 30)] + \
              [ (19.125 / i) for i in range(30, 121)]
# job衰减系数
JOB_FACTOR = [-0.0002 * i * i * i - 0.0063 * i * i + 1.0065 * i for i in range(1, 30)] + \
             [ 19.125 for i in range(30, 121)]


def top_k(arr, k):
    top_k_index = np.argsort(arr)[:k]
    top_k = np.array(arr)[top_k_index]
    return top_k, top_k_index


def core_time_steps(core_time_list):
    res = []
    iter_core_time_list = []
    for i in range(len(core_time_list)):
        core_time = core_time_list[i]
        if core_time not in iter_core_time_list:
            res.append([core_time, core_time_list.count(core_time)])
            iter_core_time_list.append(core_time)
    return sorted(res, key=lambda item: item[0])


def get_best_job(unsched_job, unsched_job_core_list):
    max_sched_ratio = -1
    best_index = -1
    best_job = None
    for i in range(len(unsched_job)):
        job = unsched_job[i]
        sched_ratio = unsched_job_core_list[i] / job.scoreNum
        if sched_ratio > max_sched_ratio:
            max_sched_ratio, best_job, best_index = sched_ratio, job, i
    return best_job, best_index


def placement(job_list, core_list, host_list, core_time_list, batchJobPlans):
    core_in_host = [0 for _ in range(NODE_NUM)]
    job_in_core = [-1 for _ in range(TOTAL_CORE)]
    unsched_job = copy.copy(job_list)
    unsched_job_core_list = []
    total_core_num = 0
    unsched_block_num = 0
    for job in unsched_job:
        unsched_job_core_list.append(job.coreNum)
        total_core_num += job.coreNum
        unsched_block_num += len(job.blocks)
    for i in range(total_core_num):
        best_job, best_index = get_best_job(unsched_job, unsched_job_core_list)
        best_job.schedule_core(core_list, core_in_host, best_index, job_in_core)
        unsched_job_core_list[best_index] -= 1

    unsched_local_block_num = 0
    for core in core_list:
        core.localBlocks = sorted(core.localBlocks, key=lambda x: x.get_calc_time())
        unsched_local_block_num += len(core.localBlocks)

    while unsched_block_num != 0:
        unsched_local_block_num_copy = unsched_local_block_num
        unsched_block_num_copy = unsched_block_num
        for host in host_list:
            best_core = None
            min_start_time = sys.maxsize
            for core in in host.get_core_list():
                if job_in_core[core.get_id()] == -1:
                    continue
                start_time = core_combined_local_block(host.get_highest_comm_time(), core_time_list)
                if start_time < min_start_time:
                    best_core, min_start_time = core, start_time
            if min_start_time != sys.maxsize:
                padding_time = min_start_time - host.get_highest_comm_time()
                if host.last_sched_core:
                    job_index = job_in_core[host.last_sched_core.get_id()]
                    job = job_list[job_index]
                    unsched_block_num = job.padding_comm_block(padding_time, host.get_host_id(), best_core.get_id(),
                                                                                       core_time_list, unsched_block_num)
                    #unsched_block_num = job.padding_comm_block(padding_time, host.get_host_id(), best_core.get_id(),
                    #                                                                  core_time_list, unsched_block_num                                                               
                host.last_sched_core = best_core
            if best_core:
                # print(len(best_core.prep_block))
                job_index = job_in_core[best_core.get_id()]
                job = job_list[job_index]
                unsched_local_block_num, unsched_block_num = job.schedule_block(unsched_block_num,
                                                                                unnsched_local_block_num, best_core,
                                                                                host, core_time_list,
                                                                                batchJobPlans is not None)
    
                print(unsched_local_block_num, unsched_block_num)
        if unsched_local_block_num_copy == unsched_local_block_num and unsched_block_num_copy == unsched_block_num:
            break
    for core in core_list:
        pass
    print(core_time_list)
    #if batchJobPlans is not None:
    #    batchJobPlans.append(best_job.build_host_plan())
    return job_in_core      


class Host:
    def __init__(self, host_id):
        self.host_id = host_id
        self.core_list = []
        self.highest_comm_time = 0
        self.highest_start_time = 0
        self.last_sched_core = None

    def get_host_id(self):
        return self.host_id

    def  get_core_list(self):
        return self.core_list

    def get_highest_comm_time(self):
        return self.highest_comm_time

    def get_highest_start_time(self):
        return self.highest_start_time


class Core:
    def __init__(self, id, host_id):
        self.id = id
        self.host_id = host_id
        self.blocks = []
        self.localBlocks = []
        self.prep_block = []
        self.current_time_point = 0

    def get_host_id(self):
        return self.host_id

    def get_blocks(self):
        return self.blocks

    def get_id(self):
        return self.id

    def combined_local_block(self, limit, core_time_list):
        def helper(curSum, solution, index):
            if curSum > limit:
                blocks_list.append(solution)
                blocks_sum.append(curSum)
                return
            if len(blocks_list) >= 1:
                return
            for i in range(index, len(self.localBlocks)):
                if self.localBlocks[i] in solution:
                    continue
                helper(curSum + self.localBlocks[i].get_calc_time(), solution + [self.localBlocks[i]], i)
        if len(self.localBlocks) == 0:
            self.prep_block = []
            return sys.maxsize
        if core_time_list[self.id] > limit:
            self.prep_block = []
            return core_time_list[self.id]
        total_calc_time = 0
        for block in self.localBlocks:
            total_calc_time += block.get_calc_time()
        if core_time_list[self.id] + total_calc_time < limit:
            self.prep_block = self.localBlocks
            return sys.maxsize
        
        block_list = []
        block_sum = []
        helper(core_time_list[self.id], [], 0)
        best_index = block_sum.index(min(block_sum))
        self.prep_block = block_list[best_index]
        return block_sum[best_index]


    def get_current_time_point(self):
        return self.current_time_point

    def get_prep_block(self):
        return self.prep_block

class Job:
    totalSize = 0
    coreNum = 0

    def __init__(self, jobId, calcSpeed, blocks):
        self.jobId = jobId
        self.calcSpeed = calcSpeed
        self.blocks = blocks
        self.cores = [[] for i in range(TOTAL_CORE)]
        self.blockInHost = [[] for i in range(NODE_NUM)]
        self.hostSizeRatio = []
        self.hostTotalSize = [0 for i in range(NODE_NUM)]
        self.calcTime = 0

    def get_job_id(self):
        return self.jobId

    def get_calc_speed(self):
        return self.calcSpeed

    def get_calc_time(self):
        return self.calcTime

    def computeTotalBlockSize(self):
        self.totalSize = sum([self.blocks[i].get_size() for i in range(len(self.blocks))])

    def computeHostSizeRatio(self):
        for block in self.blocks:
            for host_id in block.get_hosts_id():
                self.blockInHost[host_id].append(block)
                self.hostTotalSize[host_id] += block.get_size()

        for i in range(len(self.blockInHost)):
            self.blockInHost[i].sort(key=lambda x: x.get_size(), reversed=True)

    def ComputeCommTime(self):
        for block in self.blocks:
            block.compute_comm_time()

    def computeCalcTime(self):
        self.calcTime = self.totalSize / (self.calcSpeed * JOB_FACTOR[self.coreNum - 1])
        for block in self.blocks:
            block.computeCalcTime(self.coreNum, self.calcSpeed)
        self.blocks.sort(key=lambda x: x.get_calc_time(), reverse=True)
    
    def padding_comm_block(self, padding_time, host_id, core_id, core_time_list, unsched_block_num):
        sched_time = 0
        delete_block = []
        for i in range(host_id, len(self.blockInHost)):
            block_list = self.blockInHost[i]
            for block in block_list:
                if block in delete_block:
                    continue
                calc_time = block.get_calc_time()
                comm_time = block.get_comm_time_by_host(i)
                if sched_time + calc_time + comm_time < padding_time:
                    delete_block.append(block)
                    sched_time += calc_time + comm_time
        for  i in range(0. host_id):
            block_list = self.blockInHost[i]
            for block in block_list:
                if block in delete_block:
                    continue
                calc_time = block.get_calc_time()
                comm_time = block.get_comm_time_by_host(i)
                if sched_time + calc_time + comm_time < padding_time:
                    delete_block.append(block)
                    sched_time += calc_time + comm_time
        core_time_list[core_id] += sched_time
        for block in delete_block:
            for i in range(len(self.blockInHost)):
                if block in self.blockInHost[i]:
                    self.blockInHost[i].remove(block)
                    self.hostTotalSize[i] -= block.get_size()
            unsched_block_num -= 1
        return unsched_block_num


    def schedule_block(self, unsched_block_num, unsched_local_block_num, core, host, core_time_list, in_output):
        local_block_calc_time = 0
        delete_block = []
        for block in core.get_prep_block():
            local_block_calc_time += block.get_calc_time()
            delete_block.append(block)
            if is_output:
                self.cores[core.get_id()].append(block.get_block_id())
        for block in delete_block:
            core.localBlocks.remove(block)
            unsched_local_block_num -= 1
            unsched_block_num -= 1
        core_time_list[core.get_id()] += local_block_calc_time
        best_block = None
        max_block_size = -1

        for i in range(len(self.blockInHost)):
            block_list = self.blockInHost[i]
            if len(block_list) == 0:
                continue
            block = block_list[-1]
            block_size = block.get_size()
            if block_size > max_block_size:
                best_block, max_block_size = block, block_size
        if best_block:
            calc_time = best_block.get_calc_time()
            comm_time = best_block.get_comm_time_by_host(host.get_host_id())
            host.highest_comm_time = core_time_list[core.get_id()] + comm_time
            core_time_list[core.get_id()] += calc_time + comm_time
            unsched_block_num -= 1
            for i in range(len(self.blockInHost)):
                if best_block in self.blockInHost[i]:
                    self.blockInHost[i].remove(best_block)
                    self.hostTotalSize[i] -= best_block.get_size()
        return unsched_local_block_num, unsched_block_num

    def schedule_core(self, core_list, core_in_host, job_index, job_in_core):
        max_size_host = self.hostTotalSize.index(max(self.hostTotalSize))
        ratio_limit = 1 / self.coreNum
        ratio = 0
        delete_block = []
        use_core_flag = False
        core_index = -1
        for i in range(len(self.blockInHost[max_size_host])):
            block = self.blockInHost[max_size_host][i]
            block_size_ratio = block.get_size() / self.totalSize
            if block_size_ratio + ratio > ratio_limit:
                break
            else:
                if core_in_host[max_size_host] >= CORE_PER_NODE:
                    self.hostTotalSize[max_size_host] = 0
                    break
                use_core_flag = True
                delete_block.append(block)

                ratio += block.get_size() / self.totalSize
                if core_index == -1:
                    start_core_index = max_size_host * CORE_PER_NODE
                    end_core_index = max_size_host * CORE_PER_NODE + CORE_PER_NODE
                    for j in range(start_core_index, end_core_index):
                        if job_in_core[j] == -1:
                            core_index = j
                            break
                core_list[core_index].localBlocks.append(block)
        if use_core_flag:
            core_in_host[max_size_host] += 1
            job_in_core[core_index] = job_index

        for block in delete_block:
            for i in range(len(self.blockInHost)):
                if block in self.blockInHost[i]:
                    self.blockInHost[i].remove(block)
                    self.hostTotalSize[i] -= block.get_size()

        # WF
        for block in remaining_blocks:
            worst_fit_index = -1
            worst_fit_time = sys.maxsize
            for i in best_core_index:
                if i not in sched_core_index and sched_core_num >= self.coreNum：
                    continue
                host_id = core_list[i].get_host_id()
                start_time = core_time_list[i]
                comm_time = block.get_comm_time_by_host(host_id, start_time, core_list[host_id * 20:host_id * 20 + 20])
                total_time = core_time_list[i] + comm_time + block.get_calc_time()

                if total_time <= worst_fit_time:
                    worst_fit_time, worst_fit_index = total_time, i

            core_time_list[worst_fit_index] = worst_fit_time
            if is_output:
                self.cores[worst_fit_index].append(block.get_block_id())
            self.adjust_comm_time(core_time_list, worst_fit_index, core_list, block)
            if worst_fit_index not in sched_core_index:
                core_in_host[worst_fit_index // CORE_PER_NODE] += 1
                sched_core_index.add(worst_fit_index)
                job_in_core[worst_fit_index] = self.jobId
                sched_core_num += 1

    
    def adjust_comm_time(self, core_time_list, best_index, core_list, cur_block):
        start_time = core_time_list[best_index]
        host_id = core_list[best_index].get_host_id()
        comm_end_time = start_time + cur_block.get_comm_time_by_host(host_id, start_time,
                                                                     core_list[host_id * 20:host_id * 20 + 20])
        if start_time == comm_end_time:
            return
        cur_block.startTime = start_time
        cur_block.commEndTime = comm_end_time
        for i in range(host_id * 20, host_id * 20 + 20):
            core = core_list[i]
            rollover = 0
            for block in core.get_blocks():
                block.startTime += rollover
                block.commEndTime += rollover
                if start_time <= block.startTime < comm_end_time:
                    block.commEndTime += block.get_comm_time()
                    rollover += block.get_comm_time()
            core_time_list[i] += rollover
        core_list[best_index].get_blocks().append(cur_block)

    def build_host_plan(self):
        res = {'jobId': self.jobId}
        host_plans = []
        index = 0

        for i in range(NODE_NUM):
            core_plans = []
            for j in range(CORE_PER_NODE):
                if self.cores[index]:
                    core_plans.append({"blockIds": self.cores[index]})
                index += 1
            if core_plans:
                host_plans.append({"hostId": i, "corePlans": core_plans})
        res["hostPlans"] = host_plans
        return res


class Block:
    def __init__(self, blockId, size, hostIds):
        self.blockId = blockId
        self.size = size
        self.hostIds = hostIds
        self.commTime = 0
        self.calcTime = 0
        self.startTime = 0
        self.commEndTime = 0

    def get_size(self):
        return self.size

    def compute_comm_time(self):
        self.commTime = self.size / 131072

    def get_comm_time(self):
        return self.commTime

    def get_comm_time_by_host(self, host_id):
        if host_id in self.hostIds:
            return 0
        n = 1
        for core in core_list:
            for block in core.get_blocks():
                if block.startTime <= start_time < block.commEndTime:
                    n += 1
                elif block.startTime > start_time:
                    break
        return self.commTime

    def computeCalcTime(self, n, calc_speed):
        self.calcTime = self.size / (calc_speed * CORE_FACTOR[n - 1])
        return
    
    def get_calc_time(self):
        return self.calcTime

    def get_block_id(self):
        return self.blockId
    
    def get_hosts_id(self):
        return self.hostIds


class HostPlan:
    hostId = 0
    coreNum = 0
    corePlans = []

    def __init__(self, coreNum, corePlans):
        self.coreNum = coreNum
        self.corePlans = corePlans


class Solution:
    @staticmethod
    def readFromFile(inputFileName: str) -> dict:
        with open(inputFileName, 'r') as f:
            originData = json.load(f)

        batchJobs = []

        for eachJob in originData['batchJobs']:
            job = Job(eachJob['jobId'], eachJob['calcSpeed'] * 1024 * 1024 / 1000, [])
            for eachBlock in eachJob['blocks']:
                block = Block(eachBlock['blockId'], eachBlock['size'], eachBlock['hostIds'])
                job.blocks.append(block)
            batchJobs.append(job)
        result = {
            "batchJobs": batchJobs
        }

        return result

    @staticmethod
    def writeToFile(outputFileName: str, result: dict) -> int:
        with open(outputFileName, 'w') as f:
            json.dump(result, f)
        return 0
    
    @staticmethod
    def allocateCoreForTopJob(job_list: list, core_time_list, core_list, host_list, batchJobPlans):
        a = [26, 20, 16, 8, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2]
        for i in range(len(job_list)):
            job = job_list[i]
            job.coreNum = a[i]
            job.computeCalcTime()
        placement(job_list, core_list, core_time_list, is_back_fill=False, batchJobPlans=batchJobPlans)
        return

        for i in range(len(job_list)):
            job = job_list[i]
            job.coreNum = 1
            job.computeCalcTime()
        usedCoreNum = len(job_list)

        while 1:
            max_job = None
            max_time = -1
            core_time_list_copy = [0 for _ in range(len(core_list))]
            time_list, core_job_list, _ = placement(job_list, core_list, host_list, core_time_list_copy,
                                                    batchJobPlans=None)
            time = max(time_list)
            time_index = time_list.index(time)
            jobId = core_job_list[time_index]
            best_job = None
            for job in job_list:
                if job.get_job_id() == jobId:
                    best_job = job
            if max_time < time:
                max_time, max_job = time, best_job
                print("max_job_id: " + str(max_job.get_job_id()) + "max_time: " + str(time))
            print("usedCoreNum:" + str(usedCoreNum))
            max_job.coreNum += 1
            max_job.computeCalcTime()
            usedCoreNum += 1
            if usedCoreNum == TOTAL_CORE:
                break
        placement(job_list, core_list, host_list, core_time_list, batchJobPlans=batchJobPlans)

    @staticmethod
    def backFilling(job_list: list, core_time_list: list, core_list: list, batchJobPlans):
        maxCoreTime = max(core_time_list)
        print(core_time_list)
        print(maxCoreTime)
        while 1:
            sched_job_id = []
            core_time_list_copy = copy.deepcopy(core_time_list)
            unsched_job_list = copy.copy(job_list)
            time_step = core_time_steps(core_time_list_copy)
            core_list_copy = copy.deepcopy(core_list)
            print(time_step)
            black = 0
            while len(unsched_job_list) != 0:
                sched_flag = 0

                for job in unsched_job_list:
                    job.coreNum = time_step[0][1]
                    job.computeCalcTime()
                    time_list, _, waste_time = placement([job], core_list_copy, copy.deepcopy(core_time_list_copy),
                                                        batchJobPlans=None)
                    if waste_time > black:
                        continue
                    else:
                        time = max(time_list)
                        if time <= maxCoreTime:
                            unsched_job_list.remove(job)
                            sched_job_id.append(job.get_job_id())
                            placement([job], core_list_copy, core_time_list_copy, is_back_fill=True, batchJobPlans=None)

                            sched_flag = 1
                            break
                if sched_flag:
                    time_step = core_time_steps(core_time_list_copy)
                    print(time_steps)
                else:
                    if len(time_steps) != 1:
                        core_num = time_steps[0][1]
                        time_steps[1][1] += core_num
                        black += (time_steps[1][0] - time_steps[0][0]) * core_num
                        time_step.pop(0)
                    else:
                        break
            if len(unsched_job_list) == 0:
                for job_id in sched_job_id:
                    for job in job_list:
                        if job.get_job_id() == job_id:
                            placement([job], core_list, core_time_list, is_back_fill=True,
                                      batchJobPlans=batchJobPlans)
                break
            # 在limitTime限制内装不下，就缓慢加上限100ms
            maxCoreTime += 5000
            print(maxCoreTime)
            print(maxCoreTime)

        def schedulingJob(self, batchJobs: dict) -> dict:
            global SMALL_JOB_NUM
            batchJobPlans = []
            if len(batchJobs['batchJobs']) == 20:
                SMALL_JOB_NUM = 4
                pass
            else:
                SMALL_JOB_NUM = 117

            host_list = []
            core_list = []
            index = 0
            for node_id in range(NODE_NUM):
                host = Host(node_id)
                for _ in range(CORE_PER_NODE):
                    core = Core(index, node_id)
                    core_list.append(core)
                    host.core_list.append(core)
                    index += 1
                host_list.append(host)

            job_list = []
            for job in batchJobs['batchJobs']:
                job.computeTotalBlockSize()
                job.computeHostSizeRatio()
                job.computeCommTime()
                job_list.append(job)
            job_list.sort(key=lambda x: x.totalSize / x.get_calc_speed(), reverse=True)

            core_time_list = [0.0 for _ in range(len(core_list))]

            self.allocateCoreForTopJob(job_list[:-SMALL_JOB_NUM], core_time_list, core_list, host_list, batchJobPlans)

            self.backFilling(job_list[-SMALL_JOB_NUM:], core_time_list, core_list, batchJobPlans)
            print([job_list[i].coreNum for i in range(len(job_list))])
            print(max(core_time_list))
            result = {
                "batchJobPlans": batchJobPlans
            }
            return result

        def func(self, inputFileName: str, outputFileName: str) -> int:
            # 解析输入的json文件到内存
            batchJobs = self.readFromFile(inputFileName)

            # 算法实现主体
            batchJobPlans = self.schedulingJob(batchJobs)

            # 结果写入输出的json文件
            self.writeToFile(outputFileName, batchJobPlans)

            return 0
