# import numpy as np

from data.XMLProcess import XMLtoDAG
from entity.subtask import SubTask

TYPE = ['./data/Sipht_29.xml', './data/Montage_25.xml', './data/Inspiral_30.xml', './data/Epigenomics_24.xml',
        './data/CyberShake_30.xml']
NUMBER = [29, 25, 30, 24, 30]
TASK_TYPE = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 0, 1, 2, 3],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2],
             [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 1, 2],
             [0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]]


class Workflow:
    def __init__(self, num):  # 输入0-4
        self.id = num + 1           # 1-5

        self.type = TYPE[num]     # 工作流文件名
        self.size = NUMBER[num]        # 工作流任务数量
        # TASK_TYPE 任务类型，存储了每个工作流中每个任务的任务类型
        # self.SubTask存储了所有的所有工作流中任务和任务类型的对应关系[(工作流编号|任务编码), 任务类型]  工作流编码和任务编码各占两位
        self.subTask = [SubTask((num + 1) * 1000 + i + 1, TASK_TYPE[num][i]) for i in range(self.size)]  # 子任务/遍历每一个任务编码

        dag = XMLtoDAG(self.type, self.size)
        self.structure = dag.get_dag()  # 带权DAG  矩阵格式[父节点，子节点]
        self.precursor = dag.get_precursor()    # 先驱节点列表

        # print(self.precursor)
        # self.structure = np.delete(self.structure, self.precursor, 0)
        # self.structure = np.delete(self.structure, self.precursor, 1)
        # print(self.structure)
