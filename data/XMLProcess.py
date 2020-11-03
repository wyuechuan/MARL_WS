#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xml.dom.minidom

import numpy as np


class XMLtoDAG:

    def __init__(self, file, n_task):
        self.xmlFile = file
        self.n_task = n_task        # 任务数量
        self.DAG = np.zeros((self.n_task, self.n_task), dtype=int)  # shape: 任务数*任务数

    def get_dag(self):         # 将工作流的XML解析为DAG格式
        # 使用minidom解析器打开 XML 文档
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        childrens = collection.getElementsByTagName("child")

        for child in childrens:
            child_id = child.getAttribute('ref')
            child_id = int(child_id[2:])
            # print('Child: ', child_id)
            parents = child.getElementsByTagName('parent')
            for parent in parents:
                parent_id = parent.getAttribute('ref')
                parent_id = int(parent_id[2:])
                # print(parent_id)
                self.DAG[parent_id, child_id] = 1
        return self.DAG

    def get_precursor(self):       # 寻找先驱(没有父节点)
        precursor = []
        for i in range(self.n_task):
            temp = self.DAG[:, i]    # 以子节点为索引取矩阵(整列)
            if np.sum(temp) == 0:    # 如果第i个子节点没有父节点
                precursor.append(i)
        return precursor

    def print_graph(self):
        print(self.DAG)
        for i in range(self.n_task):
            for j in range(self.n_task):
                if self.DAG[i, j] != 0:
                    print(i, ' -> ', j)


if __name__ == '__main__':
    # temps = [XMLtoDAG('Sipht_29.xml', 29), XMLtoDAG('Montage_25.xml', 25), XMLtoDAG('Inspiral_30.xml', 30),
    #          XMLtoDAG('Epigenomics_24.xml', 24), XMLtoDAG('CyberShake_30.xml', 30)]
    # for graph in temps:
    #     print(graph.get_dag())
    #     print(graph.get_precursor())
    sipht_dag = XMLtoDAG('Sipht_29.xml', 29)
    print('DAG展示：\n', sipht_dag.get_dag())
    print('Precursor展示：\n', sipht_dag.get_precursor())
    sipht_dag.print_graph()
