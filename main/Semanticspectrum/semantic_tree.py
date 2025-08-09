from collections import deque
from argparse import ArgumentParser
import os


# offsets = [2,2,1,2,1]
offsets = [1,1,2,1,2,1]

class OutOfLevel(Exception):
    def __init__(self, message):
        super().__init__(message)

class Node:
    def __init__(self,domain,depth:int=0,lb:int=0,ub:int=0) -> None:
        self.childs = {}
        self.domain = domain
        self.depth = depth
        self.lb = lb
        self.ub = ub
    
    def __repr__(self):
        return f'{type(self).__name__}({self.domain}(lb:{self.lb},ub:{self.ub}))'
    
def BuildDomainQueue(domains:str) -> deque[str]:
    domain_queue = deque([])
    pre_offset = 0
    for offest in offsets:
        current_offset = pre_offset + offest
        domain_queue.append(domains[pre_offset:current_offset]) # 將每個範疇透過拆成對應的階層並儲存在 deque 中 eg: "Aa01A01=" -> deque(['Aa','01','A','01','='])
        pre_offset = current_offset
    
    return domain_queue


class SemanticSpectrum:
    def __init__(self,file_path=os.path.join(os.path.dirname(__file__),'哈大同義詞林(same)-new.txt')) -> None:
        self.vocabs = []
        self.domain_queues = []
        self.load_datas(file_path)
        self.encode_len = len(self.domain_queues[0])
        self.root = Node('_',lb=0,ub=len(self.domain_queues))
        self.Create()

    # 載入哈大同義詞林
    def load_datas(self,file_path):
        with open(file_path,encoding="utf-8-sig") as f:
            for line in f.readlines():
                elements = line.strip().split(' ')
                self.domain_queues.append(BuildDomainQueue(elements[0])) 
                self.vocabs.append(elements[1:])

    # 透過 BFS 的概念去建立這個結構
    def Create(self):
        bfs_queue = deque([self.root])
        while bfs_queue:
            root = bfs_queue.popleft()
            lb = root.lb
            ub = root.ub
            
            # 使用每一層 depth 對應的 lb、ub 去紀錄
            for i in range(lb,ub):
                # 當我的 domain queue 還有時才繼續深入
                if self.domain_queues[i]:
                    cur_domain = self.domain_queues[i].popleft()
                    depth = self.encode_len - len(self.domain_queues[i])
                    # 當目前的 domain 還未在 root childs 中，建立對應節點並記錄 lb 
                    if cur_domain not in root.childs:
                        child_node = Node(cur_domain,depth=depth,lb=i,ub=i+1)
                        root.childs[cur_domain] = child_node
                        # 將該節點加到 bfs_queue
                        bfs_queue.append(child_node)
                    else:
                        node = root.childs[cur_domain]
                        # 由於 ub 紀錄的是資料的 index value，因此在 for loop 時不會進到 ub 的 index ，因此需要 + 1 
                        node.ub = i + 1
                    
        del self.domain_queues
    
    
    def get_interval_val(self,lb,ub) -> list[str]:
        """

        Args:
            lb (int): the lower bound of data index
            ub (int): the upper bound of data index

        Returns:
            output (list[str]): interval 中所有的詞單
        """
        out:list[list[str]] = self.vocabs[lb:ub]
        # 將對應範疇的所有詞單攤平變成一個 list 儲存 
        output = [item for sublist in out for item in sublist]
        return output
        
    
    def Target_level_datas(self,level:int,scope_only:bool=False)->list|dict:
        """
        Args:
            level (int):  詞性譜第 level 層的解析度
            scope_only(bool) : if True only return 對應 level 的範疇，反之 return 範疇以及對應的詞彙
        Returns:
            scope_only = True -> level 的範疇清單
            scope_only = False -> result(dict) key: domain 對應的範疇, value: vocabs 該範疇對應的詞單
        """
        if level > 6:
            raise OutOfLevel('Maximum level is 5 !!! Please reset it')
        
        root = self.root
        def bfs(root:Node):
            
            result = {}
            bfs_queue = deque([(root,'')])
            
            while bfs_queue:
                cur_node,parent_domain = bfs_queue.popleft()
                fully_domain = parent_domain + cur_node.domain
                if cur_node.depth == level:
                    result[fully_domain] = self.get_interval_val(cur_node.lb,cur_node.ub)
                else:
                    for _,child_node in cur_node.childs.items():
                        bfs_queue.append((child_node,fully_domain))
            
            return result
        
        results = bfs(root)
        
        if scope_only:
            return list(results.keys())
        else:
            return results
                
def main():
    spectrum = SemanticSpectrum()
    parser = ArgumentParser()
    parser.add_argument('-l','--level',default=4,type=int)
    args = parser.parse_args()

    level_data = spectrum.Target_level_datas(args.level,scope_only=True)

    # print(spectrum.root.childs.get('A').childs)
    print(len(level_data))

    # Aa_node = spectrum.root.childs.get('Aa')
    # Aa01a01_node = spectrum.root.childs.get('Aa').childs.get('01').childs.get('A')
    # print(spectrum.get_interval_val(Aa01a01_node.lb,Aa01a01_node.ub),len(spectrum.get_interval_val(Aa01a01_node.lb,Aa01a01_node.ub)))
    # print(spectrum.root.childs.get('Aa'))
    
    # qq = list(filter(lambda x:len(x)<9,list(level_data.keys())))
    # print(qq)

    
if __name__ == "__main__":
    main()