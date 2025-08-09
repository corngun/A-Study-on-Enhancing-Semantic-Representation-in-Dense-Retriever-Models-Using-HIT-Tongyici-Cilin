import os
from semantic_tree import SemanticSpectrum
import json
from pathlib import Path


def main():
    semanticspec = SemanticSpectrum()
    vocab_path = Path(__file__).parent.parent.joinpath('data','vocabulary.txt')
    with open(vocab_path,'r') as f:
        total_vocabs = list(map(lambda x:x.strip(),f.readlines()))
    
    for i in range(1,7):
        scopes:dict = semanticspec.Target_level_datas(i,scope_only=False)
        mapping_table = {}
        
        for vocab in total_vocabs:
            if vocab not in mapping_table:
                mapping_table[vocab] = []
            for scope,vocabs in scopes.items():
                if vocab in vocabs :
                    if scope not in mapping_table[vocab]:
                        mapping_table[vocab].append(scope)
        
        store_path = os.path.join(os.path.dirname(__file__),'vocab_to_scope_table',f'granularity_{i}')
        if not Path(store_path).exists():
            Path(store_path).mkdir(parents=True,exist_ok=True)

        with open(Path(store_path,'result_new.json'),'w',encoding='utf-8') as f:
            json.dump(mapping_table,f,ensure_ascii=False)
            
if __name__ == '__main__':
    main()