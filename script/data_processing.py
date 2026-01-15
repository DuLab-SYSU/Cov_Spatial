import os
import sys
import multiprocessing as mp
import math
import time
import itertools
from collections import defaultdict
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import jax
import jax.numpy as jnp
import networkx as nx

"""
Note: 
In compliance with the data-sharing policy of GISAID (https://gisaid.org/), the actual SARS-CoV-2 genome sequences used in this study have been removed from this repository.
1. The sequence data used in the analysis process has not been uploaded. 
The original sequences can be downloaded using the sequence ID numbers provided in Supplementary_Data_S2.csv and Supplementary_Data_S3.csv, 
and then processed using the data_processing.py program to obtain the sequence dataset.
2. Due to the excessive quantity, the overall sequence file metadata.tsv and the overseas file obtained after processing with the Nextclade tool were not uploaded. 
The process syntax follows Nextclade CLI Usage (https://docs.nextstrain.org/projects/nextclade/en/stable/user/nextclade-cli/usage.html)
"""

# Define the work path
os.chdir('../data/')
#######################################################################

def display_time(func):
    def wrapper(*args, **kwargs):
        funname = func.__name__
        print('******** %s ********' % funname)
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print('Time(seconds):', t2 - t1)
        print('********************')
        print('\n')
        return result
    return wrapper

def Loadseq(filename, iterator=False):
    def loadfile(filename):
        file = open(filename)
        # 使用了groupby函数用于分组，相当漂亮！
        faiter = (x[1] for x in itertools.groupby(
            file, lambda line: line[0] == ">"))
        ADict = {}
        for header in tqdm(faiter):
            str = header.__next__()[1:].strip()
            seq = "".join(s.strip() for s in faiter.__next__())
            yield (str, seq)
    f1 = loadfile(filename)  # 开始使用该函数吧！
    if iterator == True:
        return f1
    else:
        SeqDict = {}
        for i in f1:
            SeqDict[i[0]] = i[1]
        return SeqDict

def seq_shift_pool(ID, seq, number):
    shft_dct = {'A': 1, 'C': 2, 'G': 4, 'T': 8, '-': 16}
    if number == True:
        final_new_seq = np.array(
            [shft_dct[i] if i in shft_dct else 16 for i in seq.upper()]).astype(np.int8)
    elif number == False:
        final_new_seq = seq.upper()
        for i in set(list(final_new_seq)):
            if i not in ['A', 'C', 'G', 'T', '-']:
                final_new_seq = final_new_seq.replace(i, '-')
    return ID, final_new_seq

def seq_shift(seq_dct, number=True, core_num=1):
    shft_seq_dct = {}
    pool = mp.Pool(processes=core_num)
    res = pool.starmap_async(seq_shift_pool, tqdm(
        [(ID, seq, number) for ID, seq in seq_dct.items()], total=len(seq_dct))).get()
    pool.close()
    pool.join()
    for (ID, seq) in res:
        shft_seq_dct[ID] = seq
    return shft_seq_dct

def jaxapplyfunc(x):
    x = jnp.abs(x)
    return jnp.where((x >= 1) & (x <= 7), True, False)

def jax_hm_distance(carry, x):
    applyall = jnp.vectorize(jaxapplyfunc)
    res = jnp.sum(applyall(x - carry), axis=1)
    return carry, res

def jax_equal_distance(carry, x):
    res = jnp.sum(jnp.not_equal(carry, x), axis=1)
    return carry, res

@display_time
def gpu_redund(seq_dct, distance='hamming'):
    epi_lst = [i for i in seq_dct.keys()]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    seq_arr_jnp = jnp.array(np.array(list(seq_dct.values()))).astype(jnp.int8)
    if len(seq_arr_jnp) < 10000:
        split_num = 1
    else:
        split_num = 300
    print('Finished shift numpy to jaxnp')
    step = int(len(seq_arr_jnp) / split_num)
    df_mat_lst = []
    df_redund_lst = []
    for i in tqdm(range(0, len(seq_arr_jnp), step), total=split_num):
        split_index = epi_lst[i:i + step]
        if distance == 'hamming':
            res = jax.lax.scan(jax_hm_distance,
                               seq_arr_jnp[i:i + step], seq_arr_jnp)
        else:
            res = jax.lax.scan(jax_equal_distance,
                               seq_arr_jnp[i:i + step], seq_arr_jnp)
        dfi = pd.DataFrame(tuple(res)[1], columns=split_index, index=epi_lst)
        dfi['key'] = epi_lst
        df_melt = pd.melt(dfi, ['key'])
        df_filter = df_melt[df_melt.value == 0]
        df_redund_lst.append(df_filter)
    df_redund = pd.concat(df_redund_lst)
    df_redund.rename(
        columns={'key': 'source', 'variable': 'target'}, inplace=True)
    return df_redund

def CN_ExtractData_QualityControl(total_metafile,domestic_nextclade_file,domestic_nextclade_fasta):
    # Load the  metadata
    df_total = pd.read_table(total_metafile)
    # Quality control
    # Extract sequences metafile from the mainland of China and exclude those that do not meet the standards.
    df_total['date_length'] = [len(i) for i in df_total['Collection date']]
    df = df_total[
        (df_total['Is complete?'].isin([True])) &
        (~df_total['Is low coverage?'].isin([True])) &
        (~(df_total['N-Content'] > 0.05)) &
        (df_total['Sequence length'] > 29000) &
        (df_total['date_length'].isin([10])) &
        (df_total['Collection date'] >= '2019-12-30') &
        (df_total['Host'].isin(['Human']))
    ]
    df['Country'] = [loc.split('/')[1].strip() for loc in df['Location']]
    df = df.query('Country=="China"')
    df = df.sort_values(by='Collection date', ascending=True)
    df.drop_duplicates(subset=['Virus name'], keep='first', inplace=True)
    loc_shft_dct = {'Guangzhou': 'Guangdong', 'Wuhan': 'Hubei', 'Jining': 'Shandong', 'Harbin': 'Heilongjiang', 'NanChang': 'Jiangxi', 'Changde': 'Hunan',
                    "Lu'an": 'Anhui', 'Nanning': 'Guangxi', 'Tibet': 'Taiwan', 'Inner Mongolia': 'Neimenggu', 'Hangzhou': 'Zhejiang', 'South China': 'error'
                    }
    loc_lst = []
    for i, j in zip(df['Location'], df['Virus name']):
        locinfo = i.split('/')
        if len(locinfo) < 3:
            if j.split('/')[1] != 'China':
                loc_lst.append(j.split('/')[1].strip())
            else:
                loc_lst.append('error')
        else:
            loc_lst.append(locinfo[2].strip())

    df['Province'] = [loc_shft_dct[i] if i in loc_shft_dct else i for i in loc_lst]
    df = df[~df['Province'].isin(['error','Taiwan','Hong Kong','Macau'])]
    df = df[['Accession ID', 'Virus name', 'Collection date',
             'Country', 'Province', 'Pango lineage', 'Variant']]
    # Using Nextclade tool to conduct further quality control
    df_nextclade = pd.read_table(domestic_nextclade_file)
    df_nextclade = df_nextclade[(df_nextclade['qc.overallScore'] <= 30) & (
        ~df_nextclade['clade'].isin(['recombinant']))].dropna(subset=['clade', 'Nextclade_pango'])
    Id_infoset = {i:(a,b,c)  for i,a,b,c in zip(df_nextclade['seqName'],df_nextclade['clade'],df_nextclade['partiallyAliased'],df_nextclade['clade_display'])}
    df = df[df['Accession ID'].isin(df_nextclade['seqName'].tolist())]
    for n,col in zip(range(0,3),['clade', 'partiallyAliased', 'clade_display']):
        df[col] = [Id_infoset[i][n] for i in df['Accession ID']]
    print(df)
    df.to_csv('cn_meta_unredund.csv',index =False)
    # Build sequence database for the mainland of mainland.
    seq_dct = seq_shift(Loadseq(domestic_nextclade_fasta),number = False,core_num=30)
    CN_seq_dct = {i:seq_dct[i] for i in df['Accession ID'].tolist()}
    print(len(CN_seq_dct))
    with open('cn_seq_unredund.fasta', 'w') as fw:
        for x,y in CN_seq_dct.items():
            fw.write('>' + x + '\n')
            fw.write(y+ '\n')    
    # Identify redundant sequences for further processing      
    # Converting nucleotide sequences into numerical form facilitates the use of GPUs for accelerated computing.
    CN_seq_numer_dct = seq_shift(CN_seq_dct,number=True,core_num=30)
    df_redund = gpu_redund(CN_seq_numer_dct)
    df_redund['diag'] = [1 if i==j else 0 for i,j in zip(df_redund['source'],df_redund['target'])]
    df_redund = df_redund.query('diag==0')
    df_redund = df_redund.drop(['diag'], axis=1)
    df_redund.to_csv('redund.csv',index= False)
    return df,CN_seq_dct

def CN_data_processing(total_metafile,domestic_nextclade_file,domestic_nextclade_fasta):
    # Load data
    df,CN_seq_dct = CN_ExtractData_QualityControl(total_metafile,domestic_nextclade_file,domestic_nextclade_fasta)

    # Define the dominant Omicron sublineages
    lineage_dct = {}
    for i in df['Pango lineage']:
        judge_lst = []
        for lin in dominant_Omicron_sublineages:
            judge_lst.append(lin in i.strip())
        if True in judge_lst:
            lineage_dct[i] = dominant_Omicron_sublineages[judge_lst.index(True)]
        else:
            lineage_dct[i] = 'Others'
    df['Omicron sublineages'] = [lineage_dct[i] for i in df['Pango lineage']]
    df_Others = df[df['Omicron sublineages'].isin(['Others'])]
    for pango,i in zip(df_Others['Pango lineage'],df_Others['partiallyAliased']):
        judge_lst = []
        for lin in dominant_Omicron_sublineages:
            judge_lst.append(lin.strip() in i.strip())
        if True in judge_lst:
            lineage_dct[pango] = dominant_Omicron_sublineages[judge_lst.index(True)]
        else:
            lineage_dct[pango] = 'Others'
    df['Omicron sublineages'] = [lineage_dct[i] for i in df['Pango lineage']]
    df['Omicron sublineages'] = [j.split('(')[1].strip(')') if i=='Others' and j in ['21K (BA.1)','21L (BA.2)','21A (Delta)','21I (Delta)','21J (Delta)'] else i for i,j in zip(df['Omicron sublineages'],df['clade_display'])]
    Id_lineage_date_province = {i:(a,b,c) for i,a,b,c in zip(df['Accession ID'],df['Omicron sublineages'],df['Collection date'],df['Province'])}
    df_redund = pd.read_csv('redund.csv')
    df_redund['lineage_1'] = [Id_lineage_date_province[i][0] for  i in df_redund['source']]
    df_redund['lineage_2'] = [Id_lineage_date_province[i][0] for  i in df_redund['target']]
    df_redund['excl'] = [1 if i!=j else 0 for i,j in zip(df_redund['lineage_1'],df_redund['lineage_2'])]
    df_redund = df_redund.query('excl==0')
    Id_gap = {i:j.count('-') for i,j in CN_seq_dct.items()}
    Id_cluster = {}
    Id_size = {}
    G = nx.from_pandas_edgelist(df_redund)
    for n,comp in enumerate(nx.connected_components(G)):
        prov_dct = {}
        for i in comp:
            Id_cluster[i] = 'cluster_%s' % (n+1)
            prov = Id_lineage_date_province[i][2]
            prov_dct[prov] = prov_dct.get(prov,0)+1
        Id_size.update({i: prov_dct[Id_lineage_date_province[i][2]]  for i in comp}) 
    df_cluster = pd.DataFrame({'Accession ID':list(G.nodes()),
                               'clade':[Id_lineage_date_province[i][0] for i in G.nodes()],
                               'Province':[Id_lineage_date_province[i][2] for i in G.nodes()],
                               'cluster':[Id_cluster[i] for i in G.nodes()],
                               'Collection date':[Id_lineage_date_province[i][1] for i in G.nodes()],
                               'size':[Id_size[i] for i in G.nodes() ],
                               'gap':[Id_gap[i] for i in G.nodes() ],
                               })
    ance_lst,descend_lst,lineage_lst = [],[],[]
    for cluster,nodes in df_cluster.groupby('cluster')['Accession ID']:
        node_rank = sorted(list(nodes), key=lambda x: (Id_lineage_date_province[x][1],Id_size[x],Id_gap[x]))
        for i in node_rank[1:]:
            ance_lst.append(node_rank[0])
            descend_lst.append(i)
            lineage_lst.append(Id_lineage_date_province[i][0])
    df_redund_cluster = pd.DataFrame({'ancestor':ance_lst,'descend':descend_lst,'lineage':lineage_lst})
    df_redund_cluster.to_csv('redund_cluster_lineage.csv',index = False)
    df.to_csv('cn_meta_unredund.csv',index= False)
    df = df[~df['Accession ID'].isin(df_redund_cluster.descend.unique())]
    df.to_csv('cn_meta.csv',index= False)
    with open('cn_seq.fasta','w') as fw:
        for Id in df['Accession ID']:
            fw.write('>'+Id+'\n')
            fw.write(CN_seq_dct[Id]+'\n')    

def Overseas_meta_processing(total_metafile,overseas_nextclade_file):
    df_total = pd.read_table(total_metafile)
    df_total['date_length'] = [len(i) for i in df_total['Collection date']]
    df_total = df_total[
        (df_total['Is complete?'].isin([True])) &
        (~df_total['Is low coverage?'].isin([True])) &
        (~(df_total['N-Content'] > 0.05)) &
        (df_total['Sequence length'] > 29000) &
        (df_total['date_length'].isin([10])) &
        (df_total['Collection date'] >= '2019-12-30') &
        (df_total['Host'].isin(['Human']))
    ]
    df_total['country'] = [loc.split('/')[1].strip()for loc in df_total['Location']]
    df_total = df_total.query('country!="China"')
    df_total = df_total.sort_values(by='Collection date', ascending=True)
    df_total.drop_duplicates(subset=['Virus name'], keep='first', inplace=True)
    df = df_total[['Accession ID', 'Virus name', 'Collection date',
                         'country', 'Pango lineage', 'Variant']]
   
    # Using Nextclade tool to conduct further quality control
    df_nextclade = pd.read_table(overseas_nextclade_file)
    df_nextclade = df_nextclade[(df_nextclade['qc.overallScore'] <= 30) & (
        ~df_nextclade['clade'].isin(['recombinant']))].dropna(subset=['clade', 'Nextclade_pango'])
    Id_infoset = {i:(a,b,c)  for i,a,b,c in zip(df_nextclade['seqName'],df_nextclade['clade'],df_nextclade['partiallyAliased'],df_nextclade['clade_display'])}
    df = df[df['Accession ID'].isin(df_nextclade['seqName'].tolist())]
    for n,col in zip(range(0,3),['clade', 'partiallyAliased', 'clade_display']):
        df[col] = [Id_infoset[i][n] for i in df['Accession ID']]

    # Define the dominant Omicron sublineages
    lineage_dct = {}
    for i in df['Pango lineage']:
        judge_lst = []
        for lin in dominant_Omicron_sublineages:
            judge_lst.append(lin in i.strip())
        if True in judge_lst:
            lineage_dct[i] = dominant_Omicron_sublineages[judge_lst.index(True)]
        else:
            lineage_dct[i] = 'Others'
    df['Omicron sublineages'] = [lineage_dct[i] for i in df['Pango lineage']]
    df_Others = df[df['Omicron sublineages'].isin(['Others'])]
    for pango,i in zip(df_Others['Pango lineage'],df_Others['partiallyAliased']):
        judge_lst = []
        for lin in dominant_Omicron_sublineages:
            judge_lst.append(lin.strip() in i.strip())
        if True in judge_lst:
            lineage_dct[pango] = dominant_Omicron_sublineages[judge_lst.index(True)]
        else:
            lineage_dct[pango] = 'Others'
    df['Omicron sublineages'] = [lineage_dct[i] for i in df['Pango lineage']]
    df['Omicron sublineages'] = [j.split('(')[1].strip(')') if i=='Others' and j in ['21K (BA.1)','21L (BA.2)','21A (Delta)','21I (Delta)','21J (Delta)'] else i for i,j in zip(df['Omicron sublineages'],df['clade_display'])]
    df.to_csv('Overseas_total.csv', index=False)
    df_Omicron = df[df['Omicron sublineages'].isin(dominant_Omicron_sublineages)]
    df_Omicron.to_csv('Overseas_Omicron.csv', index=False)  

def Overseas_seq_processing(overseas_nextclade_fasta):
    df = pd.read_csv('Overseas_Omicron.csv')
    seq_dct = seq_shift(Loadseq(overseas_nextclade_fasta),number = False,core_num=30)
    for lineage in dominant_Omicron_sublineages:
        dfi = df[df['Omicron sublineages'].isin([lineage])]
        with open(JobPath + 'overseas_Omicron_seq/%s.fasta' % lineage, 'w') as fw:
            for x in dfi['Accession ID']:
                fw.write('>' + x + '\n')
                fw.write(seq_dct[x] + '\n')

if __name__ == "__main__":

    dominant_Omicron_sublineages = ['BA.5','BF.7','DY','XBB','EG.5', 'HK']
    prov_lst = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 
                'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 
                'Liaoning', 'Neimenggu', 'Ningxia', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 
                'Tianjin', 'Xinjiang', 'Xizang', 'Yunnan', 'Zhejiang']
    # Load data
    total_metafile = 'metadata.tsv'
    domestic_nextclade_file = 'nextclade/nextclade_domestic.tsv'
    domestic_nextclade_fasta = 'nextclade/nextclade.aligned.domestic.fasta'
    overseas_nextclade_file = 'nextclade/nextclade_overseas_Omicron.tsv'
    overseas_nextclade_fasta = 'nextclade/nextclade_Overseas_Omicron.aligned.fasta'

    # data processing
    CN_data_processing(total_metafile,domestic_nextclade_file,domestic_nextclade_fasta)
    Overseas_meta_processing(total_metafile,overseas_nextclade_file)
    Overseas_seq_processing(overseas_nextclade_fasta)


