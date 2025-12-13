import os
import sys
import multiprocessing as mp
import math
import time
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import networkx as nx
import glob
from data_processing import Loadseq,seq_shift_pool,seq_shift,jaxapplyfunc,jax_hm_distance,jax_equal_distance
from mutation_network import gpu_distance_mat,initial_mut_network,connected_mut_network,node_dis_to_root,directed_network,save_network
"""
Note: 
In compliance with the data-sharing policy of GISAID (https://gisaid.org/), the actual SARS-CoV-2 genome sequences used in this study have been removed from this repository.
"""

# Get the current working directory 
current_dir = os.getcwd()
print(current_dir)

# os.chdir('./data/paper/')
#######################################################################

def cdhit_extract_seq():
    prov_lst = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou',
                'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu',
                'Jiangxi', 'Jilin', 'Liaoning', 'Neimenggu', 'Ningxia', 'Qinghai',
                'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang', 'Xizang', 'Yunnan', 'Zhejiang']
    seq_dct = Loadseq('cn_seq_unredund.fasta')
    lineages = ['BA.5', 'BF.7', 'DY',
                'XBB', 'EG.5', 'HK'
                ]
    df = pd.read_csv('cn_meta_unredund.csv')
    df = df.query('Province==@prov_lst')
    for lineage in lineages:
        print('-' * 60)
        print(lineage)
        print('-' * 60)
        df_l = df[df['Omicron sublineages'].isin([lineage])]
        df_l.to_csv('sensitive_analysis/cdhit/raw/%s.csv' % lineage, index=False)

        for prov in df_l.Province.unique():
            df_pro = df_l.query('Province==@prov')
            with open('sensitive_analysis/cdhit/raw/%s/%s.fasta' % (lineage,prov), 'w') as fw:
                for x in df_pro['Accession ID']:
                    fw.write('>' + x + '\n')
                    fw.write(seq_dct[x] + '\n')

def cdhit_main(lineage,threshold=0.9995):
    files = glob.glob('sensitive_analysis/cdhit/raw/%s/*.fasta' % (lineage))
    print(len(files))
    for file in files:
        prov = file.split('/')[-1].split('.')[0]
        os.system('mkdir sensitive_analysis/cdhit/raw/%s/output/' % lineage)
        outfile = 'sensitive_analysis/cdhit/raw/%s/output/%s_%s.fasta' % (lineage,prov,threshold)
        os.system('cd-hit-est -i %s -o  %s -c %s -T 24'  % (file,outfile,threshold))

def cdhit_sample(df,total_seq_dct,lineage,sample_number):
    sample_Id_lst = []
    files =  glob.glob(r'sensitive_analysis/cdhit/raw/%s/output/*_0.9995.fasta.clstr' % lineage)
    for file in files:
        clu_dct = {}
        clu_lst = []
        adict = Loadseq(file)
        for x,y in adict.items():
            pat_Id = r"EPI_ISL_\d*"
            clu_Id_lst = re.compile(pat_Id).findall(y)
            clu_dct[x] = clu_Id_lst
            clu_lst.append(x)
        if len(clu_lst) >=sample_number:
            sample_cluster = random.sample(clu_lst, sample_number)
        else:
            sample_cluster = clu_lst
        for clu in sample_cluster:
            sample_Id = random.choice(clu_dct[clu])
            sample_Id_lst.append(sample_Id)
    df_sample = df[df['Accession ID'].isin(sample_Id_lst)]
    df_sample.to_csv('sensitive_analysis/cdhit/sample/%s.csv' % lineage, index=False)
    with open('sensitive_analysis/cdhit/sample/%s.fasta' % lineage, 'w') as fw:
        for x in df_sample['Accession ID']:
            fw.write('>' + x + '\n')
            fw.write(total_seq_dct[x] + '\n') 

def statistic_cdhit_clusters():
    dominant_Omicron_sublineages = ['BA.5','BF.7','DY','XBB','EG.5', 'HK']
    lineage_clu_pro = defaultdict(dict)
    for lineage in dominant_Omicron_sublineages:
        seqfiles = glob.glob('sensitive_analysis/cdhit/raw/%s/output/*0.9995.fasta' % lineage)
        for file in seqfiles:
            prov = file.split('/')[-1].split('_')[0]
            seq_dct = Loadseq(file)
            lineage_clu_pro[lineage].update({prov:len(seq_dct)})
    df = pd.DataFrame(lineage_clu_pro)
    prov_lst = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou',
                'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu',
                'Jiangxi', 'Jilin', 'Liaoning', 'Neimenggu', 'Ningxia', 'Qinghai',
                'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang', 'Xizang', 'Yunnan', 'Zhejiang']
    df = df.reindex(prov_lst).fillna(0)
    df.to_csv('../results/sensitive_analysis/cdhit/cdhit_clusters.csv')

def construct_network(metafile, seqfile, redundfile, outfile):
    df = pd.read_csv(metafile)
    Id_lst = list(df['Accession ID'])

    seq_dct = Loadseq(seqfile)
    df_redund = pd.read_csv(redundfile)
    redund_Id = df_redund.descend.unique().tolist()
    redund_df_ID = df[df['Accession ID'].isin(
        redund_Id)]['Accession ID'].tolist()
    df = df[~df['Accession ID'].isin(redund_Id)]
    for i in redund_df_ID:
        seq_dct.pop(i)
    df_redund = df_redund.query('ancestor==@Id_lst & descend==@Id_lst')
    epi_date = {i: j for i, j in zip(
        df['Accession ID'], df['Collection date'])}
    root = list(df.sort_values(by='Collection date',
                               ascending=True)['Accession ID'])[0]
    shft_seq_dct = seq_shift(seq_dct, number=True, core_num=core_num)
    dis_matrix = gpu_distance_mat(shft_seq_dct)
    G = initial_mut_network(dis_matrix)
    G = connected_mut_network(G, dis_matrix, core_num)
    G = directed_network(G, root, dis_matrix, epi_date, core_num)
    redund_edges = [(i, j, 0) for i, j in zip(
        df_redund['ancestor'], df_redund['descend'])]
    G.add_weighted_edges_from(redund_edges)
    df_G = nx.to_pandas_edgelist(G)
    df_G.to_csv(outfile, index=False)

def get_final_network(df,lineage, networkfile, outfile):
    Id_date_prov = {i: (a, b) for i, a, b in zip(
        df['Accession ID'], df['Collection date'], df['Province'])}
    df_global = pd.read_csv('../results/mutnet/Overseas_%s_network.csv' % lineage)
    df_cn = pd.read_csv(networkfile)
    df_cn = df_cn[~df_cn['target'].isin(df_global['target'].unique())]
    df_global.insert(4, 's_prov', 'None')
    df_cn = df_cn.assign(
        s_country='China',
        s_prov=lambda x: [Id_date_prov[i][1] for i in x["source"]],
        t_prov=lambda x: [Id_date_prov[i][1] for i in x["target"]],
        s_date=lambda x: [Id_date_prov[i][0] for i in x["source"]],
        t_date=lambda x: [Id_date_prov[i][0] for i in x["target"]]
    )
    df_net = pd.concat([df_global, df_cn])
    df_net.to_csv(outfile, index=False)

def transmission(tar_lineage, metafile, networkfile, outfile):
    df = pd.read_csv(metafile)
    df['year_month'] = ['%s-%s' %(i.split('-')[0], i.split('-')[1]) for i in df['Collection date']]
    Id_date_prov = {i: (a, b) for i, a, b in zip(
        df['Accession ID'], df['Collection date'], df['Province'])}
    df_net = pd.read_csv(networkfile)
    df_net = df_net.query('s_country=="China"')
    prov_lst = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou',
                'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu',
                'Jiangxi', 'Jilin', 'Liaoning', 'Neimenggu', 'Ningxia', 'Qinghai',
                'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang', 'Xizang', 'Yunnan', 'Zhejiang']
    prov_trans_mat = defaultdict(dict)
    for x, y in zip(df_net['s_prov'], df_net['t_prov']):
        prov_trans_mat[x][y] = prov_trans_mat[x].get(y, 0) + 1
    for x in prov_lst:
        if x not in prov_trans_mat:
            for i in prov_lst:
                prov_trans_mat[x][i] = 0
    df_mat = pd.DataFrame(prov_trans_mat)
    df_mat = df_mat.T
    df_mat = df_mat[prov_lst]
    df_mat = df_mat.reindex(prov_lst).fillna(0)
    prov_seqnum_dct = dict(df.Province.value_counts())
    for i in prov_lst:
        if i not in prov_seqnum_dct:
            prov_seqnum_dct[i] = 0
    df_mat['sum'] = [df_mat.loc[i].sum() for i in df_mat.index]
    df_mat['seqnum'] = [prov_seqnum_dct[i] for i in df_mat.index]
    df_mat['local'] = [df_mat[i][i] for i in df_mat.index]
    df_mat['import'] = [df_mat[i].sum() - df_mat['local'][i]
                        for i in df_mat.index]
    df_mat['export'] = df_mat['sum'] - df_mat['local']
    df_mat['sink_source'] = [i - (df_mat[j].sum() - df_mat['local'][j])
                             for i, j in zip(df_mat['export'], df_mat.index)]
    df_mat = df_mat.sort_values(by='export', ascending=False)
    df_mat.to_csv(outfile)


if __name__ == "__main__":

    time_A = time.time()
    core_num = 30
# ---------------------------------------------
    cdhit_extract_seq()
    dominant_Omicron_sublineages = ['BA.5','BF.7','DY','XBB','EG.5', 'HK']
    df = pd.read_csv('cn_meta_unredund.csv')
    total_seq_dct = Loadseq('cn_seq_unredund.fasta')
    sample_number = 30
    for lineage in dominant_Omicron_sublineages()
        cdhit_main(lineage,threshold=0.9995)
        cdhit_sample(df, total_seq_dct, lineage,sample_number)
        metafile = 'sensitive_analysis/cdhit/sample/%s.csv' % lineage
        seqfile =  'sensitive_analysis/cdhit/sample/%s.fasta' % lineage
        outfile =  'sensitive_analysis/cdhit/mutnet/%s_network.csv' % lineage
        outfile_finalnet = 'sensitive_analysis/cdhit/mutnet/final_%s_network.csv' % lineage
        outfile_trans = '../results/sensitive_analysis/cdhit/result/%s_trans.csv' % lineage
        redundfile = 'redund_cluster_lineage.csv'
        construct_network(metafile, seqfile, redundfile, outfile)
        get_final_network(df,lineage, outfile, outfile_finalnet)
        transmission(lineage, metafile, outfile_finalnet, outfile_trans)
# # ---------------------------------------------

    time_B = time.time()
    print('Time:', time_B - time_A)
