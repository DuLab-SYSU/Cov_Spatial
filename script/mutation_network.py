import os
import sys
import time
import math
import itertools
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
import glob
import jax
import jax.numpy as jnp
from tqdm import tqdm
import networkx as nx
from data_processing import Loadseq,seq_shift_pool,seq_shift,jaxapplyfunc,jax_hm_distance,jax_equal_distance

"""
Note: 
In compliance with the data-sharing policy of GISAID (https://gisaid.org/), the actual SARS-CoV-2 genome sequences used in this study have been removed from this repository.
"""

# Define the work path
# os.chdir('../data/')

#######################################################################

def gpu_distance_mat(seq_dct, distance='hamming'):
    epi_lst = [i for i in seq_dct.keys()]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    seq_arr_jnp = jnp.array(np.array(list(seq_dct.values()))).astype(jnp.int8)
    if len(seq_arr_jnp) < 10000:
        split_num = 1
    else:
        split_num = 300
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
        df_mat_lst.append(pd.DataFrame(
            tuple(res)[1], columns=split_index, index=epi_lst))

    df_mat = pd.concat(df_mat_lst, axis=1)
    return df_mat

def gpu_distance_seqpair_mat(ref_seq_dct, seq_dct, distance='hamming'):
    ref_epi_lst = [i for i in ref_seq_dct.keys()]
    epi_lst = [i for i in seq_dct.keys()]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    ref_seq_arr_jnp = jnp.array(
        np.array(list(ref_seq_dct.values()))).astype(jnp.int8)
    if len(seq_dct) < 100000:
        split_num = 1
    else:
        split_num = 100
    step = int(len(seq_dct) / split_num)
    df_mat_lst = []
    df_redund_lst = []
    for i in range(0, len(seq_dct), step):
        split_index = epi_lst[i:i + step]
        seq_arr_jnp = jnp.array(
            np.array(list(seq_dct.values())[i:i + step])).astype(jnp.int8)
        if distance == 'hamming':
            res = jax.lax.scan(
                jax_hm_distance, ref_seq_arr_jnp, seq_arr_jnp)
        else:
            res = jax.lax.scan(jax_equal_distance,
                               ref_seq_arr_jnp, seq_arr_jnp)
        df_mat_lst.append(pd.DataFrame(
            tuple(res)[1], columns=ref_epi_lst, index=split_index))
    df_mat = pd.concat(df_mat_lst, axis=0)
    return df_mat

def initial_mut_network(dis_matrix):
    G = nx.Graph()
    matrix_v = dis_matrix.values.astype(float)
    np.fill_diagonal(matrix_v, np.inf)
    dis_matrix_new = pd.DataFrame(
        matrix_v, columns=dis_matrix.columns, index=dis_matrix.index)
    edgelst = []
    for col in dis_matrix_new.columns:
        col_min = dis_matrix_new[col].min()
        tar_index = dis_matrix_new.index[dis_matrix_new[col] == col_min].tolist(
        )
        edgelst.extend([(col, i, col_min) for i in tar_index])
    G.add_weighted_edges_from(edgelst)
    return G

def connected_mut_network(G, mut_dis_dct, core_num=20):
    while nx.is_connected(G) is False:
        print(nx.number_connected_components(G))
        Components = sorted(nx.connected_components(
            G), key=lambda x: len(x), reverse=True)
        comp_dct = {'comp_%s' % (i + 1): j for (i, j) in enumerate(Components)}
        comp_lst = list(comp_dct.keys())
        comp_connect_dct = defaultdict(dict)
        res = []
        for (comp_A, comp_B) in itertools.combinations(comp_lst, 2):
            comp_dislst = [(Ida, Idb, mut_dis_dct[Ida][Idb]) for (
                Ida, Idb) in itertools.product(comp_dct[comp_A], comp_dct[comp_B])]
            mindis = min(i[2] for i in comp_dislst)
            mindis_seqpair = (
                [(i, l)for i, l, d in comp_dislst if d == mindis], mindis)
            res.append((comp_A, comp_B, mindis_seqpair))
        for i in list(res):
            comp_connect_dct[i[0]].update({i[1]: i[2]})
            comp_connect_dct[i[1]].update({i[0]: i[2]})
        comp_newedge = []
        for comp_A, comp_disinfo in comp_connect_dct.items():
            mindis = min([i[1] for i in comp_disinfo.values()])
            for i in comp_disinfo.values():
                if i[1] == mindis:
                    comp_newedge.extend([(x[0], x[1], mindis) for x in i[0]])
        G.add_weighted_edges_from(comp_newedge)
    return G


def node_dis_to_root(G, root, node_i):
    return node_i, nx.shortest_path_length(G, root, node_i, weight='weight')

def directed_network(G, root, mut_dis_dct, epi_date, core_num=20):
    pool = mp.Pool(processes=core_num)
    res = pool.starmap_async(
        node_dis_to_root, [(G, root, node_i) for node_i in G.nodes()]).get()
    pool.close()
    pool.join()
    net_dis_to_root = {i: j for i, j in res}
    mut_dis_to_root = {i: j for i, j in mut_dis_dct[root].items()}
    rank_nodes = sorted(G.nodes(), key=lambda x: (
        net_dis_to_root[x], epi_date[x], mut_dis_to_root[x]))
    directed_edges = []
    for e in G.edges():
        new_e = tuple(sorted(e, key=lambda x: rank_nodes.index(x)))
        directed_edges.append((new_e[0], new_e[1], mut_dis_dct[e[0]][e[1]]))
    G_new = nx.DiGraph()
    G_new.add_weighted_edges_from(directed_edges)
    return G_new


def save_network(G, outfile):
    df_G = nx.to_pandas_edgelist(G)
    df_G.to_csv(outfile, index=False)

def construct_network_process(metafile, seqfile, redundfile, outpath, lineage):
    df = pd.read_csv(metafile)
    seq_dct = Loadseq(seqfile)
    df_redund = pd.read_csv(redundfile)
    exclu_Id = df[~df['Omicron sublineages'].isin([lineage])]['Accession ID'].to_list()
    df = df[~df['Accession ID'].isin(exclu_Id)]
    for i in exclu_Id:
        seq_dct.pop(i)
    df_redund = df_redund[df_redund['lineage'].isin([lineage])]
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
    print(G.number_of_nodes())
    print(G.number_of_edges())
    save_network(G, outpath + '%s_network.csv' % lineage)

def Initial_network_main(metafile,seqfile,redundfile):
    for lineage in dominant_Omicron_sublineages:
        print('-' * 60)
        print(lineage)
        outpath = '../results/mutnet/' 
        construct_network_process(metafile,
                          seqfile,
                          redundfile,
                          outpath,
                          lineage)

def Distance_matrix_with_refseq(metafile_unredund,seqfile_unredund,ref_seqfile):
    df = pd.read_csv(metafile_unredund)
    ref_seq_dct = Loadseq(ref_seqfile)
    shft_ref_seq_dct = seq_shift(ref_seq_dct, number=True, core_num=core_num)
    dis_mat_ref_lst = []
    for lineage in dominant_Omicron_sublineages:
        print('-' * 60)
        print(lineage)
        dfi = df[df['Omicron sublineages'].isin([lineage])]
        seq_dct_overseas = Loadseq('Overseas_Omicron_seq/%s.fasta' % lineage)
        shft_seq_dct_overseas = seq_shift(
            seq_dct_overseas, number=True, core_num=core_num)
        dis_mat_pair = gpu_distance_seqpair_mat(
            shft_ref_seq_dct, shft_seq_dct_overseas, distance='hamming')
        dis_mat_ref_lst.append(dis_mat_pair)
    seq_dct_cn = Loadseq(seqfile_unredund)
    shft_seq_dct_cn = seq_shift(seq_dct_cn, number=True, core_num=core_num)
    dis_mat_pair_cn = gpu_distance_seqpair_mat(shft_ref_seq_dct, shft_seq_dct_cn, distance='hamming')
    dis_mat_ref_lst.append(dis_mat_pair_cn)
    dis_mat_ref = pd.concat(dis_mat_ref_lst)
    print(dis_mat_ref)
    dis_mat_ref.to_csv('dis_mat/dis_mat_ref.csv')


def Distance_matrix_between_CN_Overseas(metafile_unredund,seqfile_unredund,Overseas_metafile):
    df = pd.read_csv(metafile_unredund)
    seq_dct_cn = Loadseq(seqfile_unredund)
    shft_seq_dct_cn = seq_shift(seq_dct_cn, number=True, core_num=core_num)
    df_Overseas = pd.read_csv(Overseas_metafile)
    for lineage in dominant_Omicron_sublineages:
        print('*' * 60)
        print(lineage)
        df_cn_i = df[df['Omicron sublineages'].isin([lineage])]
        df_Overseas_i = df_Overseas[df_Overseas['Omicron sublineages'].isin([lineage])]
        df_Overseas_i = df_Overseas.query('lineage==@lineage')
        shft_seq_dct_Overseas = seq_shift(Loadseq('Overseas_Omicron_seq/%s.fasta' % lineage), number=True, core_num=core_num)
        for Id in df_cn_i['Accession ID']:
            seq_dct_Id = {Id: shft_seq_dct_cn[Id]}
            dis_mat_pair = gpu_distance_seqpair_mat(seq_dct_Id, shft_seq_dct_Overseas, distance='hamming')
            dis_mat_pair.to_csv('dis_mat/%s/%s.csv' % (lineage, Id))

def Overseas_Imports_main(metafile_unredund,Overseas_metafile):
    df = pd.read_csv(metafile_unredund)
    df['year_month'] = ['%s-%s' %(i.split('-')[0], i.split('-')[1]) for i in df['Collection date']]
    Id_date_prov = {i:(a,b) for i,a,b in zip(df['Accession ID'],df['Collection date'],df['province'])}
    Id_mon = {i:j for i,j in zip(df['Accession ID'],df['year_month'])}
    df_Overseas = pd.read_csv(Overseas_metafile)
    Overseas_Id_date_country = {i:(a,b) for i,a,b in zip(df_Overseas['Accession ID'],df_Overseas['Collection date'],df_Overseas['country'])}
    ref_mat = pd.read_csv('dis_mat/dis_mat_ref.csv',index_col=0)
    ref_mat_dct  = {i:j for i,j in zip(ref_mat.index,ref_mat[ref_mat.columns.tolist()[0]])}
    for lineage in dominant_Omicron_sublineages:
        print('-' * 60)
        print(lineage)
        df_lineage = df[df['Omicron sublineages'].isin([lineage])]
        netfile = '../results/mutnet/%s_network.csv' % lineage
        df_net = pd.read_csv(netfile)
        df_net = df_net.assign(
            s_date=lambda x: [Id_date_prov[i][0] for i in x["source"]],
            s_prov=lambda x: [Id_date_prov[i][1] for i in x["source"]],
            s_mon=lambda x: [Id_mon[i] for i in x["source"]],
            t_date=lambda x: [Id_date_prov[i][0] for i in x["target"]],
            t_prov=lambda x: [Id_date_prov[i][1] for i in x["target"]],
            t_mon=lambda x: [Id_mon[i] for i in x["target"]],
        )
        df_net = df_net[df_net['weight'] > 0]
        Overseas_Imports_edges = []
        df_net_rm_lst = []
        for node in tqdm(df_lineage['Accession ID']):
            df_source_cn = df_net[df_net['target'].isin([node])].sort_values(['weight', 's_date'],ascending=[True, True])
            if len(df_source_cn) > 0:
                source_cn_i_info = (df_source_cn['source'].tolist()[0],df_source_cn['weight'].tolist()[0],df_source_cn['s_date'].tolist()[0])
                ref_dis_node = ref_mat_dct[node]
                Overseas_node_mat = pd.read_csv('dis_mat/%s/%s.csv' % (lineage,node),index_col = 0)
                Overseas_node_mat['s_date'] = [Overseas_Id_date_country[i][0] for i in Overseas_node_mat.index]
                Overseas_node_mat['ref_dis'] = [ref_mat_dct[i] for i in Overseas_node_mat.index]
                Overseas_node_mat = Overseas_node_mat[Overseas_node_mat['ref_dis'] <= ref_dis_node]
                if len(Overseas_node_mat)>0:
                    Overseas_node_mat = Overseas_node_mat.sort_values([node, 's_date'],ascending=[True, True])
                    source_Overseas_i_info = (Overseas_node_mat.index.tolist()[0],Overseas_node_mat[node].tolist()[0],Overseas_node_mat['s_date'].tolist()[0])
                    if source_Overseas_i_info[1] < source_cn_i_info[1]:
                        Overseas_Imports_edges.append((source_Overseas_i_info[0],node,source_Overseas_i_info[1]))
                        df_net_rm_lst.append(df_source_cn)
                    else:
                        if (source_Overseas_i_info[1] == source_cn_i_info[1]):
                            if source_Overseas_i_info[2] < source_cn_i_info[2]:
                                Overseas_Imports_edges.append((source_Overseas_i_info[0],node,source_Overseas_i_info[1]))
                                df_net_rm_lst.append(df_source_cn)
                            else:
                                pass
                        else:
                            pass
        source_lst,target_lst,weight_lst,s_country_lst,t_prov_lst,s_date_lst,t_date_lst = [],[],[], [],[],[],[]
        for (x,y ,weight) in Overseas_Imports_edges:
            source_lst.append(x)
            target_lst.append(y)
            weight_lst.append(weight)
            s_country_lst.append(Overseas_Id_date_country[x][1])
            s_date_lst.append(Overseas_Id_date_country[x][0])
            t_prov_lst.append(Id_date_prov[y][1])
            t_date_lst.append(Id_date_prov[y][0])
        df_out = pd.DataFrame({'source':source_lst,'target':target_lst,'weight':weight_lst,
                               's_country':s_country_lst,'t_prov':t_prov_lst,'s_date':s_date_lst,'t_date':t_date_lst
                                })

        df_out.to_csv('../results/mutnet/Overseas_%s_network.csv' % lineage,index =False)
        trans_dct =defaultdict(dict)
        for x,y in zip(df_out['s_country'],df_out['t_prov']):
            trans_dct[x][y] = trans_dct[x].get(y,0)+1
        df_trans = pd.DataFrame(trans_dct).T
        df_trans['sum'] =[df_trans.loc[i].sum() for i in df_trans.index]
        df_trans = df_trans.sort_values(by='sum', ascending=False)
        df_trans.to_csv('../results/Overseas_trans/Overseas_%s.csv' % lineage)

def Domestic_trans_main(metafile_unredund):
    df = pd.read_csv('cn_meta_unredund.csv')
    Id_date_prov = {i:(a,b) for i,a,b in zip(df['Accession ID'],df['Collection date'],df['Province'])}
    for lineage in dominant_Omicron_sublineages:
        print('-' * 60)
        print(lineage)
        print('-' * 60)
        df_Overseas = pd.read_csv('../results/mutnet/Overseas_%s_network.csv' % lineage)
        df_cn = pd.read_csv('../results/mutnet/%s_network.csv' % lineage)
        df_cn = df_cn[~df_cn['target'].isin(df_Overseas['target'].unique())]
        df_Overseas.insert(4,'s_prov','None')
        df_cn = df_cn.assign(
            s_country='China',
            s_prov=lambda x: [Id_date_prov[i][1] for i in x["source"]],
            t_prov=lambda x: [Id_date_prov[i][1] for i in x["target"]],
            s_date=lambda x: [Id_date_prov[i][0] for i in x["source"]],
            t_date=lambda x: [Id_date_prov[i][0] for i in x["target"]]
        )
        df_net = pd.concat([df_Overseas,df_cn])
        df_net = df_net.query('s_country=="China"')
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
        df_mat['sum'] = [df_mat.loc[i].sum() for i in df_mat.index]
        df_mat['local'] = [df_mat[i][i] for i in df_mat.index]
        df_mat['import'] = [df_mat[i].sum() - df_mat['local'][i]
                            for i in df_mat.index]
        df_mat['export'] = df_mat['sum'] - df_mat['local']
        df_mat['sink_source'] = [i - (df_mat[j].sum() - df_mat['local'][j])
                                 for i, j in zip(df_mat['export'], df_mat.index)]
        df_mat = df_mat.sort_values(by='sink_source', ascending=False)
        print(df_mat)
        df_net.to_csv('../results/mutnet/final_%s_network.csv' % lineage,index= False)
        df_mat.to_csv('../results/Domestic_trans/%s.csv' % lineage)

if __name__ == "__main__":

    time_A = time.time()
    core_num = 30
# # ---------------------------------------------

    dominant_Omicron_sublineages = ['BA.5','BF.7','DY','XBB','EG.5', 'HK']
    prov_lst = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 
                'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 
                'Liaoning', 'Neimenggu', 'Ningxia', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 
                'Tianjin', 'Xinjiang', 'Xizang', 'Yunnan', 'Zhejiang']
    # Load data
    metafile = 'cn_meta.csv' 
    seqfile = 'cn_seq.fasta'
    redundfile = 'redund_cluster_lineage.csv' 
    metafile_unredund = 'cn_meta_unredund.csv'
    seqfile_unredund = 'cn_seq_unredund.fasta'
    ref_seqfile = 'reference.fasta'
    Overseas_metafile = 'Overseas_Omicron.csv'

    # Analysis protocol
    Initial_network_main(metafile,seqfile,redundfile)
    Distance_matrix_with_refseq(metafile_unredund,seqfile_unredund,ref_seqfile)
    Distance_matrix_between_CN_Overseas(metafile_unredund,seqfile_unredund,Overseas_metafile)
    Overseas_Imports_main(metafile_unredund,Overseas_metafile)
    Domestic_trans_main(metafile_unredund)

# # ---------------------------------------------
    time_B = time.time()
    print('Time:', time_B - time_A)
