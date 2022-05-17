import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.size'] = 18

def isnum(x):
    try:
        y = x + 1
    except TypeError:
        return False
    return True

if __name__ == '__main__':

    df = pd.concat([pd.read_csv(f'out/results/digits-{a}-{b}.csv')
        for a in [0,1,100] for b in [0,100,1234]])
    # print(df)
    nans = sum([sum(row.isnull()) for _, row in df.iterrows() 
        if any(row.isnull())])
    print(df.columns[df.isna().any()].tolist())
    print(type(df[df.columns[1]].iloc[0]))
    float_columns = len([c for c in df.columns 
        if isnum(df[c].iloc[0])])
    print(float_columns * len(df))
    print('Nans:', nans, f'({nans / len(df) * 100 / float_columns: .4f}%)')
    # df = df.dropna()
    # print(df)
    get_name = lambda s: s.split('-')[0]
    # limit to ood
    df = df[df['source'].map(get_name) != df['target'].map(get_name)]
    # print('Nans:', nans, f'({nans / len(df) * 100 : .4f}%)')
    pairs = {
         ' v. '.join(sorted([source, target])): 
        ( (df['source'].map(get_name) == source) \
            & (df['target'].map(get_name) == target) ) | \
        ( (df['source'].map(get_name) == target) \
            & (df['target'].map(get_name) == source) ) 
        for source in ['usps', 'mnist', 'svhn']
        for target in ['usps', 'mnist', 'svhn']
        if source != target
    }

    # divergence
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    divs = [('r', 'fb_pm_mdp_div'), ('g', 'fb_pm_mid_div'), 
        ('b', 'pm_mdp_div')]
    sprior = 0.01
    damps = [0.1, 0.05, 0.01, 0]
    dmap = {
        'fb_pm_mdp_div' : 'dep. (dann)',
        'fb_pm_mid_div' : 'ind. (dann)',
        'pm_mdp_div' : 'dep. (no dann)'}
    all_data = []
    for color, dname in divs:  
        for i, (title, idx) in enumerate(pairs.items()):
            all_data.append([])
            x = df[idx].copy()
            comps = []
            # stats_ub = []
            # stats_lb = []
            stats = []
            for damp in damps:
                ext = f'_{sprior}_{damp}'
                comps.append(x[f'kld{ext}'].median())
                # stats_lb.append(x[f'{dname}{ext}'].quantile(0.25))
                stats.append(x[f'{dname}{ext}'].median())
                # stats_ub.append(x[f'{dname}{ext}'].quantile(0.75))
                ax.flat[i].scatter(x[f'kld{ext}'], x[f'{dname}{ext}'],
                    color=color, s=0.3)
                for xi, yi in zip(x[f'kld{ext}'], x[f'{dname}{ext}']):
                    all_data[i].append((xi, yi))
            ax.flat[i].plot(comps, stats, label=dmap[dname], 
                color=color, lw=4)
            # ebars = abs(np.array([stats_lb, stats_ub]) - np.array(stats))
            # ax.flat[i].errorbar(comps, stats, yerr=ebars,
            #     label=dmap[dname], color=color, lw=4)
            ax.flat[i].set_xscale('log')
            ax.flat[i].set_xlabel('Complexity (KL Div.)')
            ax.flat[i].set_title(title)
            if i == 0:
                ax.flat[i].legend()
                ax.flat[i].set_ylabel('Dist. Divergence')
    plt.tight_layout()
    plt.savefig(f'div-full')
    for i, axi in enumerate(ax.flat):
        allx = [xi for xi,_ in all_data[i] if not np.isnan(xi)]
        ally = [yi for _,yi in all_data[i] if not np.isnan(yi)]
        ylb = np.quantile(ally, 0.05)
        yub = np.quantile(ally, 0.95)
        xlb = np.quantile(allx, 0.01)
        xub = np.quantile(allx, 0.96)
        axi.set_ylim((min(ylb, 0.7), 1.02))
        axi.set_xlim((xlb, xub))
    # if i == 0:
    ax.flat[0].set_xticks([100, 1000, 10_000])
    # if i == 1:
    ax.flat[1].set_xticks([1000, 10_000, 100_000])
    # if i == 2:
    ax.flat[2].set_xticks([1000, 10_000, 100_000])
    plt.tight_layout()
    plt.savefig(f'div')

    # lambda
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    sprior = 0.01
    damps = [0.1, 0.05, 0.01, 0]
    all_data = []
    for i, (title, idx) in enumerate(pairs.items()):
        all_data.append([])
        x = df[idx].copy()
        comps = []
        stats = []
        for damp in damps:
            ext = f'_{sprior}_{damp}'
            comps.append(x[f'kld{ext}'].median())
            stat = x[f'fb_sdp_idl_src_err{ext}'].median()
            stat += x[f'fb_sdp_idl_trg_err{ext}'].median()
            stats.append(stat)
            stat = x[f'fb_sdp_idl_src_err{ext}']
            stat += x[f'fb_sdp_idl_trg_err{ext}']
            ax.flat[i].scatter(x[f'kld{ext}'], stat,
                color='r', s=0.3)
            for xi, yi in zip(x[f'kld{ext}'], stat):
                all_data[i].append((xi, yi))
        # print('comps', comps)
        # print('stats', stats)
        ax.flat[i].plot(comps, stats, label='post dann', 
            color='r', lw=4)
        stat = x[f'sdp_idl_src_err'].median()
        stat += x[f'sdp_idl_trg_err'].median()
        ax.flat[i].axhline(stat, label=f'pre dann', lw=4,
            color='b')
        ax.flat[i].set_xscale('log')
        ax.flat[i].set_xlabel('Complexity (KL Div.)')
        ax.flat[i].set_title(title)
        if i == 0:
            ax.flat[i].legend()
            ax.flat[i].set_ylabel('Adaptability')
    plt.tight_layout()
    plt.savefig(f'ada-full')
    for i, axi in enumerate(ax.flat):
        allx = [xi for xi,_ in all_data[i] if not np.isnan(xi)]
        ally = [yi for _,yi in all_data[i] if not np.isnan(yi)]
        ylb = np.quantile(ally, 0.01)
        yub = np.quantile(ally, 0.96)
        xlb = np.quantile(allx, 0.01)
        xub = np.quantile(allx, 0.96)
        axi.set_ylim((ylb, yub))
        axi.set_xlim((xlb, xub))
    # if i == 0:
    ax.flat[0].set_xticks([100, 1000, 10_000])
    # if i == 1:
    ax.flat[1].set_xticks([1000, 10_000, 100_000])
    # if i == 2:
    ax.flat[2].set_xticks([1000, 10_000, 100_000])
    plt.tight_layout()
    plt.savefig(f'ada') 

    # rho
    all_data = []
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    damps = [0.1, 0]
    # smap = {0.05 : 'high', 0.025 : 'med', 0.01 : 'low'}
    spriors = [('r', 0.05), ('g', 0.025), ('b', 0.01)]
    for color, sprior in spriors:  
        for i, (title, idx) in enumerate(pairs.items()):
            all_data.append([])
            x = df[idx].copy()
            comps = []
            stats = []
            for damp in damps:
                ext = f'_{sprior}_{damp}'
                comps.append(x[f'kld{ext}'].median())
                d1 = abs(x[f'pm_trg_err{ext}'] - x[f'pst_trg_err{ext}'])
                d2 = abs(x[f'pm_src_err{ext}'] - x[f'pst_src_err{ext}'])
                stat = d1 + d2
                stats.append(stat.median())
                ax.flat[i].scatter(x[f'kld{ext}'], stat,
                    color=color, s=0.3)
                for xi, yi in zip(x[f'kld{ext}'], stat):
                    all_data[i].append((xi, yi))
            ax.flat[i].plot(comps, stats, label=sprior, 
                color=color, lw=4)
            ax.flat[i].set_xscale('log')
            ax.flat[i].set_xlabel('Complexity (KL Div.)')
            ax.flat[i].set_title(title)
            if i == 0:
                ax.flat[i].legend()
                ax.flat[i].set_ylabel('rho')
    plt.tight_layout()
    plt.savefig(f'rho-2-full')
    for i, axi in enumerate(ax.flat):
        allx = [xi for xi,_ in all_data[i] if not np.isnan(xi)]
        ally = [yi for _,yi in all_data[i] if not np.isnan(yi)]
        ylb = np.quantile(ally, 0.01)
        yub = np.quantile(ally, 0.96)
        xlb = np.quantile(allx, 0.01)
        xub = np.quantile(allx, 0.96)
        axi.set_ylim((ylb, yub))
        axi.set_xlim((xlb, xub))
    # if i == 0:
    ax.flat[0].set_xticks([100, 1000, 10_000])
    # if i == 1:
    ax.flat[1].set_xticks([1000, 10_000, 100_000])
    # if i == 2:
    ax.flat[2].set_xticks([1000, 10_000, 100_000])
    plt.tight_layout()
    plt.savefig(f'rho-2')

    # error
    pairs = {
         f'{source} to {target}' : 
        ( (df['source'].map(get_name) == source) \
            & (df['target'].map(get_name) == target) )
        for source in ['usps', 'mnist', 'svhn']
        for target in ['usps', 'mnist', 'svhn']
        if source != target
    }
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    damps = [0.1, 0]
    sprior = 0.01
    all_data = []
    for i, (title, idx) in enumerate(pairs.items()):
        all_data.append([])
        x = df[idx].copy()
        comps = []
        stats = []
        for damp in damps:
            ext = f'_{sprior}_{damp}'
            comps.append(x[f'kld{ext}'].median())
            stat = x[f'pst_trg_err{ext}']
            stats.append(stat.median())
            ax.flat[i].scatter(x[f'kld{ext}'], stat,
                color='r', s=0.3)
            for xi, yi in zip(x[f'kld{ext}'], stat):
                all_data[i].append((xi, yi))
        ax.flat[i].plot(comps, stats, label='post dann', 
            color='r', lw=4)
        stat = x[f'pri_trg_err'].median()
        ax.flat[i].axhline(stat, label=f'pre dann', lw=4,
            color='b')
        ax.flat[i].set_xscale('log')
        ax.flat[i].set_xlabel('Complexity (KL Div.)')
        ax.flat[i].set_title(title)
        if i % 3 == 0:
            if i == 0:
                ax.flat[i].legend()
            ax.flat[i].set_ylabel('Target Error')
    plt.tight_layout()
    plt.savefig(f'err-full')
    for i, axi in enumerate(ax.flat):
        allx = [xi for xi,_ in all_data[i] if not np.isnan(xi)]
        ally = [yi for _,yi in all_data[i] if not np.isnan(yi)]
        ylb = np.quantile(ally, 0.01)
        yub = np.quantile(ally, 0.96)
        xlb = np.quantile(allx, 0.01)
        xub = np.quantile(allx, 0.96)
        axi.set_ylim((ylb, yub))
        axi.set_xlim((xlb, xub))
    plt.tight_layout()
    plt.savefig(f'err')