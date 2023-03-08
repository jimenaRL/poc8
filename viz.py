# visualizations

import seaborn as sns
import matplotlib.pyplot as plt

color_dic = {
    '0': 'blue',
    '1': 'red',
    '2': 'gold',
    '3': 'orange',
    '4': 'green',
    '5': 'violet',
    '6': 'cyan',
    '7': 'magenta',
    '8': 'brown',
    '9': 'gray'
}

extra_color_nb = max(0, ref_coords_ide['group'].nunique()-9)
palette = sns.color_palette("hls", extra_color_nb)
color_dic.update({str(k+9+1): palette[k] for k in range(extra_color_nb)})


# load users_coords_ide/att, ref_coords_ide/att and ref_group


# figure 1: ideological embedding

g = sns.jointplot(
    data=users_coords_ide.drop_duplicates(),
    x='latent_dimension_0',
    y='latent_dimension_1',
    kind="hex"
)
ax = g.ax_joint
for k in ref_coords_ide['group'].unique():
    df_k = ref_coords_ide[ref_coords_ide['group'] == k]
    ax.scatter(
        df_k['latent_dimension_0'],
        df_k['latent_dimension_1'],
        marker='+',
        s=30,
        alpha=0.5,
        color=color_dic[str(int(k))]
    )

# figure 2: attitudinale embedding

if att_model_fitted:
    ref_coords_att = ref_coords_att.merge(ref_group, how='left', on='entity')

    g = sns.jointplot(
        data=users_coords_att.drop_duplicates(),
        x='issue_1',
        y='issue_2',
        kind="hex")
    ax = g.ax_joint
    for k in ref_coords_att['group'].unique():
        df_k = ref_coords_att[ref_coords_att['group'] == k]
        df_k_mean = df_k[['issue_1', 'issue_2']].mean()
        ax.scatter(
            df_k['issue_1'],
            df_k['issue_2'],
            marker='+',
            s=30,
            alpha=0.5,
            color=color_dic[k])
        ax.plot(
            df_k_mean['issue_1'],
            df_k_mean['issue_2'],
            'o',
            mec='k',
            color=color_dic[k],
            ms=7
        )

plt.show()